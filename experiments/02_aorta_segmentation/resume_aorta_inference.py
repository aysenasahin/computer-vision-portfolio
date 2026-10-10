"""Resume from the verified NIfTI; run this from normal PowerShell."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/02_aorta_segmentation/D1"
PREFLIGHT = """
from totalsegmentator.config import setup_nnunet
setup_nnunet()
from totalsegmentator.bin.TotalSegmentator import main
from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.nnunet import nnUNet_predict_image
from totalsegmentator.custom_trainers import nnUNetTrainer_MOSAIC_1k_QuarterLR_NoMirroring
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from sklearn.model_selection import KFold
from sklearn.utils._openmp_helpers import _openmp_parallelism_enabled
print('FULL_IMPORT_OK', _openmp_parallelism_enabled(), flush=True)
"""


def main() -> int:
    expected_python = ROOT / ".venv/Scripts/python.exe"
    if Path(sys.executable).resolve() != expected_python.resolve():
        raise RuntimeError(f"Use the project Python: {expected_python}")
    ct = SOURCE / "ct.nii.gz"
    check_path = SOURCE / "conversion_check.json"
    for required in (expected_python, ct, check_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    check = json.loads(check_path.read_text(encoding="utf-8"))
    if (check.get("voxel_values_identical") is not True
            or check.get("physical_geometry_identical_with_tolerance") is not True
            or Path(check["converted_file"]).resolve() != ct.resolve()):
        raise RuntimeError("Existing conversion verification does not approve this NIfTI.")

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "outputs/02_aorta_segmentation" / f"D1_resume_{stamp}_{uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    # Keep downloaded weights/config local and outside version control.
    env["TOTALSEG_HOME_DIR"] = str(ROOT / "outputs/02_aorta_segmentation/model_cache")
    env["TOTALSEG_WEIGHTS_PATH"] = str(Path(env["TOTALSEG_HOME_DIR"]) / "nnunet/results")
    Path(env["TOTALSEG_WEIGHTS_PATH"]).mkdir(parents=True, exist_ok=True)
    packages = {}
    for name in ("TotalSegmentator", "torch", "nnunetv2", "scikit-learn", "nibabel"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    record = {
        "status": "started", "started_at": datetime.now().astimezone().isoformat(),
        "python": str(expected_python), "input_ct": str(ct),
        "conversion_check": str(check_path), "conversion_repeated": False,
        "reference_segmentation_used_as_input": False,
        "settings": {"device": "cpu", "task": "total", "roi_subset": ["aorta"], "fast": True},
        "versions": packages, "mask_created": False, "stages": [],
        "model_cache": env["TOTALSEG_HOME_DIR"],
    }
    metadata_path = out / "run_metadata.json"

    def save() -> None:
        metadata_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    def run_stage(name: str, command: list[str]) -> int:
        stage = {"name": name, "command_argv": command,
                 "command": subprocess.list2cmdline(command),
                 "started_at": datetime.now().astimezone().isoformat()}
        record["stages"].append(stage)
        save()
        start = time.perf_counter()
        try:
            with (out / "terminal_output.txt").open("a", encoding="utf-8") as log:
                log.write(f"\n=== {name} ===\n{stage['command']}\n")
                log.flush()
                with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                      errors="replace") as process:
                    assert process.stdout is not None
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                    code = process.wait()
                log.write(f"\nExit code: {code}\n")
            stage["return_code"] = code
            return code
        finally:
            stage["elapsed_seconds"] = time.perf_counter() - start
            stage["finished_at"] = datetime.now().astimezone().isoformat()
            save()

    print(f"New run directory: {out}", flush=True)
    save()
    try:
        code = run_stage("import_preflight", [str(expected_python), "-u", "-c", PREFLIGHT])
        if code:
            record["status"] = "preflight_failed"
            record["return_code"] = code
            return code
        command = [str(expected_python), "-u", "-m", "totalsegmentator.bin.TotalSegmentator",
                   "-i", str(ct), "-o", str(out / "masks"), "--device", "cpu",
                   "--task", "total", "--roi_subset", "aorta", "--fast",
                   "--report", str(out / "totalsegmentator_report.json")]
        code = run_stage("inference", command)
        record["return_code"] = code
        if code:
            record["status"] = "inference_failed"
            return code
        mask = out / "masks/aorta.nii.gz"
        if not mask.is_file() or mask.stat().st_size == 0:
            record["status"] = "failed_missing_mask"
            record["return_code"] = 1
            return 1
        # File existence is reported separately from segmentation validation.
        record["status"] = "inference_completed_mask_file_present"
        record["mask_created"] = True
        record["mask_file"] = str(mask)
        record["mask_content_validated"] = False
        print(f"CLI completed; mask file exists: {mask}. Content is not yet validated.")
        return 0
    except KeyboardInterrupt:
        record["status"] = "interrupted"
        record["return_code"] = 130
        return 130
    except Exception as error:
        record["status"] = "runner_failed"
        record["error"] = f"{type(error).__name__}: {error}"
        record["return_code"] = 1
        print(record["error"], file=sys.stderr)
        return 1
    finally:
        record["finished_at"] = datetime.now().astimezone().isoformat()
        save()
        print(f"Status: {record['status']}; metadata: {metadata_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
