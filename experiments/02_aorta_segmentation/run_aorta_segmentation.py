from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pretrained TotalSegmentator aorta inference on a 3D CT."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input CT image, for example NRRD.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for conversion, masks, logs, and figures.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="auto selects CUDA when available; otherwise CPU (default: auto).",
    )
    return parser.parse_args()


def geometry(image: sitk.Image) -> dict[str, object]:
    direction = image.GetDirection()
    dimension = image.GetDimension()
    return {
        "size_xyz": list(image.GetSize()),
        "spacing_xyz_mm": list(image.GetSpacing()),
        "origin_lps_mm": list(image.GetOrigin()),
        "direction_matrix_row_major": [
            list(direction[row * dimension : (row + 1) * dimension])
            for row in range(dimension)
        ],
    }


def same_physical_grid(first: sitk.Image, second: sitk.Image) -> bool:
    return (
        first.GetDimension() == second.GetDimension()
        and first.GetSize() == second.GetSize()
        and np.allclose(first.GetSpacing(), second.GetSpacing(), rtol=0, atol=1e-5)
        and np.allclose(first.GetOrigin(), second.GetOrigin(), rtol=0, atol=1e-4)
        and np.allclose(first.GetDirection(), second.GetDirection(), rtol=0, atol=1e-6)
    )


def save_overlay(
    ct_slice: np.ndarray,
    mask_slice: np.ndarray,
    path: Path,
    title: str,
    horizontal_label: str,
    vertical_label: str,
    horizontal_spacing: float,
    vertical_spacing: float,
    window: tuple[float, float],
    origin: str,
) -> None:
    height, width = ct_slice.shape
    if origin == "upper":
        extent = (0, width * horizontal_spacing, height * vertical_spacing, 0)
    else:
        extent = (0, width * horizontal_spacing, 0, height * vertical_spacing)

    figure, axis = plt.subplots(figsize=(8, 7))
    axis.imshow(
        ct_slice,
        cmap="gray",
        vmin=window[0],
        vmax=window[1],
        extent=extent,
        origin=origin,
        aspect="equal",
    )
    overlay = np.zeros((*mask_slice.shape, 4), dtype=np.float32)
    overlay[mask_slice > 0] = (1.0, 0.1, 0.1, 0.48)
    axis.imshow(overlay, extent=extent, origin=origin, aspect="equal")
    axis.set_title(title)
    axis.set_xlabel(horizontal_label)
    axis.set_ylabel(vertical_label)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"CT input does not exist: {input_path}")
    if input_path.name.lower().endswith(".seg.nrrd"):
        raise ValueError("A segmentation file cannot be used as the model input; provide the CT image.")

    cuda_available = torch.cuda.is_available()
    if args.device == "gpu" and not cuda_available:
        raise RuntimeError("GPU was requested, but CUDA is unavailable in this Python environment.")
    device = "gpu" if args.device == "gpu" or (args.device == "auto" and cuda_available) else "cpu"

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    nifti_path = output_dir / "ct.nii.gz"
    source_image = sitk.ReadImage(str(input_path))
    if source_image.GetDimension() != 3:
        raise ValueError(f"Expected a 3D CT; received {source_image.GetDimension()}D.")

    source_array = sitk.GetArrayFromImage(source_image)
    sitk.WriteImage(source_image, str(nifti_path), useCompression=True)
    nifti_image = sitk.ReadImage(str(nifti_path))
    nifti_array = sitk.GetArrayFromImage(nifti_image)
    voxel_values_match = source_array.dtype == nifti_array.dtype and np.array_equal(
        source_array, nifti_array
    )
    geometry_matches = same_physical_grid(source_image, nifti_image)
    conversion_check = {
        "source_file": str(input_path),
        "converted_file": str(nifti_path),
        "voxel_values_identical": bool(voxel_values_match),
        "physical_geometry_identical_with_tolerance": bool(geometry_matches),
        "source_geometry": geometry(source_image),
        "nifti_geometry": geometry(nifti_image),
        "source_pixel_type": source_image.GetPixelIDTypeAsString(),
        "nifti_pixel_type": nifti_image.GetPixelIDTypeAsString(),
    }
    (output_dir / "conversion_check.json").write_text(
        json.dumps(conversion_check, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    if not voxel_values_match or not geometry_matches:
        raise RuntimeError(
            "NRRD to NIfTI verification failed; inference was not started. "
            f"See {output_dir / 'conversion_check.json'}"
        )

    executable = Path(sys.executable).parent / "TotalSegmentator.exe"
    if not executable.is_file():
        raise FileNotFoundError(f"Project environment CLI was not found: {executable}")
    command = [
        str(executable),
        "-i",
        str(nifti_path),
        "-o",
        str(mask_dir),
        "--task",
        "total",
        "--roi_subset",
        "aorta",
        "--device",
        device,
    ]
    if device == "cpu":
        command.append("--fast")
    command.extend(["--report", str(output_dir / "totalsegmentator_report.json")])
    command_text = subprocess.list2cmdline(command)
    versions = {
        package: metadata.version(package)
        for package in ("TotalSegmentator", "torch", "SimpleITK", "numpy", "matplotlib", "nibabel", "nnunetv2")
    }
    metadata_path = output_dir / "run_metadata.json"
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    print(f"Running pretrained aorta inference on {device}; output: {output_dir}", flush=True)
    start_time = time.perf_counter()
    with (output_dir / "terminal_output.txt").open("w", encoding="utf-8") as log_file:
        log_file.write(f"Command: {command_text}\n\n")
        log_file.flush()
        result = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed_seconds = time.perf_counter() - start_time
    if result.returncode != 0:
        terminal_log = output_dir / "terminal_output.txt"
        terminal_output = terminal_log.read_text(
            encoding="utf-8", errors="replace"
        )
        failure_metadata = {
            "status": "failed",
            "input_ct": str(input_path),
            "reference_segmentation_used_as_input": False,
            "converted_ct": str(nifti_path),
            "conversion_check": conversion_check,
            "command": command_text,
            "device": device,
            "cuda_available": cuda_available,
            "fast_option_used": device == "cpu",
            "task": "total",
            "roi_subset": ["aorta"],
            "started_at": started_at,
            "inference_elapsed_seconds": elapsed_seconds,
            "return_code": result.returncode,
            "versions": versions,
            "mask_expected_at": str(mask_dir / "aorta.nii.gz"),
            "mask_created": False,
            "terminal_log": str(terminal_log),
            "failure_output_tail": terminal_output[-5000:],
        }
        metadata_path.write_text(
            json.dumps(failure_metadata, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        tail = terminal_output[-5000:]
        raise RuntimeError(
            f"TotalSegmentator failed with exit code {result.returncode}. "
            f"Last terminal output:\n{tail}"
        )

    mask_path = mask_dir / "aorta.nii.gz"
    if not mask_path.is_file():
        raise FileNotFoundError(
            f"CLI completed without the expected aorta mask: {mask_path}. "
            f"See {output_dir / 'terminal_output.txt'}"
        )
    model_mask = sitk.ReadImage(str(mask_path))
    if model_mask.GetDimension() != 3:
        raise ValueError(f"Expected a 3D aorta mask; received {model_mask.GetDimension()}D.")

    mask_matches_ct_grid = same_physical_grid(nifti_image, model_mask)
    if mask_matches_ct_grid:
        aligned_mask = model_mask
        aligned_mask_path = mask_path
    else:
        aligned_mask = sitk.Resample(
            model_mask,
            nifti_image,
            sitk.Transform(3, sitk.sitkIdentity),
            sitk.sitkNearestNeighbor,
            0,
            model_mask.GetPixelID(),
        )
        aligned_mask_path = output_dir / "aorta_on_ct_grid.nii.gz"
        sitk.WriteImage(aligned_mask, str(aligned_mask_path), useCompression=True)

    labels, counts = np.unique(sitk.GetArrayFromImage(aligned_mask), return_counts=True)
    mask_array = sitk.GetArrayFromImage(aligned_mask)
    positive = mask_array > 0
    positive_voxels = int(np.count_nonzero(positive))
    voxel_volume_mm3 = float(np.prod(nifti_image.GetSpacing()))
    volume_ml = positive_voxels * voxel_volume_mm3 / 1000.0
    mask_is_empty = positive_voxels == 0

    overlay_files: list[str] = []
    slice_indices: dict[str, int] = {}
    if not mask_is_empty:
        oriented_ct = sitk.DICOMOrient(nifti_image, "LPS")
        oriented_mask = sitk.DICOMOrient(aligned_mask, "LPS")
        ct_zyx = sitk.GetArrayFromImage(oriented_ct)
        mask_zyx = sitk.GetArrayFromImage(oriented_mask)
        positive_by_axis = (
            np.flatnonzero(np.any(mask_zyx, axis=(1, 2))),
            np.flatnonzero(np.any(mask_zyx, axis=(0, 2))),
            np.flatnonzero(np.any(mask_zyx, axis=(0, 1))),
        )
        slice_indices = {
            axis: int(np.median(indices))
            for axis, indices in zip(("z", "y", "x"), positive_by_axis)
        }
        slices = [
            ("axial", "z", ct_zyx[slice_indices["z"], :, :], mask_zyx[slice_indices["z"], :, :], "Patient right to left (mm)", "Anterior to posterior (mm)", oriented_ct.GetSpacing()[0], oriented_ct.GetSpacing()[1], "upper"),
            ("coronal", "y", ct_zyx[:, slice_indices["y"], :], mask_zyx[:, slice_indices["y"], :], "Patient right to left (mm)", "Inferior to superior (mm)", oriented_ct.GetSpacing()[0], oriented_ct.GetSpacing()[2], "lower"),
            ("sagittal", "x", ct_zyx[:, :, slice_indices["x"]], mask_zyx[:, :, slice_indices["x"]], "Anterior to posterior (mm)", "Inferior to superior (mm)", oriented_ct.GetSpacing()[1], oriented_ct.GetSpacing()[2], "lower"),
        ]
        sampled_intensity = np.concatenate([item[2].ravel() for item in slices])
        window = tuple(float(value) for value in np.percentile(sampled_intensity, (1, 99)))
        for plane, axis_name, ct_slice, mask_slice, horizontal, vertical, h_spacing, v_spacing, origin in slices:
            filename = f"overlay_{plane}.png"
            save_overlay(
                ct_slice,
                mask_slice,
                output_dir / filename,
                f"Aorta on {plane} slice (LPS, {axis_name}={slice_indices[axis_name]})",
                horizontal,
                vertical,
                h_spacing,
                v_spacing,
                window,
                origin,
            )
            overlay_files.append(filename)

    run_metadata = {
        "status": "completed" if not mask_is_empty else "completed_with_empty_mask",
        "input_ct": str(input_path),
        "reference_segmentation_used_as_input": False,
        "converted_ct": str(nifti_path),
        "conversion_check": conversion_check,
        "command": command_text,
        "device": device,
        "cuda_available": cuda_available,
        "fast_option_used": device == "cpu",
        "task": "total",
        "roi_subset": ["aorta"],
        "started_at": started_at,
        "inference_elapsed_seconds": elapsed_seconds,
        "return_code": result.returncode,
        "versions": versions,
        "mask_file": str(mask_path),
        "mask_on_ct_grid_file": str(aligned_mask_path),
        "model_mask_physical_geometry_matches_ct": mask_matches_ct_grid,
        "mask_was_resampled_to_ct_grid": not mask_matches_ct_grid,
        "mask_empty": mask_is_empty,
        "labels_and_voxel_counts": {
            str(int(label)): int(count) for label, count in zip(labels, counts)
        },
        "positive_voxel_count": positive_voxels,
        "voxel_volume_mm3": voxel_volume_mm3,
        "segmentation_volume_ml": volume_ml,
        "overlay_slices_lps": slice_indices,
        "overlay_files": overlay_files,
        "terminal_log": str(output_dir / "terminal_output.txt"),
    }
    metadata_path.write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"Aorta mask: {mask_path}")
    print(f"Device: {device}; inference elapsed: {elapsed_seconds:.1f} seconds")
    print(f"Mask labels: {run_metadata['labels_and_voxel_counts']}")
    print(f"Positive voxels: {positive_voxels}; volume: {volume_ml:.3f} mL")
    print(f"Metadata: {output_dir / 'run_metadata.json'}")
    if mask_is_empty:
        raise RuntimeError("TotalSegmentator produced an empty aorta mask; no overlay slices were generated.")


if __name__ == "__main__":
    main()
