"""Inspect an existing mask without importing any inference packages."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def geometry(image):
    return {"size_xyz": list(image.GetSize()), "spacing_mm": list(image.GetSpacing()),
            "origin_lps_mm": list(image.GetOrigin()), "direction": list(image.GetDirection())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.run_dir.resolve()
    metadata = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    ct_path = Path(metadata["input_ct"])
    mask_path = out / "masks/aorta.nii.gz"
    report = {"ct": str(ct_path), "mask": str(mask_path), "status": "started",
              "anatomical_visual_review": "pending", "reference_used": False,
              "mask_modified": False, "overlays": []}
    try:
        ct, mask = sitk.ReadImage(str(ct_path)), sitk.ReadImage(str(mask_path))
        if ct.GetDimension() != 3 or mask.GetDimension() != 3:
            raise ValueError("Both images must be three-dimensional.")
        array = sitk.GetArrayViewFromImage(mask)
        labels, counts = np.unique(array, return_counts=True)
        report.update(readable=True, data_type=mask.GetPixelIDTypeAsString(),
                      numpy_dtype=str(array.dtype),
                      labels_and_counts={str(v.item()): int(n) for v, n in zip(labels, counts)},
                      finite=bool(np.isfinite(array).all()))
        tolerances = {"spacing_mm": 1e-5, "origin_lps_mm": 1e-4, "direction": 1e-6}
        cg, mg = geometry(ct), geometry(mask)
        checks = {"size_xyz": {"matches": cg["size_xyz"] == mg["size_xyz"],
                               "difference_mask_minus_ct": (np.array(mg["size_xyz"]) - cg["size_xyz"]).tolist(),
                               "absolute_tolerance": 0}}
        for key, tolerance in tolerances.items():
            delta = np.array(mg[key]) - cg[key]
            checks[key] = {"matches": bool(np.all(np.abs(delta) <= tolerance)),
                           "difference_mask_minus_ct": delta.tolist(),
                           "max_absolute_difference": float(np.max(np.abs(delta))),
                           "absolute_tolerance": tolerance, "relative_tolerance": 0}
        report.update(ct_geometry=cg, mask_geometry=mg, geometry_checks=checks,
                      geometry_matches=all(v["matches"] for v in checks.values()))
        positive = array > 0
        count = int(np.count_nonzero(positive))
        voxel_volume = float(abs(np.linalg.det(np.array(mask.GetDirection()).reshape(3, 3)))
                             * np.prod(mask.GetSpacing()))
        report.update(positive_voxel_count=count, empty=count == 0,
                      voxel_volume_mm3=voxel_volume, segmentation_volume_ml=count * voxel_volume / 1000)
        binary = sitk.Cast(mask > 0, sitk.sitkUInt8)
        components = sitk.ConnectedComponent(binary, False)
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(components)
        report.update(component_connectivity="6-neighbor (face-connected)",
                      connected_component_count=stats.GetNumberOfLabels(),
                      component_voxel_counts_descending=sorted(
                          [int(stats.GetNumberOfPixels(v)) for v in stats.GetLabels()], reverse=True))
        if not report["finite"] or count == 0:
            raise ValueError("Mask is empty or contains non-finite values.")
        if not report["geometry_matches"]:
            raise ValueError("Geometry mismatch: overlays stopped; no resampling or correction performed.")
        # Orient both verified grids to LPS. Indices then represent anatomical axes.
        oriented_ct = sitk.DICOMOrient(ct, "LPS")
        oriented_mask = sitk.DICOMOrient(mask, "LPS")
        if not np.allclose(oriented_ct.GetDirection(), np.eye(3).ravel(), atol=1e-6, rtol=0):
            raise ValueError("Oblique grid: anatomical overlays require explicit reslicing.")
        ct_array = sitk.GetArrayViewFromImage(oriented_ct)
        mask_array = sitk.GetArrayViewFromImage(oriented_mask) > 0
        spacing = oriented_ct.GetSpacing()
        specs = [("axial", 0, spacing[0], spacing[1], "R", "L", "A", "P"),
                 ("coronal", 1, spacing[0], spacing[2], "R", "L", "I", "S"),
                 ("sagittal", 2, spacing[1], spacing[2], "A", "P", "I", "S")]
        for name, axis, sx, sy, left, right, bottom, top in specs:
            # Pick the slice with the largest mask cross-section in this plane.
            totals = mask_array.sum(axis=tuple(i for i in range(3) if i != axis))
            index = int(np.argmax(totals))
            cs, ms = np.take(ct_array, index, axis=axis), np.take(mask_array, index, axis=axis)
            origin = "upper" if name == "axial" else "lower"
            h, w = cs.shape
            extent = [0, w * sx, h * sy, 0] if origin == "upper" else [0, w * sx, 0, h * sy]
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(cs, cmap="gray", vmin=-200, vmax=600, extent=extent, origin=origin, aspect="equal")
            rgba = np.zeros((*ms.shape, 4), dtype=np.float32)
            rgba[ms] = (1, 0.1, 0.1, 0.45)
            ax.imshow(rgba, extent=extent, origin=origin, aspect="equal")
            ax.set_title(f"{name.capitalize()} | LPS index {index} | anatomical review pending")
            ax.set_xlabel(f"{left} → {right} (mm)")
            ax.set_ylabel(f"{bottom} → {top} (mm)")
            fig.tight_layout()
            filename = f"overlay_{name}.png"
            fig.savefig(out / filename, dpi=160)
            plt.close(fig)
            report["overlays"].append({"file": filename, "plane": name, "lps_array_axis": axis,
                                       "index": index, "slice_positive_voxels": int(totals[index])})
        report.update(status="technical_checks_passed", display_window_stored_units=[-200, 600],
                      slice_selection="maximum positive mask cross-section per plane",
                      volume_interpretation="Volume of positive model-mask voxels; not vessel-wall volume")
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (out / "mask_check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
