"""Descriptive comparison only: no inference, editing, Dice or IoU."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import SimpleITK as sitk

COLORS = {"common": (0.1, 0.85, 0.25), "prediction_only": (1, 0.55, 0.05),
          "reference_only": (0.1, 0.65, 1)}
LABELS = {"common": "Ortak", "prediction_only": "Yalnız tahmin", "reference_only": "Yalnız referans"}


def geometry(image):
    return {"size": list(image.GetSize()), "spacing": list(image.GetSpacing()),
            "origin": list(image.GetOrigin()), "direction": list(image.GetDirection())}


def check_grid(ct, image):
    first, second = geometry(ct), geometry(image)
    result = {"size_matches": first["size"] == second["size"], "geometry": second}
    for key, tolerance in (("spacing", 1e-5), ("origin", 1e-4), ("direction", 1e-6)):
        delta = np.array(second[key]) - first[key]
        result[key] = {"difference_mask_minus_ct": delta.tolist(), "absolute_tolerance": tolerance,
                       "rtol": 0, "matches": bool(np.all(np.abs(delta) <= tolerance))}
    result["matches"] = result["size_matches"] and all(result[k]["matches"] for k in ("spacing", "origin", "direction"))
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("ct", "prediction", "reference", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    report = {"status": "started", "inputs": {k: str(getattr(args, k)) for k in ("ct", "prediction", "reference")},
              "interpretation": "Betimleyici karşılaştırma; kapsamlar eşleştirilmedi. Bölge farkları hata sınıfları değildir.",
              "colors": {k: {"label": LABELS[k], "rgb": v} for k, v in COLORS.items()},
              "mask_modified": False, "resampled": False, "dice_iou_computed": False, "figures": []}
    try:
        ct, pred, ref = [sitk.ReadImage(str(getattr(args, key))) for key in ("ct", "prediction", "reference")]
        if any(i.GetDimension() != 3 for i in (ct, pred, ref)):
            raise ValueError("This comparison requires 3D masks; layer extraction must be explicit.")
        report["ct_geometry"] = geometry(ct)
        report["geometry_checks"] = {"prediction": check_grid(ct, pred), "reference": check_grid(ct, ref)}
        if not all(v["matches"] for v in report["geometry_checks"].values()):
            raise ValueError("Physical grids differ; direct array comparison stopped. No resampling performed.")
        for name, image in (("prediction", pred), ("reference", ref)):
            values = np.unique(sitk.GetArrayViewFromImage(image))
            report[name+"_labels"] = values.tolist()
            if not set(values.tolist()).issubset({0, 1}):
                raise ValueError("Expected inspected binary labels 0/1; refusing implicit segment merging.")
        # Verified near-identical grids are oriented together. No data is resampled.
        ct, pred, ref = [sitk.DICOMOrient(i, "LPS") for i in (ct, pred, ref)]
        if not np.allclose(ct.GetDirection(), np.eye(3).ravel(), atol=1e-6, rtol=0):
            raise ValueError("Oblique grid needs explicit anatomical reslicing; comparison stopped.")
        ca = sitk.GetArrayViewFromImage(ct)
        pa, ra = sitk.GetArrayViewFromImage(pred) > 0, sitk.GetArrayViewFromImage(ref) > 0
        regions = {"common": pa & ra, "prediction_only": pa & ~ra, "reference_only": ra & ~pa}
        counts = {k: int(np.count_nonzero(v)) for k, v in regions.items()}
        ptotal, rtotal = int(pa.sum()), int(ra.sum())
        checks = {"prediction_identity": counts["common"]+counts["prediction_only"] == ptotal,
                  "reference_identity": counts["common"]+counts["reference_only"] == rtotal,
                  "regions_disjoint": not bool(np.any(regions["prediction_only"] & regions["reference_only"]))}
        if not all(checks.values()):
            raise ValueError("Count identities failed")
        voxel_ml = float(abs(np.linalg.det(np.array(ct.GetDirection()).reshape(3, 3))) * np.prod(ct.GetSpacing()) / 1000)
        report.update(regions={k: {"voxels": n, "volume_ml": n*voxel_ml} for k, n in counts.items()},
                      totals={"prediction_voxels": ptotal, "reference_voxels": rtotal},
                      consistency_checks=checks, voxel_volume_ml=voxel_ml,
                      volume_basis="All region volumes use the CT grid; tiny NRRD/NIfTI spacing rounding differences are within tolerance.")
        slice_counts = {k: v.sum(axis=(1, 2)) for k, v in regions.items()}
        with (out / "axial_region_counts.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["lps_z_index", "physical_s_lps_mm", *regions, "prediction_total", "reference_total"])
            for z in range(ca.shape[0]):
                row = [int(slice_counts[k][z]) for k in regions]
                writer.writerow([z, ct.TransformIndexToPhysicalPoint((0, 0, z))[2], *row, row[0]+row[1], row[0]+row[2]])
        report["consistency_checks"]["csv_sums_match_volume"] = all(int(slice_counts[k].sum()) == counts[k] for k in regions)
        active = np.flatnonzero((pa | ra).sum(axis=(1, 2)))
        if not len(active):
            raise ValueError("Both masks empty")
        selected = sorted(set(active[np.rint(np.linspace(0, len(active)-1, 9)).astype(int)].tolist()
                              + [int(np.argmax(slice_counts[k])) for k in regions if counts[k] > 0]))
        report["selection_method"] = "Axial: 9 evenly spaced ranks among union-positive slices, plus each region's maximum-area slice, deduplicated. Coronal/sagittal: maximum union area. Not an exhaustive anatomical assessment."
        spacing = ct.GetSpacing()

        def draw(ax, axis, index):
            name = ("axial", "coronal", "sagittal")[axis]
            cs = np.take(ca, index, axis=axis)
            sx, sy = ((spacing[0], spacing[1]), (spacing[0], spacing[2]), (spacing[1], spacing[2]))[axis]
            h, w = cs.shape
            origin = "upper" if axis == 0 else "lower"
            extent = [0, w*sx, h*sy, 0] if axis == 0 else [0, w*sx, 0, h*sy]
            ax.imshow(cs, cmap="gray", vmin=-200, vmax=600, extent=extent, origin=origin, aspect="equal")
            rgba = np.zeros((*cs.shape, 4), np.float32)
            for key, region in regions.items():
                rgba[np.take(region, index, axis=axis)] = (*COLORS[key], 0.65)
            ax.imshow(rgba, extent=extent, origin=origin, aspect="equal")
            xyz = [0, 0, 0]
            xyz[2-axis] = index
            physical = list(ct.TransformIndexToPhysicalPoint(tuple(xyz)))
            ax.set_title(f"{name} index={index}\nLPS {'SPL'[axis]}={physical[2-axis]:.2f} mm")
            ax.set_xlabel(("R → L", "R → L", "A → P")[axis]+" (mm)")
            ax.set_ylabel(("A → P", "I → S", "I → S")[axis]+" (mm)")
            return {"plane": name, "index": index, "index_space": "LPS-oriented CT", "physical_anchor_lps_mm": physical,
                    "plane_coordinate_lps_mm": physical[2-axis]}

        legend = [Patch(color=COLORS[k], label=LABELS[k]) for k in regions]
        rows = int(np.ceil(len(selected)/3))
        fig, axes = plt.subplots(rows, 3, figsize=(15, rows*6), squeeze=False)
        slices = [draw(ax, 0, index) for ax, index in zip(axes.flat, selected)]
        for ax in list(axes.flat)[len(selected):]:
            ax.axis("off")
        fig.legend(handles=legend, loc="lower center", ncol=3)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(out / "axial_comparison_gallery.png", dpi=130)
        plt.close(fig)
        report["figures"].append({"file": "axial_comparison_gallery.png", "slices": slices})
        union = pa | ra
        for axis, name in ((1, "coronal"), (2, "sagittal")):
            index = int(np.argmax(union.sum(axis=tuple(i for i in range(3) if i != axis))))
            fig, ax = plt.subplots(figsize=(9, 10))
            info = draw(ax, axis, index)
            fig.legend(handles=legend, loc="lower center", ncol=3)
            fig.tight_layout(rect=(0, 0.04, 1, 1))
            filename = f"comparison_{name}.png"
            fig.savefig(out / filename, dpi=160)
            plt.close(fig)
            report["figures"].append({"file": filename, "slices": [info]})
        report.update(status="descriptive_comparison_completed", display_window_stored_units=[-200, 600])
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (out / "comparison_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
