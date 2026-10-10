"""Inspect a 3D medical image and save geometry data and representative slices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report 3D image geometry and save axial, coronal, and sagittal slices."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input 3D image, such as a NRRD file.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for report.json and PNG images.")
    return parser.parse_args()


def save_slice(
    pixels: np.ndarray,
    output_path: Path,
    title: str,
    horizontal_label: str,
    vertical_label: str,
    horizontal_spacing: float,
    vertical_spacing: float,
    lower_percentile: float,
    upper_percentile: float,
    origin: str,
) -> None:
    height, width = pixels.shape
    if origin == "upper":
        extent = (0, width * horizontal_spacing, height * vertical_spacing, 0)
    else:
        extent = (0, width * horizontal_spacing, 0, height * vertical_spacing)

    figure, axis = plt.subplots(figsize=(8, 7))
    axis.imshow(
        pixels,
        cmap="gray",
        vmin=lower_percentile,
        vmax=upper_percentile,
        extent=extent,
        origin=origin,
        aspect="equal",
    )
    axis.set_title(title)
    axis.set_xlabel(horizontal_label)
    axis.set_ylabel(vertical_label)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {input_path}")

    image = sitk.ReadImage(str(input_path))
    if image.GetDimension() != 3:
        raise ValueError(f"Expected a 3D image; received {image.GetDimension()}D.")

    source_array_zyx = sitk.GetArrayFromImage(image)
    oriented_image = sitk.DICOMOrient(image, "LPS")
    oriented_array_zyx = sitk.GetArrayFromImage(oriented_image)
    size_xyz = image.GetSize()
    spacing_xyz = image.GetSpacing()
    oriented_size_xyz = oriented_image.GetSize()
    oriented_spacing_xyz = oriented_image.GetSpacing()

    axial_index = oriented_array_zyx.shape[0] // 2
    coronal_index = oriented_array_zyx.shape[1] // 2
    sagittal_index = oriented_array_zyx.shape[2] // 2
    slices = [
        {
            "name": "axial",
            "index_axis": "z",
            "index": axial_index,
            "pixels": oriented_array_zyx[axial_index, :, :],
            "horizontal_label": "Patient right to left (mm)",
            "vertical_label": "Anterior to posterior (mm)",
            "horizontal_spacing": oriented_spacing_xyz[0],
            "vertical_spacing": oriented_spacing_xyz[1],
            "origin": "upper",
        },
        {
            "name": "coronal",
            "index_axis": "y",
            "index": coronal_index,
            "pixels": oriented_array_zyx[:, coronal_index, :],
            "horizontal_label": "Patient right to left (mm)",
            "vertical_label": "Inferior to superior (mm)",
            "horizontal_spacing": oriented_spacing_xyz[0],
            "vertical_spacing": oriented_spacing_xyz[2],
            "origin": "lower",
        },
        {
            "name": "sagittal",
            "index_axis": "x",
            "index": sagittal_index,
            "pixels": oriented_array_zyx[:, :, sagittal_index],
            "horizontal_label": "Anterior to posterior (mm)",
            "vertical_label": "Inferior to superior (mm)",
            "horizontal_spacing": oriented_spacing_xyz[1],
            "vertical_spacing": oriented_spacing_xyz[2],
            "origin": "lower",
        },
    ]
    display_sample = np.concatenate([item["pixels"].ravel() for item in slices])
    lower_percentile, upper_percentile = np.percentile(display_sample, (1, 99)).tolist()

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    slice_results = []
    for item in slices:
        filename = f"{item['name']}.png"
        save_slice(
            pixels=item["pixels"],
            output_path=output_dir / filename,
            title=f"{item['name'].capitalize()} slice (LPS, {item['index_axis']}={item['index']})",
            horizontal_label=item["horizontal_label"],
            vertical_label=item["vertical_label"],
            horizontal_spacing=item["horizontal_spacing"],
            vertical_spacing=item["vertical_spacing"],
            lower_percentile=float(lower_percentile),
            upper_percentile=float(upper_percentile),
            origin=item["origin"],
        )
        slice_results.append(
            {
                "plane": item["name"],
                "canonical_orientation": "LPS",
                "array_axis": item["index_axis"],
                "index": item["index"],
                "file": filename,
            }
        )

    direction = image.GetDirection()
    report = {
        "input_file": str(args.input),
        "file_size_bytes": input_path.stat().st_size,
        "image_dimension": image.GetDimension(),
        "size_xyz": list(size_xyz),
        "array_shape_zyx": list(source_array_zyx.shape),
        "array_axis_order": ["z", "y", "x"],
        "spacing_xyz_mm": list(spacing_xyz),
        "origin_lps_mm": list(image.GetOrigin()),
        "direction_matrix_row_major": [
            list(direction[row * 3 : (row + 1) * 3]) for row in range(3)
        ],
        "direction_orientation": sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
            direction
        ),
        "pixel_type": image.GetPixelIDTypeAsString(),
        "intensity_min": source_array_zyx.min().item(),
        "intensity_max": source_array_zyx.max().item(),
        "display_window_percentiles": {
            "sampled_slices_percentile_1": float(lower_percentile),
            "sampled_slices_percentile_99": float(upper_percentile),
        },
        "slice_orientation": "LPS",
        "oriented_size_xyz": list(oriented_size_xyz),
        "oriented_spacing_xyz_mm": list(oriented_spacing_xyz),
        "sample_slices": slice_results,
    }
    with (output_dir / "report.json").open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, ensure_ascii=True)
        report_file.write("\n")

    print(f"Report: {output_dir / 'report.json'}")
    for item in slice_results:
        print(f"{item['plane']}: {output_dir / item['file']}")


if __name__ == "__main__":
    main()
