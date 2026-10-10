"""Inspect Slicer NRRD labels and physical geometry without evaluation or edits."""
import argparse
import gzip
import itertools
import json
import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def read_nrrd(path):
    header = {}
    with path.open("rb") as stream:
        magic = stream.readline().decode().strip()
        if not magic.startswith("NRRD"):
            raise ValueError("Not a NRRD file")
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("Missing NRRD header separator")
            if not line.strip():
                break
            text = line.decode("utf-8").strip()
            if text.startswith("#"):
                continue
            key, value = text.split(":=" if ":=" in text else ":", 1)
            header[key] = value.strip()
        payload = stream.read()
    if "data file" in header or "byte skip" in header or "line skip" in header:
        raise ValueError("Detached/skip NRRD layouts are not supported by this inspector")
    types = {"unsigned char": "u1", "uchar": "u1", "uint8": "u1",
             "short": "i2", "unsigned short": "u2", "int": "i4", "float": "f4", "double": "f8"}
    dtype = np.dtype(types[header["type"]])
    if dtype.itemsize > 1:
        dtype = dtype.newbyteorder("<" if header["endian"] == "little" else ">")
    if header["encoding"] in ("gzip", "gz"):
        payload = gzip.decompress(payload)
    elif header["encoding"] != "raw":
        raise ValueError("Only raw/gzip encoding supported")
    sizes = list(map(int, header["sizes"].split()))
    if len(payload) != np.prod(sizes) * dtype.itemsize:
        raise ValueError("Payload length does not match declared voxel count")
    return header, np.frombuffer(payload, dtype=dtype).reshape(sizes, order="F")


def image_geometry(image):
    return {"size_xyz": list(image.GetSize()), "origin_lps_mm": list(image.GetOrigin()),
            "basis_lps_mm": (np.array(image.GetDirection()).reshape(3, 3)
                              @ np.diag(image.GetSpacing())).tolist()}


def compare(reference, target):
    rb, tb = np.array(reference["basis_lps_mm"]), np.array(target["basis_lps_mm"])
    ro, to = np.array(reference["origin_lps_mm"]), np.array(target["origin_lps_mm"])
    rs, ts = np.array(reference["size_xyz"]), np.array(target["size_xyz"])
    offset = np.linalg.solve(tb, ro - to)
    index_transform = np.linalg.solve(tb, rb)
    corners = np.array(list(itertools.product(*[(-0.5, float(n)-0.5) for n in rs])))
    physical = ro + corners @ rb.T
    target_indices = (physical - to) @ np.linalg.inv(tb).T
    return {"same_size": bool(np.array_equal(rs, ts)),
            "origin_difference_mm": (ro-to).tolist(),
            "spacing_difference_mm": (np.linalg.norm(rb, axis=0)-np.linalg.norm(tb, axis=0)).tolist(),
            "direction_max_difference": float(np.max(np.abs(rb/np.linalg.norm(rb, axis=0)-tb/np.linalg.norm(tb, axis=0)))),
            "reference_origin_in_target_indices": offset.tolist(),
            "reference_to_target_index_linear_transform": index_transform.tolist(),
            "spacing_matches": bool(np.allclose(np.linalg.norm(rb, axis=0), np.linalg.norm(tb, axis=0), atol=1e-5, rtol=0)),
            "direction_matches": bool(np.allclose(rb/np.linalg.norm(rb, axis=0), tb/np.linalg.norm(tb, axis=0), atol=1e-6, rtol=0)),
            "origin_matches": bool(np.allclose(ro, to, atol=1e-4, rtol=0)),
            "origin_on_target_lattice": bool(np.allclose(offset, np.round(offset), atol=1e-4, rtol=0)),
            "reference_boundary_corners_lps_mm": physical.tolist(),
            "reference_bounds_in_target_indices": [target_indices.min(axis=0).tolist(), target_indices.max(axis=0).tolist()],
            "reference_within_target_extent": bool(np.all(target_indices >= -0.5-1e-4) and np.all(target_indices <= ts-0.5+1e-4)),
            "interpretation": "Size alone does not establish alignment; inspect basis, origin offset, lattice and physical extent."}


def main():
    parser = argparse.ArgumentParser()
    for name in ("ct", "reference", "prediction", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    h, array = read_nrrd(args.reference)
    dimension = int(h["dimension"])
    tokens = re.findall(r"none|\([^)]*\)", h["space directions"])
    if dimension not in (3, 4) or len(tokens) != dimension:
        raise ValueError("Expected 3 spatial axes and optional list axis")
    spatial = [i for i, v in enumerate(tokens) if v != "none"]
    nonspatial = [i for i, v in enumerate(tokens) if v == "none"]
    if len(spatial) != 3 or len(nonspatial) > 1:
        raise ValueError("Unsupported spatial/list layout")
    if nonspatial and h.get("kinds", "").split()[nonspatial[0]] != "list":
        raise ValueError("Nonspatial axis is not a declared list layer")
    basis = np.array([list(map(float, tokens[i].strip("()").split(","))) for i in spatial]).T
    origin = np.array(list(map(float, h["space origin"].strip("()").split(","))))
    if h["space"] == "right-anterior-superior":
        basis = np.diag([-1, -1, 1]) @ basis
        origin = np.diag([-1, -1, 1]) @ origin
    elif h["space"] != "left-posterior-superior":
        raise ValueError("Unsupported coordinate space")
    layers = np.moveaxis(array, nonspatial[0], 0) if nonspatial else array[None, ...]
    ref = {"size_xyz": [array.shape[i] for i in spatial], "origin_lps_mm": origin.tolist(),
           "basis_lps_mm": basis.tolist()}
    voxel_volume = float(abs(np.linalg.det(basis)))
    segments = []
    for index in sorted({int(m.group(1)) for key in h if (m := re.match(r"Segment(\d+)_", key))}):
        fields = {key.split("_", 1)[1]: value for key, value in h.items() if key.startswith(f"Segment{index}_")}
        item = {"index": index, "fields": fields,
                "missing_fields": [key for key in ("Name", "ID", "LabelValue", "Layer") if key not in fields]}
        if "Layer" in fields and "LabelValue" in fields:
            layer, value = int(fields["Layer"]), int(fields["LabelValue"])
            if not 0 <= layer < len(layers):
                raise ValueError("Segment layer out of bounds")
            selection = layers[layer] == value
            count = int(np.count_nonzero(selection))
            occupied = np.where(selection)
            item.update(voxel_count=count, volume_ml=count*voxel_volume/1000,
                        occupied_extent_xyz=([int(x.min()) for x in occupied]+[int(x.max()) for x in occupied]) if count else None)
        else:
            item.update(voxel_count=None, volume_ml=None)
        segments.append(item)
    report = {"status": "inspection_completed", "nrrd_header": h,
              "dimension": dimension, "array_shape_nrrd_axis_order": list(array.shape),
              "dtype": str(array.dtype), "layer_axis": nonspatial[0] if nonspatial else None,
              "layer_count": len(layers), "segment_count": len(segments), "segments": segments,
              "labels_per_layer": [{str(v.item()): int(n) for v, n in zip(*np.unique(layer, return_counts=True))} for layer in layers],
              "voxel_volume_mm3": voxel_volume, "reference_geometry": ref,
              "tolerances": {"spacing_mm": 1e-5, "origin_mm": 1e-4, "direction": 1e-6, "index": 1e-4, "rtol": 0},
              "geometry_comparison": {name: compare(ref, image_geometry(sitk.ReadImage(str(path)))) for name, path in (("ct", args.ct), ("prediction", args.prediction))},
              "inputs": {"ct": str(args.ct), "reference": str(args.reference), "prediction": str(args.prediction)},
              "evaluation_performed": False, "input_modified": False,
              "user_visual_observation": "Kullanıcı Slicer'da üç yönde ön inceleme yaptı; gösterilen kesitlerde ana aorta sınırlarını yakın, referansı daha fazla dallı bildirdi. Uzman doğrulaması veya ölçülmüş başarı değildir.",
              "scope_conclusion": ("Dosyada tek segment, tek katman ve tek pozitif etiket var; kullanıcının gördüğü aorta ve dallar ayrı segmentler olarak kodlanmamış, aynı birleşik etiket içindedir. Hangi dalların bulunduğu voxel etiketlerinden isimlendirilemez. Aorta terminolojisi anatomik kapsamı veya uç sınırları doğrulamaz." if len(segments) == 1 and len(layers) == 1 and len(np.unique(layers[0][layers[0] > 0])) == 1 else "Segment/layer/etiket yapısı JSON içinde raporlandı; anatomik kapsam metaveri ve bağımsız görsel değerlendirmeyle doğrulanmalıdır."),
              "required_before_comparison": "Kaynak anotasyon protokolü ve model aorta sınıfının dahil/haric damarları, proksimal/distal sınırları ve lümen/duvar tanımı netleştirilmeli. Gerekirse CT üzerinden, tahmine bakılmadan, uzman tarafından dalları ve aortayı ayrı etiketleyen bağımsız referans düzenlenmeli; ikinci gözden geçirme ve sürümleme yapılmalı. Tahmine göre kırpma veya otomatik budama yapılmamalı.",
              "format_source": "https://slicer.readthedocs.io/en/latest/developer_guide/modules/segmentations.html"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reference_inspection.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    lines = ["# AVT D1 referans incelemesi", "", f"Yapı: {dimension}D; {len(layers)} katman; {len(segments)} segment; veri tipi {array.dtype}."]
    for segment in segments:
        fields = segment['fields']
        lines.append(f"Segment {segment['index']}: ad={fields.get('Name')}, ID={fields.get('ID')}, LabelValue={fields.get('LabelValue')}, Layer={fields.get('Layer')}; voxel={segment['voxel_count']}; hacim={segment['volume_ml']} mL. Eksik alanlar: {segment['missing_fields']}.")
        lines.append(f"Tags: {fields.get('Tags')}. Terminoloji ve durum alanları doğrulama kanıtı değildir.")
    lines += [f"Etiketler: {report['labels_per_layer']}", f"Grid: {ref['size_xyz']}; voxel hacmi {voxel_volume} mm³."]
    for name, result in report['geometry_comparison'].items():
        lines.append(f"{name}: boyut eşit={result['same_size']}, spacing uyumlu={result['spacing_matches']}, origin uyumlu={result['origin_matches']}, direction uyumlu={result['direction_matches']}; fiziksel extent içinde={result['reference_within_target_extent']}. Origin farkı mm={result['origin_difference_mm']}; spacing farkı mm={result['spacing_difference_mm']}; direction en büyük fark={result['direction_max_difference']}.")
    lines += ["Toleranslar: spacing 1e-5 mm; origin 1e-4 mm; direction 1e-6; index 1e-4; göreli tolerans 0. Farklı dizi boyutu tek başına hizalama hatası sayılmaz; fiziksel origin, baz vektörleri ve extent incelendi.", report["user_visual_observation"], report["scope_conclusion"], report["required_before_comparison"], "Dice/IoU hesaplanmadı; veri ve maskeler değiştirilmedi. Hacim damar duvarı hacmi değildir."]
    (args.output_dir / "reference_report_tr.md").write_text("\n\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("dimension", "layer_count", "segment_count", "segments", "labels_per_layer", "geometry_comparison")}, indent=2))


if __name__ == "__main__":
    main()
