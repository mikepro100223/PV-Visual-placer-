"""Contract tests for the RID2 obstacle acquisition and conversion pipeline."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from PIL import Image
import yaml

from rooftop_pv.obstacle_data import (
    CANONICAL_CLASSES,
    RAW_LABEL_TO_CLASS,
    RID2_UNSUPPORTED_LABELS,
    build_group_split,
    geometry_to_yolo_lines,
    parse_rid2_site_group,
    prepare_rid2_dataset,
)


def _png_bytes() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (10, 10), color=(120, 140, 160)).save(stream, format="PNG")
    return stream.getvalue()


def _fixture_archive(path: Path) -> Path:
    ids = ((1000.0, 2000.0), (11000.0, 12000.0), (21000.0, 22000.0), (31000.0, 32000.0))
    features = []
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for easting, northing in ids:
            image_id = f"{easting}_{northing}"
            image_path = f"case_study_roof_centered/images_roof_centered/{image_id}.png"
            archive.writestr(image_path, _png_bytes())
            features.append(
                {
                    "type": "Feature",
                    "properties": {"id": image_id, "image_width_px": 10, "image_height_px": 10},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [easting, northing],
                                [easting + 10, northing],
                                [easting + 10, northing + 10],
                                [easting, northing + 10],
                                [easting, northing],
                            ]
                        ],
                    },
                }
            )
        archive.writestr(
            "geometries/gdf_images_roof_centered_512_case_study.json",
            json.dumps({"type": "FeatureCollection", "features": features}),
        )
        superstructures = []
        for easting, northing in ids:
            superstructures.append(
                {
                    "type": "Feature",
                    "properties": {"label": "Chimney", "type": "polygon"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [easting + 2, northing + 2],
                                [easting + 5, northing + 2],
                                [easting + 5, northing + 5],
                                [easting + 2, northing + 5],
                                [easting + 2, northing + 2],
                            ]
                        ],
                    },
                }
            )
        archive.writestr(
            "geometries/gdf_all_superstructures.json",
            json.dumps({"type": "FeatureCollection", "features": superstructures}),
        )
    return path


def test_rid2_contract_uses_explicit_classes_and_excludes_tree_shadow() -> None:
    assert CANONICAL_CLASSES[:6] == (
        "solar_panel",
        "chimney",
        "skylight",
        "dormer",
        "roof_window",
        "hvac",
    )
    assert RAW_LABEL_TO_CLASS["PVModule"] == "solar_panel"
    assert RAW_LABEL_TO_CLASS["AC System"] == "hvac"
    assert RID2_UNSUPPORTED_LABELS == ("Shadow", "Tree")


def test_site_group_is_coarse_and_year_independent() -> None:
    assert parse_rid2_site_group("134249.59_444454.69.png") == "13-44"
    assert parse_rid2_site_group("134249.59_444454.69.tif") == "13-44"


def test_group_split_is_deterministic_and_disjoint() -> None:
    names = [
        "1000.0_2000.0.png",
        "11000.0_12000.0.png",
        "21000.0_22000.0.png",
        "31000.0_32000.0.png",
    ]
    first = build_group_split(names, seed=17)
    second = build_group_split(list(reversed(names)), seed=17)
    assert first == second
    groups = [set(first[split]["groups"]) for split in ("train", "val", "test")]
    assert all(groups)
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])


def test_geometry_to_yolo_clips_to_image_bounds() -> None:
    geometry = {
        "type": "Polygon",
        "coordinates": [[[ -2, 2], [5, 2], [5, 8], [-2, 8], [-2, 2]]],
    }
    lines = geometry_to_yolo_lines(geometry, (0, 0, 10, 10), class_id=1, width=10, height=10)
    assert len(lines) == 1
    fields = [float(value) for value in lines[0].split()]
    assert fields[0] == 1
    assert all(0.0 <= value <= 1.0 for value in fields[1:])
    assert min(fields[1::2]) == 0.0


def test_prepare_rid2_writes_yolo_layout_and_manifest(tmp_path: Path) -> None:
    archive = _fixture_archive(tmp_path / "rid2.zip")
    output = tmp_path / "prepared"
    prepared = prepare_rid2_dataset(archive, output, seed=17)

    assert prepared.dataset_yaml == output / "dataset.yaml"
    assert prepared.dataset_yaml.exists()
    assert Path(yaml.safe_load(prepared.dataset_yaml.read_text())["path"]).is_absolute()
    assert (output / "images" / "train").exists()
    assert (output / "labels" / "train").exists()
    manifest = json.loads((output / "splits.json").read_text(encoding="utf-8"))
    assert manifest["source"]["archive_verified"] is False
    assert manifest["source"]["license"] is None
    assert set(manifest["splits"]) == {"train", "val", "test"}
    assert sum(len(values["images"]) for values in manifest["splits"].values()) == 4
    assert any((output / "labels" / split / "1000.0_2000.0.txt").exists() for split in manifest["splits"])


def test_prepare_rid2_refuses_to_delete_existing_output(tmp_path: Path) -> None:
    archive = _fixture_archive(tmp_path / "rid2.zip")
    output = tmp_path / "prepared"
    output.mkdir()
    (output / "keep.txt").write_text("unrelated", encoding="utf-8")
    try:
        prepare_rid2_dataset(archive, output)
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing output should be protected")
    assert (output / "keep.txt").read_text(encoding="utf-8") == "unrelated"
