"""Contract tests for the Swiss PV dataset acquisition/preparation module."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rooftop_pv.data import (
    DATASET_REF,
    SUPPORTED_CLASSES,
    build_split_manifest,
    inspect_archive,
    mask_to_yolo_lines,
    parse_site_group,
    polygon_mask_iou,
    prepare_dataset,
)


def _png_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(array.astype(np.uint8), mode="L").save(stream, format="PNG")
    return stream.getvalue()


def _jpeg_bytes() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (10, 10), color=(120, 140, 160)).save(stream, format="JPEG")
    return stream.getvalue()


def _fixture_archive(path: Path, *, unsafe_member: str | None = None) -> Path:
    image_names = [
        "swissimage-dop10_2021_2575.0-1206.0.jpg",
        "swissimage-dop10_2021_2575.1-1206.0.jpg",
        "swissimage-dop10_2021_2576.0-1207.0.jpg",
        "swissimage-dop10_2021_2577.0-1208.0.jpg",
    ]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in image_names:
            archive.writestr(f"images/{name}", _jpeg_bytes())
            mask = np.zeros((10, 10), dtype=np.uint8)
            if name.endswith("1206.0.jpg"):
                mask[2:8, 2:7] = 255
            archive.writestr(f"labels/{Path(name).stem}.png", _png_bytes(mask))
        if unsafe_member:
            archive.writestr(unsafe_member, b"must not be extracted")
    return path


def test_source_contract_has_one_explicit_class() -> None:
    assert DATASET_REF == "jeanprbt/swiss-solar-panels-segmentation"
    assert SUPPORTED_CLASSES == ("solar_panel",)


def test_parse_site_group_ignores_imagery_year() -> None:
    assert parse_site_group("swissimage-dop10_2021_2575.1-1206.3.jpg") == "2575-1206"
    assert parse_site_group("swissimage-dop10_2024_2575.1-1206.3.jpg") == "2575-1206"


def test_parse_site_group_rejects_unknown_tile_names() -> None:
    with pytest.raises(ValueError, match="SwissImage tile name"):
        parse_site_group("random.jpg")


def test_split_manifest_is_deterministic_and_group_disjoint() -> None:
    names = [
        f"swissimage-dop10_2021_{east}.0-{north}.0.jpg"
        for east, north in ((2500, 1100), (2501, 1101), (2502, 1102), (2503, 1103), (2504, 1104), (2505, 1105))
    ]
    first = build_split_manifest(names, seed=17)
    second = build_split_manifest(list(reversed(names)), seed=17)

    assert first == second
    groups = [set(first.groups[split]) for split in ("train", "val", "test")]
    assert all(groups)
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])


def test_mask_to_yolo_lines_normalizes_polygon() -> None:
    mask = np.zeros((10, 20), dtype=np.uint8)
    mask[2:8, 4:14] = 255

    lines = mask_to_yolo_lines(mask)

    assert len(lines) == 1
    fields = [float(value) for value in lines[0].split()]
    assert fields[0] == 0
    assert len(fields) == 9  # class + four (x, y) vertices
    assert all(0.0 <= value <= 1.0 for value in fields[1:])
    assert polygon_mask_iou(mask, lines) > 0.95


def test_inspect_archive_lists_only_matching_pairs(tmp_path: Path) -> None:
    archive = _fixture_archive(tmp_path / "fixture.zip")

    inventory = inspect_archive(archive)

    assert len(inventory.images) == 4
    assert len(inventory.labels) == 4
    assert inventory.unmatched_images == ()
    assert inventory.unmatched_labels == ()


def test_prepare_dataset_writes_yolo_layout_yaml_and_manifest(tmp_path: Path) -> None:
    archive = _fixture_archive(tmp_path / "fixture.zip")
    output = tmp_path / "prepared"

    prepared = prepare_dataset(archive, output, seed=17)

    assert prepared.dataset_yaml == output / "dataset.yaml"
    assert prepared.dataset_yaml.exists()
    assert (output / "images" / "train").exists()
    assert (output / "labels" / "train").exists()
    assert (output / "masks" / "train").exists()
    manifest = json.loads((output / "splits.json").read_text())
    assert manifest["source"]["dataset_ref"] == DATASET_REF
    assert set(manifest["splits"]) == {"train", "val", "test"}
    assert sum(len(values) for values in manifest["splits"].values()) == 4
    assert any((output / "labels" / split / "swissimage-dop10_2021_2575.0-1206.0.txt").exists() for split in manifest["splits"])


def test_prepare_dataset_rejects_path_traversal(tmp_path: Path) -> None:
    archive = _fixture_archive(tmp_path / "unsafe.zip", unsafe_member="../../outside.txt")

    with pytest.raises(ValueError, match="unsafe archive member"):
        prepare_dataset(archive, tmp_path / "prepared")


def test_prepare_dataset_refuses_to_delete_unrecognized_output(tmp_path: Path) -> None:
    archive = _fixture_archive(tmp_path / "fixture.zip")
    output = tmp_path / "prepared"
    output.mkdir()
    (output / "keep-me.txt").write_text("unrelated", encoding="utf-8")

    with pytest.raises(FileExistsError, match="unrecognized output"):
        prepare_dataset(archive, output, overwrite=True)
    assert (output / "keep-me.txt").read_text(encoding="utf-8") == "unrelated"
