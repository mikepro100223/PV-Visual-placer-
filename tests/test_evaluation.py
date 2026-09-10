import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from rooftop_pv.evaluation import (
    aggregate_pixel_scores,
    dataset_manifest_hashes,
    mask_scores,
    native_mask_metrics,
    per_class_pixel_metrics,
    quality_gate,
    read_labels,
    reference_mask,
    role_union_metrics,
)
from rooftop_pv.inference import Detection


def test_area_score_includes_negative_false_positives():
    scores = mask_scores(np.zeros((10, 10), dtype=bool), np.ones((10, 10), dtype=bool))
    assert scores["iou"] == 0
    assert scores["false_positive_pixels"] == 100
    assert scores["relative_area_error"] is None


def test_empty_agreement_is_well_defined():
    scores = mask_scores(np.zeros((3, 3), dtype=bool), np.zeros((3, 3), dtype=bool))
    assert scores["iou"] == 1


def test_missing_metric_fails_gate():
    assert not quality_gate({"mask_map50": .99, "mask_map50_95": .8})["passed"]
    assert not quality_gate({"mask_map50": .99, "mask_map50_95": .8,
                             "union_iou": float("nan")})["passed"]


def test_multiclass_quality_gate_requires_obstacle_union_iou():
    metrics = {"mask_map50": .9, "mask_map50_95": .8, "union_iou": .9,
               "obstacle_union_iou": .5}

    gate = quality_gate(metrics, require_obstacle=True)

    assert not gate["passed"]
    assert gate["checks"]["obstacle_union_iou"] == {
        "value": .5, "minimum": .65, "passed": False,
    }


def test_yolo_detection_boxes_are_not_segmentation_labels(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("0 .5 .5 .25 .25\n")
    with pytest.raises(ValueError, match="segmentation"):
        read_labels(path, 100, 100)


def test_missing_labels_are_not_silently_negative(tmp_path):
    with pytest.raises(ValueError, match="Missing label"):
        read_labels(tmp_path / "missing.txt", 100, 100)


def test_area_metrics_keep_original_holes_in_semantic_masks(tmp_path):
    image_path = tmp_path / "images" / "test" / "roof.jpg"
    original_path = tmp_path / "masks" / "test" / "roof.png"
    original_path.parent.mkdir(parents=True)
    raster = np.zeros((10, 10), dtype=np.uint8)
    raster[2:8, 2:8] = 255
    raster[4:6, 4:6] = 0
    Image.fromarray(raster).save(original_path)
    polygons = [Detection("solar_panel", 1., ((2, 2), (7, 2), (7, 7), (2, 7)))]
    mask, source = reference_mask(image_path, polygons, 10, 10)
    assert source == "original_source_raster"
    assert mask.sum() == 32
    assert not mask[4, 4]


def test_numeric_labels_use_explicit_model_name_mapping(tmp_path):
    path = tmp_path / "label.txt"
    path.write_text("1 .2 .2 .8 .2 .8 .8 .2 .8\n")

    labels = read_labels(path, 100, 100, class_names={1: "superstructure", 0: "solar_panel"})

    assert labels[0].class_name == "superstructure"


def test_native_mask_metrics_follow_ap_class_index_not_name_insertion_order():
    native = SimpleNamespace(
        ap_class_index=np.asarray([2, 0]),
        p=np.asarray([0.2, 0.8]),
        r=np.asarray([0.3, 0.9]),
        ap50=np.asarray([0.4, 0.95]),
        ap=np.asarray([0.25, 0.85]),
    )

    metrics = native_mask_metrics(native, {0: "solar_panel", 1: "superstructure", 2: "chimney"})

    assert metrics["solar_panel"]["mask_ap50"] == pytest.approx(0.95)
    assert metrics["chimney"]["mask_ap50"] == pytest.approx(0.4)
    assert metrics["superstructure"]["mask_ap50"] is None


def test_native_mask_metrics_reject_unknown_class_ids():
    native = SimpleNamespace(ap_class_index=np.asarray([4]), p=[.5], r=[.5], ap50=[.5], ap=[.5])
    with pytest.raises(ValueError, match="unknown class ids"):
        native_mask_metrics(native, {0: "solar_panel"})


def test_per_class_and_role_union_areas_do_not_mix_pv_and_obstacles():
    pv = mask_scores(np.asarray([[1, 0], [0, 0]], dtype=bool), np.asarray([[1, 0], [0, 0]], dtype=bool))
    obstacle = mask_scores(np.asarray([[0, 1], [0, 0]], dtype=bool), np.zeros((2, 2), dtype=bool))
    rows = [{
        "class_scores": {"solar_panel": pv, "superstructure": obstacle},
        "role_scores": {"pv": pv, "obstacle": obstacle},
    }]

    per_class = per_class_pixel_metrics(rows, {0: "solar_panel", 1: "superstructure"})

    assert per_class["solar_panel"]["iou"] == 1.0
    assert per_class["superstructure"]["iou"] == 0.0
    assert role_union_metrics(rows, "pv")["iou"] == 1.0
    assert role_union_metrics(rows, "obstacle")["predicted_area_pixels"] == 0


def test_obstacle_reference_does_not_reuse_binary_pv_raster(tmp_path):
    image = tmp_path / "images" / "test" / "roof.jpg"
    original = tmp_path / "masks" / "test" / "roof.png"
    original.parent.mkdir(parents=True)
    Image.fromarray(np.asarray([[255, 0], [0, 0]], dtype=np.uint8)).save(original)
    obstacle = [Detection("superstructure", 1.0, ((0, 0), (1, 0), (1, 1), (0, 1)))]

    mask, source = reference_mask(image, obstacle, 2, 2, class_name="superstructure", use_original=False)

    assert source == "converted_yolo_polygons"
    assert mask.sum() == 4


def test_dataset_manifest_hashes_include_available_split_and_source_files(tmp_path):
    data_root = tmp_path / "data"
    prepared = data_root / "processed" / "pv"
    raw = data_root / "raw"
    prepared.mkdir(parents=True)
    raw.mkdir(parents=True)
    yaml_path = prepared / "dataset.yaml"
    splits = prepared / "splits.json"
    source = raw / "source-metadata.json"
    yaml_path.write_text("train: images/train\n")
    splits.write_text(json.dumps({"seed": 42}))
    source.write_text(json.dumps({"version": 1}))

    hashes = dataset_manifest_hashes(yaml_path)

    assert all(isinstance(value, str) for value in hashes.values())


def test_empty_aggregate_is_well_defined():
    assert aggregate_pixel_scores([])["iou"] == 1.0
