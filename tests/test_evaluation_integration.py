"""End-to-end evaluation tests with deterministic Ultralytics test doubles.

These tests deliberately exercise the complete report-writing path without loading a
GPU checkpoint.  The temporary fixtures use real JPEG and PNG files so path
resolution, original-mask handling, manifest hashing, and review rendering all run
as they do for the prepared Swiss dataset.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from rooftop_pv import evaluation
from rooftop_pv.evaluation import evaluate
from rooftop_pv.inference import Detection


IMAGE_SIZE = 10


def _polygon_line(class_id: int, box: tuple[int, int, int, int]) -> str:
    left, top, right, bottom = box
    points = ((left, top), (right, top), (right, bottom), (left, bottom))
    normalized = " ".join(str(coordinate / IMAGE_SIZE) for point in points for coordinate in point)
    return f"{class_id} {normalized}\n"


def _paint_mask(boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = Image.new("1", (IMAGE_SIZE, IMAGE_SIZE), 0)
    draw = ImageDraw.Draw(mask)
    for box in boxes:
        draw.rectangle(box, fill=1)
    return np.asarray(mask, dtype=bool)


def _write_image(root, stem: str, label_lines: list[str], mask: np.ndarray) -> None:
    image_path = root / "images" / "test" / f"{stem}.jpg"
    label_path = root / "labels" / "test" / f"{stem}.txt"
    mask_path = root / "masks" / "test" / f"{stem}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (120, 140, 160)).save(image_path, quality=95)
    label_path.write_text("".join(label_lines))
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)


def _dataset(tmp_path, *, multiclass: bool):
    prepared = tmp_path / "data" / "processed" / ("multiclass" if multiclass else "swiss-pv")
    if multiclass:
        pv_box = (1, 1, 4, 4)
        obstacle_box = (6, 6, 8, 8)
        _write_image(
            prepared,
            "geneva-roof",
            [_polygon_line(0, pv_box), _polygon_line(1, obstacle_box)],
            _paint_mask([pv_box]),
        )
    else:
        pv_box = (2, 2, 8, 8)
        _write_image(prepared, "roof-with-pv", [_polygon_line(0, pv_box)], _paint_mask([pv_box]))
        _write_image(prepared, "roof-without-pv", [], np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool))

    prepared.mkdir(parents=True, exist_ok=True)
    data_yaml = prepared / "dataset.yaml"
    data_yaml.write_text("train: images/test\nval: images/test\ntest: images/test\n")
    (prepared / "splits.json").write_text(json.dumps({"seed": 42, "test": ["frozen"]}))
    source_metadata = tmp_path / "data" / "raw" / "source-metadata.json"
    source_metadata.parent.mkdir(parents=True, exist_ok=True)
    source_metadata.write_text(json.dumps({"source": "fixture", "version": 1}))
    return prepared, data_yaml, pv_box, source_metadata


class _FakeModel:
    def __init__(self, names, segment_metrics):
        self.names = names
        self.segment_metrics = segment_metrics
        self.val_calls = []

    def val(self, **kwargs):
        self.val_calls.append(kwargs)
        return SimpleNamespace(seg=self.segment_metrics)


def _segment_metrics(class_indexes: list[int], *, pv_first: bool = True):
    if pv_first:
        precision = [.9, .6][:len(class_indexes)]
        recall = [.8, .5][:len(class_indexes)]
        ap50 = [.85, .45][:len(class_indexes)]
        ap = [.75, .35][:len(class_indexes)]
    else:
        precision = [.2, .9]
        recall = [.3, .8]
        ap50 = [.4, .95]
        ap = [.25, .85]
    return SimpleNamespace(
        map50=.82,
        map=.61,
        mp=.77,
        mr=.73,
        ap_class_index=np.asarray(class_indexes),
        p=np.asarray(precision),
        r=np.asarray(recall),
        ap50=np.asarray(ap50),
        ap=np.asarray(ap),
    )


def _weights_and_root(tmp_path):
    repo_root = tmp_path / "repo"
    weights = repo_root / "artifacts" / "models" / "fixture.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"not-a-neural-network")
    return repo_root, weights


def _patch_runtime(monkeypatch, repo_root, model, predictions):
    monkeypatch.setattr(evaluation, "ROOT", repo_root)
    monkeypatch.setattr(evaluation, "load_segmenter", lambda _weights: model)
    monkeypatch.setattr(evaluation, "choose_device", lambda _device: "cpu")
    monkeypatch.setattr(evaluation, "predict", predictions)


def test_evaluate_writes_swiss_original_mask_report_and_manifests(tmp_path, monkeypatch):
    prepared, data_yaml, _, source_metadata = _dataset(tmp_path, multiclass=False)
    repo_root, weights = _weights_and_root(tmp_path)
    model = _FakeModel({0: "solar_panel"}, _segment_metrics([0]))
    pv_polygon = ((2., 2.), (8., 2.), (8., 8.), (2., 8.))

    def fake_predict(image, **_kwargs):
        return [Detection("solar_panel", .91, pv_polygon)]

    _patch_runtime(monkeypatch, repo_root, model, fake_predict)
    # A model manifest is updated only when its checkpoint digest matches.
    manifest = repo_root / "artifacts" / "models" / "fixture.json"
    manifest.write_text(json.dumps({"best_weights_sha256": evaluation.sha256(weights)}))

    report = evaluate(weights, data_yaml, split="test", confidence=.25, device="cpu", imgsz=512, batch=2)
    report_path = Path(report["report_path"])
    persisted = json.loads(report_path.read_text())
    updated_manifest = json.loads(manifest.read_text())

    assert report_path.is_file()
    assert model.val_calls[0]["split"] == "test"
    assert model.val_calls[0]["device"] == "cpu"
    assert persisted["images"] == 2
    assert persisted["area_reference"] == ["original_source_raster"]
    assert all(row["reference_type"] == "original_source_raster" for row in persisted["rows"])
    assert persisted["operating_confidence"] == .25
    assert persisted["imgsz"] == 512
    assert persisted["dataset_splits_sha256"] == evaluation.sha256(prepared / "splits.json")
    assert persisted["dataset_source_manifest_sha256"] == evaluation.sha256(source_metadata)
    assert persisted["evaluation_source_sha256"] == evaluation.sha256(Path(evaluation.__file__))
    assert persisted["inference_source_sha256"] == evaluation.sha256(Path(evaluation.__file__).with_name("inference.py"))
    assert persisted["area_metric_scope"] == "single_pv_original_source_raster"
    assert persisted["metrics"]["area_metric_scope"] == "single_pv_original_source_raster"
    assert persisted["metrics"]["per_class"]["solar_panel"]["native_mask"]["mask_ap50"] == .85
    assert persisted["metrics"]["role_unions"]["pv"]["images"] == 2
    assert persisted["metrics"]["role_unions"]["obstacle"]["iou"] == 1.0
    assert "obstacle_union_iou" not in persisted["gate"]["checks"]
    assert persisted["class_names"] == {"0": "solar_panel"}
    assert persisted["class_roles"] == {"solar_panel": "pv"}
    assert updated_manifest["test_evaluation"] == report["report_path"]
    assert updated_manifest["quality_status"] in {"dataset_gates_passed", "experimental_gates_failed"}
    assert len(list(report_path.parent.joinpath("review").glob("*.jpg"))) == 2


def test_evaluate_keeps_multiclass_pv_and_obstacle_unions_distinct(tmp_path, monkeypatch):
    prepared, data_yaml, _, _ = _dataset(tmp_path, multiclass=True)
    # Multiclass/RID2 does not have the Swiss binary source raster.  This also
    # catches accidental reuse of all labels as the PV role reference.
    (prepared / "masks" / "test" / "geneva-roof.png").unlink()
    repo_root, weights = _weights_and_root(tmp_path)
    model = _FakeModel({0: "solar_panel", 1: "chimney"}, _segment_metrics([1, 0], pv_first=False))
    pv_polygon = ((1., 1.), (4., 1.), (4., 4.), (1., 4.))
    obstacle_polygon = ((6., 6.), (8., 6.), (8., 8.), (6., 8.))

    def fake_predict(image, **_kwargs):
        return [Detection("solar_panel", .95, pv_polygon), Detection("chimney", .8, obstacle_polygon)]

    _patch_runtime(monkeypatch, repo_root, model, fake_predict)
    report = evaluate(weights, data_yaml, split="test", device="cpu")
    metrics = report["metrics"]

    assert report["area_reference"] == ["converted_yolo_polygons"]
    assert report["area_metric_scope"] == "all_classes_yolo_polygons"
    assert metrics["area_metric_scope"] == "all_classes_yolo_polygons"
    assert report["class_roles"] == {"solar_panel": "pv", "chimney": "obstacle"}
    assert metrics["union_iou"] == pytest.approx(metrics["role_unions"]["pv"]["iou"])
    assert metrics["role_unions"]["pv"]["iou"] == pytest.approx(1.0)
    assert metrics["union_iou"] == pytest.approx(1.0)
    assert metrics["role_unions"]["obstacle"]["iou"] == pytest.approx(1.0)
    assert metrics["per_class"]["solar_panel"]["pixel"]["iou"] == pytest.approx(1.0)
    assert metrics["per_class"]["chimney"]["pixel"]["iou"] == pytest.approx(1.0)
    assert metrics["native_per_class"]["chimney"]["mask_ap50"] == pytest.approx(.4)
    assert metrics["native_per_class"]["solar_panel"]["mask_ap50"] == pytest.approx(.95)
    assert report["rows"][0]["role_scores"]["pv"]["iou"] == pytest.approx(1.0)
    assert report["rows"][0]["role_scores"]["obstacle"]["iou"] == pytest.approx(1.0)


def test_multiclass_pv_only_prediction_fails_obstacle_gate(tmp_path, monkeypatch):
    prepared, data_yaml, _, _ = _dataset(tmp_path, multiclass=True)
    (prepared / "masks" / "test" / "geneva-roof.png").unlink()
    repo_root, weights = _weights_and_root(tmp_path)
    model = _FakeModel({0: "solar_panel", 1: "chimney"}, _segment_metrics([0, 1]))
    pv_polygon = ((1., 1.), (4., 1.), (4., 4.), (1., 4.))

    def fake_predict(image, **_kwargs):
        return [Detection("solar_panel", .95, pv_polygon)]

    _patch_runtime(monkeypatch, repo_root, model, fake_predict)
    report = evaluate(weights, data_yaml, split="val", device="cpu")

    assert report["metrics"]["role_unions"]["pv"]["iou"] == pytest.approx(1.0)
    assert report["metrics"]["role_unions"]["obstacle"]["iou"] == pytest.approx(0.0)
    assert report["metrics"]["obstacle_union_iou"] == pytest.approx(0.0)
    assert report["gate"]["checks"]["obstacle_union_iou"]["minimum"] == .65
    assert not report["gate"]["checks"]["obstacle_union_iou"]["passed"]
    assert not report["gate"]["passed"]


def test_evaluate_rejects_unknown_numeric_label_class(tmp_path, monkeypatch):
    prepared, data_yaml, _, _ = _dataset(tmp_path, multiclass=False)
    # Replace the first label with an unsupported class id while retaining a valid polygon.
    (prepared / "labels" / "test" / "roof-with-pv.txt").write_text(_polygon_line(1, (2, 2, 8, 8)))
    repo_root, weights = _weights_and_root(tmp_path)
    model = _FakeModel({0: "solar_panel"}, _segment_metrics([0]))
    _patch_runtime(monkeypatch, repo_root, model, lambda *_args, **_kwargs: [])

    with pytest.raises(ValueError, match="Unknown class id 1"):
        evaluate(weights, data_yaml, split="test", device="cpu")
