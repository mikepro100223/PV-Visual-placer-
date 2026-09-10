"""High-risk local contracts for training, runtime, CLI and inference.

These tests intentionally replace the YOLO trainer and torch/ultralytics
imports with small fakes. They verify manifests, role separation, resume
behaviour and input safety without downloading weights or running a GPU job.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from rooftop_pv import cli, inference, runtime, training


def _training_fixture(tmp_path: Path) -> Path:
    data_root = tmp_path / "dataset"
    for split in ("train", "val"):
        image_dir = data_root / split / "images"
        image_dir.mkdir(parents=True)
        (image_dir / f"{split}-001.jpg").write_bytes(b"local fixture")
    data_yaml = data_root / "dataset.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(data_root),
                "train": "train/images",
                "val": "val/images",
                "names": {0: "solar_panel"},
            }
        )
    )
    (data_root / "splits.json").write_text("{}")
    return data_yaml


class _FakeTrainer:
    def __init__(self, save_dir: Path):
        self.save_dir = str(save_dir)
        self.metrics = {"metrics/mask_map50": 0.71, "metrics/mask_map50-95": 0.48}


class _FakeYOLO:
    instances: list["_FakeYOLO"] = []
    should_fail = False
    names = {0: "solar_panel"}

    def __init__(self, weights: str, task: str):
        self.weights = weights
        self.task = task
        self.names = dict(type(self).names)
        self.trainer = _FakeTrainer(Path(weights).parent.parent)
        self.train_calls: list[tuple[tuple, dict]] = []
        type(self).instances.append(self)

    def train(self, *args, **kwargs):
        self.train_calls.append((args, kwargs))
        if type(self).should_fail:
            raise RuntimeError("synthetic trainer failure")
        if kwargs.get("resume"):
            save_dir = Path(self.weights).parent.parent
        else:
            save_dir = Path(kwargs["project"]) / kwargs["name"]
        self.trainer.save_dir = str(save_dir)
        (save_dir / "weights").mkdir(parents=True, exist_ok=True)
        (save_dir / "weights" / "best.pt").write_bytes(b"fake best checkpoint")
        (save_dir / "weights" / "last.pt").write_bytes(b"fake last checkpoint")


@pytest.fixture
def prepared_training(monkeypatch, tmp_path):
    data_yaml = _training_fixture(tmp_path)
    config = tmp_path / "train.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": "yolo11n-seg.pt",
                "data": str(data_yaml),
                "epochs": 1,
                "imgsz": 320,
                "batch": 1,
                "workers": 0,
                "device": "cpu",
            }
        )
    )
    monkeypatch.setattr(training, "ROOT", tmp_path)
    monkeypatch.setattr(training, "choose_device", lambda requested: "cpu")
    pretrained = tmp_path / "artifacts" / "pretrained"
    pretrained.mkdir(parents=True)
    (pretrained / "yolo11n-seg.pt").write_bytes(b"fake starting checkpoint")
    (tmp_path / "code" / "rooftop_pv").mkdir(parents=True)
    (tmp_path / "code" / "rooftop_pv" / "fixture.py").write_text("# local fixture\n")
    _FakeYOLO.instances.clear()
    _FakeYOLO.should_fail = False
    monkeypatch.setattr(training, "yolo_class", lambda: _FakeYOLO)
    return config


@pytest.mark.parametrize(
    ("role", "index_name", "class_name"),
    [("pv", "current.json", "solar_panel"), ("obstacles", "obstacles.json", "chimney")],
)
def test_train_writes_completed_role_specific_manifest_without_real_yolo(
    prepared_training, role, index_name, class_name
):
    _FakeYOLO.names = {0: class_name}
    manifest = training.train(prepared_training, overrides={"role": role})

    root = prepared_training.parent
    assert manifest["status"] == "completed"
    assert manifest["role"] == role
    assert manifest["quality_status"] == "experimental_not_evaluated"
    assert manifest["train_images"] == 1
    assert manifest["validation_images"] == 1
    assert Path(manifest["best_weights"]).is_file()
    assert (root / "artifacts" / "models" / index_name).is_file()
    other_index = "obstacles.json" if role == "pv" else "current.json"
    assert not (root / "artifacts" / "models" / other_index).is_file()


def test_train_failure_persists_failed_manifest_and_re_raises(prepared_training):
    _FakeYOLO.should_fail = True

    with pytest.raises(RuntimeError, match="synthetic trainer failure"):
        training.train(prepared_training)

    root = prepared_training.parent
    run_dirs = list((root / "artifacts" / "training").iterdir())
    assert len(run_dirs) == 1
    manifest = json.loads((run_dirs[0] / "run_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "synthetic trainer failure" in manifest["error"]
    assert not (root / "artifacts" / "models" / "current.json").exists()


def test_train_resume_uses_last_checkpoint_and_resume_flag(prepared_training):
    root = prepared_training.parent
    run_dir = root / "artifacts" / "training" / "interrupted-run"
    checkpoint = run_dir / "weights" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"interrupted checkpoint")
    configuration = yaml.safe_load(prepared_training.read_text())
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "config": {key: configuration[key] for key in ("epochs", "imgsz", "batch", "workers")},
        "dataset_yaml": configuration["data"], "starting_model": configuration["model"],
        "role": "pv", "device": "cpu", "status": "interrupted",
    }))

    manifest = training.train(prepared_training, resume=checkpoint)

    fake = _FakeYOLO.instances[0]
    assert fake.weights == str(checkpoint.resolve())
    assert fake.train_calls == [((), {"resume": True, "device": "cpu"})]
    assert manifest["resume_checkpoint"] == str(checkpoint.resolve())
    assert manifest["status"] == "completed"
    assert manifest["run_dir"] == str(run_dir)


def test_resume_preserves_obstacle_role_even_with_default_pv_configuration(prepared_training):
    original = training.train(prepared_training, overrides={"role": "obstacles"})
    resumed = training.train(prepared_training, resume=Path(original["last_weights"]))
    assert resumed["role"] == "obstacles"
    assert not (prepared_training.parent / "artifacts/models/current.json").exists()
    assert resumed["resume_history"][-1]["role"] == "obstacles"


def test_resume_rejects_missing_manifest_and_dataset_drift(prepared_training):
    checkpoint = prepared_training.parent / "unknown/weights/last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"unknown checkpoint")
    with pytest.raises(ValueError, match="manifest"):
        training.train(prepared_training, resume=checkpoint)
    original = training.train(prepared_training)
    Path(original["dataset_yaml"]).write_text("changed source")
    with pytest.raises(ValueError, match="changed"):
        training.train(prepared_training, resume=Path(original["last_weights"]))


def test_runtime_choose_device_is_explicit_for_unavailable_accelerators(monkeypatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert runtime.choose_device("auto") == "cpu"
    with pytest.raises(RuntimeError, match="MPS"):
        runtime.choose_device("mps")
    with pytest.raises(RuntimeError, match="CUDA"):
        runtime.choose_device("cuda")

    fake_torch.cuda.is_available = lambda: True
    assert runtime.choose_device("auto") == "0"


def test_runtime_yolo_class_sets_local_offline_ultralytics_settings(monkeypatch, tmp_path):
    class FakeSettings:
        def __init__(self):
            self.values = None

        def update(self, values):
            self.values = dict(values)

    settings = FakeSettings()
    sentinel = object()
    fake_ultralytics = SimpleNamespace(YOLO=sentinel, settings=settings)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.delenv("YOLO_CONFIG_DIR", raising=False)
    monkeypatch.delenv("YOLO_AUTOINSTALL", raising=False)

    assert runtime.yolo_class() is sentinel
    assert (tmp_path / "artifacts" / "ultralytics" / "Ultralytics").is_dir()
    assert settings.values["sync"] is False
    assert settings.values["wandb"] is False
    assert settings.values["tensorboard"] is False


def test_cli_defaults_are_local_and_use_expected_imported_commands(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    arguments = cli.parser().parse_args(["simulate", "--area", "20"])
    assert arguments.latitude == pytest.approx(47.4831)
    assert arguments.longitude == pytest.approx(8.2071)
    assert arguments.tilt == 30
    assert arguments.azimuth == 180
    assert arguments.output == tmp_path / "artifacts" / "simulation"

    train_arguments = cli.parser().parse_args(["train"])
    assert train_arguments.config == tmp_path / "configs" / "train.yaml"
    assert train_arguments.resume is None
    assert train_arguments.device is None


def test_cli_current_weights_requires_completed_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="completed training"):
        cli._current_weights()

    manifest = tmp_path / "artifacts" / "models" / "current.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"best_weights": str(tmp_path / "best.pt")}))
    assert cli._current_weights() == tmp_path / "best.pt"


def test_inference_rejects_missing_and_non_rooftop_checkpoints(monkeypatch, tmp_path):
    with pytest.raises(FileNotFoundError, match="Trained weights missing"):
        inference.load_segmenter(tmp_path / "missing.pt")

    weights = tmp_path / "coco.pt"
    weights.write_bytes(b"not a usable checkpoint")

    class FakeCOCO:
        task = "segment"
        names = {0: "person", 1: "car"}

        def __init__(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(inference, "yolo_class", lambda: FakeCOCO)
    with pytest.raises(ValueError, match="supported rooftop class"):
        inference.load_segmenter(weights)


def test_inference_rejects_detection_checkpoint_and_invalid_predict_inputs(monkeypatch, tmp_path):
    weights = tmp_path / "detect.pt"
    weights.write_bytes(b"fake")

    class FakeDetectionModel:
        task = "detect"
        names = {0: "solar_panel"}

        def __init__(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(inference, "yolo_class", lambda: FakeDetectionModel)
    with pytest.raises(ValueError, match="segmentation"):
        inference.load_segmenter(weights)

    tiny_image = SimpleNamespace(width=100, height=100)
    for confidence in (0.0, -0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="Confidence"):
            inference.predict(tiny_image, model=object(), confidence=confidence)

    huge_image = SimpleNamespace(width=5001, height=5001)
    with pytest.raises(ValueError, match="too large"):
        inference.predict(huge_image, model=object(), confidence=0.25)


def test_inference_rejects_nonfinite_pixel_polygon_and_bad_dimensions():
    with pytest.raises(ValueError, match="finite"):
        inference.pixel_to_map(
            [(0, 0), (float("nan"), 1), (1, 1)],
            (0, 0, 10, 10),
            100,
            100,
        )
    with pytest.raises(ValueError, match="dimensions"):
        inference.pixel_to_map([(0, 0), (1, 0), (1, 1)], (0, 0, 10, 10), 0, 100)
