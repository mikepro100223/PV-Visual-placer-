"""Reproducible YOLO11 segmentation training and checkpoint manifests."""

import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from rooftop_pv.runtime import ROOT, choose_device, yolo_class


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path, overrides: dict | None = None) -> dict:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a YAML mapping.")
    config.update({k: v for k, v in (overrides or {}).items() if v is not None})
    model = str(config.get("model", ""))
    if not re.fullmatch(r"yolo11[nslmx]-seg\.pt", Path(model).name):
        raise ValueError("Choose an official YOLO11 segmentation starting model, e.g. yolo11n-seg.pt.")
    for key in ("epochs", "imgsz", "batch"):
        if not isinstance(config.get(key), int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if config["imgsz"] % 32:
        raise ValueError("imgsz must be divisible by 32.")
    return config


def dataset_paths(data_yaml: Path, split: str) -> list[Path]:
    """Resolve a local YOLO split; never execute a YAML download directive."""
    document = yaml.safe_load(data_yaml.read_text())
    if not isinstance(document, dict):
        raise ValueError("Dataset YAML must be a mapping.")
    if document.get("download"):
        raise ValueError("Dataset YAML must not contain an executable download directive.")
    base = Path(document.get("path", data_yaml.parent))
    if not base.is_absolute():
        base = data_yaml.parent / base
    entries = document.get(split)
    if not entries:
        raise ValueError(f"No {split} split in {data_yaml}.")
    files = []
    for entry in entries if isinstance(entries, list) else [entries]:
        path = Path(entry)
        path = path if path.is_absolute() else base / path
        if path.is_dir():
            files.extend(p for p in sorted(path.rglob("*"))
                         if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"})
        elif path.is_file() and path.suffix == ".txt":
            for line in path.read_text().splitlines():
                if line.strip():
                    image = Path(line.strip())
                    files.append(image if image.is_absolute() else path.parent / image)
        else:
            raise ValueError(f"Split entry is not a local image directory/list: {path}")
    if not files or any(not p.is_file() for p in files):
        raise ValueError(f"{split} is empty or references missing images.")
    return files


def train(config_path: Path, overrides: dict | None = None, resume: Path | None = None) -> dict:
    config = load_config(config_path, overrides)
    previous_run = None
    if resume:
        resume = resume.resolve()
        if not resume.is_file():
            raise FileNotFoundError(resume)
        previous_path = resume.parent.parent / "run_manifest.json"
        if not previous_path.is_file():
            raise ValueError("Resume requires the checkpoint's original run_manifest.json.")
        previous_run = json.loads(previous_path.read_text())
        if not isinstance(previous_run, dict) or not all(
            key in previous_run for key in ("config", "dataset_yaml", "starting_model")
        ):
            raise ValueError("Resume manifest is incomplete.")
        if any(value is not None for key, value in (overrides or {}).items() if key != "device"):
            raise ValueError("Resume uses the original experiment; only --device may be overridden.")
        # An obstacle checkpoint must never be registered as the default PV
        # specialist merely because the caller omitted --config on resume.
        config = dict(previous_run["config"], model=previous_run["starting_model"],
                      data=previous_run["dataset_yaml"], role=previous_run.get("role", "pv"),
                      device=(overrides or {}).get("device") or previous_run.get("device", "auto"))
    data_yaml = Path(config.pop("data"))
    data_yaml = data_yaml if data_yaml.is_absolute() else ROOT / data_yaml
    data_yaml = data_yaml.resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Prepared dataset missing: {data_yaml}. Run prepare-data first.")
    if previous_run:
        for path, key in ((data_yaml, "dataset_yaml_sha256"),
                          (data_yaml.parent / "splits.json", "dataset_manifest_sha256")):
            if previous_run.get(key) and (not path.is_file() or sha256(path) != previous_run[key]):
                raise ValueError(f"Resume dataset changed since the original training run: {path}")
    train_images = dataset_paths(data_yaml, "train")
    val_images = dataset_paths(data_yaml, "val")
    if set(p.resolve() for p in train_images) & set(p.resolve() for p in val_images):
        raise ValueError("Training and validation image paths overlap.")
    starting_model = config.pop("model")
    role = config.pop("role", "pv")
    if role not in {"pv", "obstacles"}:
        raise ValueError("Model role must be pv or obstacles.")
    device = choose_device(str(config.pop("device", "auto")))
    run_name = datetime.now(timezone.utc).strftime(f"yolo11-{role}-%Y%m%dT%H%M%S%fZ")
    run_dir = ROOT / "artifacts" / "training" / run_name
    if resume:
        run_dir = resume.parent.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "run_manifest.json"
    code_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=False).stdout.strip()
    diff = subprocess.run(["git", "diff", "--binary"], cwd=ROOT, capture_output=True,
                          check=False).stdout
    source_digest = hashlib.sha256()
    for path in sorted((ROOT / "code").rglob("*.py")):
        source_digest.update(str(path.relative_to(ROOT)).encode())
        source_digest.update(path.read_bytes())
    manifest = {
        "status": "training", "started_utc": datetime.now(timezone.utc).isoformat(),
        "role": role,
        "run_dir": str(run_dir), "code_sha": code_sha,
        "working_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_tree_sha256": source_digest.hexdigest(),
        "config": config, "device": device, "starting_model": starting_model,
        "resume_checkpoint": str(resume) if resume else None,
        "dataset_yaml": str(data_yaml), "dataset_yaml_sha256": sha256(data_yaml),
        "dataset_manifest_sha256": sha256(data_yaml.parent / "splits.json")
        if (data_yaml.parent / "splits.json").is_file() else None,
        "train_images": len(train_images), "validation_images": len(val_images),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("ultralytics", "torch", "torchvision", "numpy")},
        "quality_status": "experimental_not_evaluated",
        "limitations": ["Only annotated classes are trained.",
                        "MPS results may not be bitwise reproducible."]}
    if previous_run:
        manifest["resume_history"] = [*previous_run.get("resume_history", []),
                                      {k: v for k, v in previous_run.items() if k != "resume_history"}]
    state_path.write_text(json.dumps(manifest, indent=2))
    try:
        YOLO = yolo_class()
        if resume:
            model = YOLO(str(resume), task="segment")
        else:
            weight_dir = ROOT / "artifacts" / "pretrained"
            weight_dir.mkdir(parents=True, exist_ok=True)
            model = YOLO(str(weight_dir / starting_model), task="segment")
            manifest["starting_weights_sha256"] = sha256(weight_dir / starting_model)
        if model.task != "segment":
            raise ValueError("Checkpoint is not an instance segmentation model.")
        arguments = dict(config, data=str(data_yaml), device=device,
                         project=str(run_dir.parent), name=run_dir.name,
                         exist_ok=True, deterministic=True, save=True, val=True)
        if resume:
            model.train(resume=True, device=device)
        else:
            model.train(**arguments)
        actual_dir = Path(model.trainer.save_dir)
        best = actual_dir / "weights" / "best.pt"
        manifest.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat(),
                        best_weights=str(best), best_weights_sha256=sha256(best),
                        last_weights=str(actual_dir / "weights" / "last.pt"),
                        classes=model.names,
                        validation_metrics={k: float(v) for k, v in model.trainer.metrics.items()})
        state_path.write_text(json.dumps(manifest, indent=2))
        index = ROOT / "artifacts" / "models" / ("current.json" if role == "pv" else "obstacles.json")
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text(json.dumps(manifest, indent=2))
        return manifest
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=f"{type(error).__name__}: {error}")
        state_path.write_text(json.dumps(manifest, indent=2))
        raise
