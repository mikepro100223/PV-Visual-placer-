"""Repository-local paths and deliberately local-only ML settings."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def yolo_class():
    (ROOT / "artifacts" / "ultralytics" / "Ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "artifacts" / "ultralytics"))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    from ultralytics import YOLO, settings

    settings.update({"sync": False, "wandb": False, "mlflow": False, "comet": False,
                     "clearml": False, "dvc": False, "raytune": False,
                     "tensorboard": False})
    return YOLO


def choose_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Apple MPS is not available; choose --device cpu.")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this machine.")
        return requested
    if torch.cuda.is_available():
        return "0"
    return "mps" if torch.backends.mps.is_available() else "cpu"
