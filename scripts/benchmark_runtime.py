"""Measure reproducible local geometry, physics and optional inference timings.

The script deliberately writes results outside version control by default.  It
does not impose performance targets: compare like-for-like reports from the
same device, package lock and workload before declaring a regression.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd
from PIL import Image
from shapely.geometry import box

from rooftop_pv.geometry import calculate_usable_area, place_panels
from rooftop_pv.inference import predict
from rooftop_pv.physics import PVConfig, simulate_pv
from rooftop_pv.runtime import ROOT


def _measure(operation: Callable[[], Any], repeat: int) -> dict[str, float]:
    """Run one deterministic operation and return milliseconds per run."""

    measurements: list[float] = []
    for _ in range(repeat):
        started = perf_counter()
        operation()
        measurements.append((perf_counter() - started) * 1_000)
    return {
        "median_ms": round(median(measurements), 3),
        "min_ms": round(min(measurements), 3),
        "max_ms": round(max(measurements), 3),
        "runs": repeat,
    }


def _weather(hours: int) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=hours, freq="h", tz="Europe/Zurich")
    day_fraction = (index.hour.to_numpy() + 0.5) / 24
    irradiance = np.maximum(np.sin(np.pi * day_fraction), 0.0)
    return pd.DataFrame(
        {
            "ghi": 700 * irradiance,
            "dni": 500 * irradiance,
            "dhi": 200 * irradiance,
            "temp_air": np.full(hours, 15.0),
            "wind_speed": np.full(hours, 2.0),
        },
        index=index,
    )


def benchmark_core(*, hours: int, repeat: int) -> dict[str, dict[str, float]]:
    """Measure pure workloads without data download, checkpoint or network use."""

    roof = box(0, 0, 25, 18)
    exclusions = [box(5, 4, 7, 7), box(12, 8, 16, 10)]
    weather = _weather(hours)
    config = PVConfig(
        latitude=47.3769,
        longitude=8.5417,
        tilt_deg=30,
        azimuth_deg=180,
        available_area_m2=80,
    )

    def geometry() -> None:
        area = calculate_usable_area(roof, exclusions=exclusions, tilt_deg=30, setback_m=0.3)
        place_panels(area, module_width_m=1.134, module_height_m=1.722, gap_m=0.1)

    return {
        "geometry_and_placement": _measure(geometry, repeat),
        "pv_physics": _measure(lambda: simulate_pv(weather, config), repeat),
    }


def _device_info() -> dict[str, str | None]:
    info: dict[str, str | None] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "processor": platform.processor() or None,
    }
    try:
        import torch

        info.update(
            torch=torch.__version__,
            cuda_available=str(torch.cuda.is_available()),
            cuda_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            mps_available=str(torch.backends.mps.is_available()),
        )
    except ImportError:
        info["torch"] = None
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=8_760, help="Synthetic hourly weather rows")
    parser.add_argument("--repeat", type=int, default=3, help="Runs per core workload")
    parser.add_argument("--image", type=Path, help="Optional RGB image for a real inference measurement")
    parser.add_argument("--weights", type=Path, help="Checkpoint used together with --image")
    parser.add_argument("--device", default="auto", help="Ultralytics device for optional inference")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "benchmarks" / "runtime.json",
        help="JSON destination; generated artifacts are ignored by Git",
    )
    arguments = parser.parse_args(argv)
    if arguments.hours < 24 or arguments.repeat < 1:
        parser.error("--hours must be at least 24 and --repeat must be positive")
    if bool(arguments.image) != bool(arguments.weights):
        parser.error("--image and --weights must be supplied together")

    report: dict[str, Any] = {
        "workload": {"weather_hours": arguments.hours, "repeat": arguments.repeat},
        "device": _device_info(),
        "timings": benchmark_core(hours=arguments.hours, repeat=arguments.repeat),
    }
    if arguments.image:
        with Image.open(arguments.image) as source:
            image = source.convert("RGB")
        report["timings"]["inference_cold"] = _measure(
            lambda: predict(image, arguments.weights, device=arguments.device), 1
        )
        report["inference"] = {
            "image": str(arguments.image.resolve()),
            "weights": str(arguments.weights.resolve()),
            "device": arguments.device,
            "image_size": [image.width, image.height],
        }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
