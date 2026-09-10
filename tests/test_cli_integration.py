"""Command-level coverage for the local rooftop-PV CLI.

These tests deliberately call :func:`rooftop_pv.cli.main` rather than only
testing argparse.  Network, model, and accelerator boundaries are replaced by
small fakes; filesystem assertions still exercise the CLI's real reporting and
artifact-writing paths.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from PIL import Image

from rooftop_pv import cli, data, evaluation, inference, obstacle_data, physics, runtime, sources, training


def _printed_json(capsys) -> dict:
    """Return the JSON object emitted by one CLI invocation."""

    captured = capsys.readouterr()
    return json.loads(captured.out)


def _jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), color=(20, 40, 60)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_download_and_prepare_pv_commands_use_paths_and_emit_dataclass_json(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    raw_dir = tmp_path / "raw-pv"
    downloaded = raw_dir / "source.zip"
    calls: dict[str, object] = {"downloads": [], "prepare": None}

    def fake_download(raw):
        raw = Path(raw)
        calls["downloads"].append(raw)
        raw.mkdir(parents=True, exist_ok=True)
        downloaded.write_bytes(b"archive")
        return downloaded

    def fake_prepare(archive, output, *, seed):
        calls["prepare"] = (Path(archive), Path(output), seed)
        return data.PreparedDataset(
            root=Path(output),
            dataset_yaml=Path(output) / "dataset.yaml",
            splits_json=Path(output) / "splits.json",
            images=12,
            polygons=18,
            split_counts={"train": 8, "val": 2, "test": 2},
        )

    monkeypatch.setattr(data, "download_dataset", fake_download)
    monkeypatch.setattr(data, "prepare_dataset", fake_prepare)

    cli.main(["download-data", "--raw-dir", str(raw_dir)])
    download_report = _printed_json(capsys)
    assert download_report == {"archive": str(downloaded)}
    assert calls["downloads"] == [raw_dir]

    output = tmp_path / "prepared-pv"
    cli.main(
        [
            "prepare-data",
            "--archive",
            str(downloaded),
            "--output",
            str(output),
            "--seed",
            "7",
        ]
    )
    prepare_report = _printed_json(capsys)
    assert calls["prepare"] == (downloaded, output, 7)
    assert prepare_report["root"] == str(output)
    assert prepare_report["dataset_yaml"] == str(output / "dataset.yaml")
    assert prepare_report["split_counts"] == {"train": 8, "val": 2, "test": 2}


def test_prepare_pv_without_archive_downloads_into_root_raw_dir(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    calls: dict[str, object] = {"download": None, "prepare": None}
    archive = tmp_path / "data" / "raw" / "auto.zip"

    def fake_download(raw):
        calls["download"] = Path(raw)
        return archive

    def fake_prepare(archive_path, output, *, seed):
        calls["prepare"] = (Path(archive_path), Path(output), seed)
        return data.PreparedDataset(
            Path(output), Path(output) / "dataset.yaml", Path(output) / "splits.json", 1, 1, {}
        )

    monkeypatch.setattr(data, "download_dataset", fake_download)
    monkeypatch.setattr(data, "prepare_dataset", fake_prepare)

    cli.main(["prepare-data", "--output", str(tmp_path / "prepared"), "--seed", "11"])
    _printed_json(capsys)
    assert calls["download"] == tmp_path / "data" / "raw"
    assert calls["prepare"] == (archive, tmp_path / "prepared", 11)


def test_download_and_prepare_rid2_commands_forward_limits_and_emit_json(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    raw_dir = tmp_path / "rid2-raw"
    archive = raw_dir / obstacle_data.RID2_ARCHIVE_NAME
    calls: dict[str, object] = {"metadata": None, "ensure": None, "prepare": None}

    def fake_metadata(raw):
        calls["metadata"] = Path(raw)
        raw_dir.mkdir(parents=True, exist_ok=True)
        path = raw_dir / "source-metadata.json"
        path.write_text("{}")
        return path

    def fake_ensure(raw):
        calls["ensure"] = Path(raw)
        return archive

    def fake_prepare(archive_path, output, *, seed, max_images):
        calls["prepare"] = (Path(archive_path), Path(output), seed, max_images)
        return obstacle_data.PreparedRid2(
            root=Path(output),
            dataset_yaml=Path(output) / "dataset.yaml",
            splits_json=Path(output) / "splits.json",
            images=3,
            polygons=4,
            split_counts={"train": 2, "val": 1, "test": 0},
            class_counts={"chimney": 4},
            skipped_polylines=1,
        )

    monkeypatch.setattr(obstacle_data, "fetch_rid2_metadata", fake_metadata)
    monkeypatch.setattr(obstacle_data, "ensure_rid2_archive", fake_ensure)
    monkeypatch.setattr(obstacle_data, "prepare_rid2_dataset", fake_prepare)

    cli.main(["download-obstacles", "--raw-dir", str(raw_dir)])
    download_report = _printed_json(capsys)
    assert calls["metadata"] == raw_dir
    assert calls["ensure"] == raw_dir
    assert download_report == {
        "archive": str(archive),
        "metadata": str(raw_dir / "source-metadata.json"),
    }

    output = tmp_path / "rid2-prepared"
    cli.main(
        [
            "prepare-obstacles",
            "--archive",
            str(archive),
            "--output",
            str(output),
            "--seed",
            "19",
            "--max-images",
            "3",
        ]
    )
    report = _printed_json(capsys)
    assert calls["prepare"] == (archive, output, 19, 3)
    assert report["class_counts"] == {"chimney": 4}
    assert report["skipped_polylines"] == 1


def test_train_command_forwards_config_overrides_and_resume(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    config = tmp_path / "train.yaml"
    data_yaml = tmp_path / "prepared" / "dataset.yaml"
    resume = tmp_path / "run" / "weights" / "last.pt"
    calls: dict[str, object] = {}

    def fake_train(config_path, overrides, resume_path):
        calls.update(config=Path(config_path), overrides=overrides, resume=resume_path)
        return {"status": "completed", "best_weights": str(tmp_path / "best.pt")}

    monkeypatch.setattr(training, "train", fake_train)
    cli.main(
        [
            "train",
            "--config",
            str(config),
            "--data",
            str(data_yaml),
            "--epochs",
            "3",
            "--imgsz",
            "640",
            "--batch",
            "2",
            "--workers",
            "0",
            "--device",
            "cpu",
            "--resume",
            str(resume),
        ]
    )
    report = _printed_json(capsys)
    assert calls["config"] == config
    assert calls["resume"] == resume
    assert calls["overrides"] == {
        "epochs": 3,
        "imgsz": 640,
        "batch": 2,
        "workers": 0,
        "device": "cpu",
        "data": str(data_yaml.resolve()),
    }
    assert report["status"] == "completed"


def test_evaluate_command_uses_current_weights_and_filters_large_report_fields(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    weights = tmp_path / "artifacts" / "models" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")
    manifest = weights.parent / "current.json"
    manifest.write_text(json.dumps({"best_weights": str(weights)}))
    data_yaml = tmp_path / "dataset.yaml"
    calls: dict[str, object] = {}

    def fake_evaluate(weights_path, data_path, split, confidence, device, imgsz, batch):
        calls.update(
            weights=Path(weights_path),
            data=Path(data_path),
            split=split,
            confidence=confidence,
            device=device,
            imgsz=imgsz,
            batch=batch,
        )
        return {
            "metrics": {"mask_map50": 0.8},
            "rows": [{"image": "large-row"}],
            "worst_examples": ["large-example"],
        }

    monkeypatch.setattr(evaluation, "evaluate", fake_evaluate)
    cli.main(
        [
            "evaluate",
            "--data",
            str(data_yaml),
            "--split",
            "val",
            "--confidence",
            "0.4",
            "--device",
            "cpu",
            "--imgsz",
            "640",
            "--batch",
            "2",
        ]
    )
    report = _printed_json(capsys)
    assert calls == {
        "weights": weights,
        "data": data_yaml,
        "split": "val",
        "confidence": 0.4,
        "device": "cpu",
        "imgsz": 640,
        "batch": 2,
    }
    assert report == {"metrics": {"mask_map50": 0.8}}


def test_predict_command_writes_overlay_and_polygon_json(monkeypatch, tmp_path, capsys):
    image_path = tmp_path / "roof.png"
    Image.new("RGB", (20, 12), color=(120, 100, 80)).save(image_path)
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"checkpoint")
    output_dir = tmp_path / "predictions"
    calls: dict[str, object] = {}
    detections = [inference.Detection("solar_panel", 0.91, ((1.0, 1.0), (10.0, 1.0), (5.0, 8.0)))]

    def fake_predict(image, weights_path, *, confidence, device, imgsz):
        calls.update(
            image=image,
            weights=Path(weights_path),
            confidence=confidence,
            device=device,
            imgsz=imgsz,
        )
        return detections

    monkeypatch.setattr(inference, "predict", fake_predict)
    monkeypatch.setattr(inference, "overlay", lambda image, found: image.copy())
    monkeypatch.setattr(training, "sha256", lambda path: "sha256-fixture")

    cli.main(
        [
            "predict",
            str(image_path),
            "--weights",
            str(weights),
            "--output",
            str(output_dir),
            "--confidence",
            "0.4",
            "--device",
            "cpu",
            "--imgsz",
            "320",
        ]
    )
    report = _printed_json(capsys)
    overlay_path = output_dir / "roof.overlay.jpg"
    polygon_path = output_dir / "roof.json"
    assert calls["weights"] == weights
    assert calls["confidence"] == 0.4
    assert calls["device"] == "cpu"
    assert calls["imgsz"] == 320
    assert calls["image"].size == (20, 12)
    assert report == {
        "detections": 1,
        "overlay": str(overlay_path),
        "polygons": str(polygon_path),
    }
    assert overlay_path.is_file()
    polygon_report = json.loads(polygon_path.read_text())
    assert polygon_report["image"] == str(image_path.resolve())
    assert polygon_report["width"] == 20
    assert polygon_report["height"] == 12
    assert polygon_report["weights_sha256"] == "sha256-fixture"
    assert polygon_report["detections"][0]["class"] == "solar_panel"


def test_fetch_site_command_writes_orthophoto_weather_and_site_json(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    location = sources.GeocodeResult(
        query="Hauptstrasse 1 Brugg",
        label="Hauptstrasse 1, 5200 Brugg",
        latitude=47.4831,
        longitude=8.2071,
        easting=2_657_000.0,
        northing=1_258_000.0,
        feature_id="fixture-1",
        source_url="https://geo.example/search",
        source_fetched_at="2026-09-10T00:00:00Z",
    )
    roof = sources.RoofFeature(
        feature_id=77,
        geometry={
            "type": "Polygon",
            "coordinates": [[[1000.0, 2000.0], [1010.0, 2000.0], [1010.0, 2010.0], [1000.0, 2000.0]]],
        },
        properties={"flaeche": 100.0, "neigung": 30.0, "ausrichtung": 0.0},
        source_url="https://geo.example/roof",
        source_data_date="2024-01-01",
        source_fetched_at="2026-09-10T00:00:00Z",
    )
    weather = pd.DataFrame(
        {"ghi": [0.0, 200.0], "dni": [0.0, 150.0], "dhi": [0.0, 50.0]},
        index=pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
    )
    weather_metadata = {"source": "fixture-pvgis", "location": {"elevation": 360.0}}
    calls: dict[str, object] = {}

    def fake_geocode(query):
        calls["query"] = query
        return location

    def fake_roofs(latitude, longitude):
        calls["roof_point"] = (latitude, longitude)
        return [roof]

    def fake_orthophoto(bbox, width, height):
        calls["orthophoto"] = (bbox, width, height)
        return sources.OrthophotoResult(
            image_bytes=_jpeg_bytes(),
            content_type="image/jpeg",
            bbox2056=tuple(bbox),
            width=width,
            height=height,
            gsd_m=(0.5, 0.5),
            north_up=True,
            source_url="https://geo.example/wms",
            source_data_date="2025-01-01",
            source_fetched_at="2026-09-10T00:00:00Z",
        )

    def fake_weather(latitude, longitude, *, cache_dir):
        calls["weather"] = (latitude, longitude, Path(cache_dir))
        return weather, weather_metadata

    monkeypatch.setattr(sources, "geocode", fake_geocode)
    monkeypatch.setattr(sources, "fetch_roofs", fake_roofs)
    monkeypatch.setattr(sources, "fetch_orthophoto", fake_orthophoto)
    monkeypatch.setattr(sources, "fetch_weather", fake_weather)

    output = tmp_path / "site"
    cli.main(["fetch-site", "Hauptstrasse 1 Brugg", "--output", str(output), "--pixels", "64", "--weather"])
    report = _printed_json(capsys)
    assert calls["query"] == "Hauptstrasse 1 Brugg"
    assert calls["roof_point"] == (location.latitude, location.longitude)
    bbox, width, height = calls["orthophoto"]
    assert bbox == (955.0, 1955.0, 1055.0, 2055.0)
    assert (width, height) == (64, 64)
    assert calls["weather"] == (location.latitude, location.longitude, tmp_path / "data" / "cache")
    assert report["roof_planes"] == 1
    assert Path(report["orthophoto"]).is_file()
    assert (output / "orthophoto.jpg").is_file()
    assert (output / "weather.csv").is_file()
    site_report = json.loads((output / "site.json").read_text())
    assert site_report["location"]["label"] == location.label
    assert site_report["roofs"][0]["feature_id"] == 77
    assert site_report["weather"] == weather_metadata


def test_fetch_site_without_roofs_uses_geocode_projection_fallback(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    location = sources.GeocodeResult(
        query="Brugg",
        label="Brugg",
        latitude=47.48,
        longitude=8.2,
        easting=2_657_100.0,
        northing=1_258_100.0,
        feature_id=None,
        source_url="fixture",
        source_fetched_at="now",
    )
    calls: dict[str, object] = {}

    monkeypatch.setattr(sources, "geocode", lambda query: location)
    monkeypatch.setattr(sources, "fetch_roofs", lambda latitude, longitude: [])

    def fake_orthophoto(bbox, width, height):
        calls["bbox"] = bbox
        return sources.OrthophotoResult(
            _jpeg_bytes(), "image/jpeg", tuple(bbox), width, height, (0.1, 0.1), True, "fixture", None, "now"
        )

    monkeypatch.setattr(sources, "fetch_orthophoto", fake_orthophoto)
    output = tmp_path / "empty-site"
    cli.main(["fetch-site", "Brugg", "--output", str(output), "--pixels", "16"])
    _printed_json(capsys)
    assert calls["bbox"] == (2_657_050.0, 1_258_050.0, 2_657_150.0, 1_258_150.0)
    assert json.loads((output / "site.json").read_text())["roofs"] == []


def test_simulate_command_writes_hourly_csv_summary_and_propagates_elevation(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    weather = pd.DataFrame({"ghi": [0.0], "dni": [0.0], "dhi": [0.0]})
    metadata = {"source": "fixture-pvgis", "location": {"elevation": 405.0}}
    calls: dict[str, object] = {}

    def fake_weather(latitude, longitude, *, cache_dir):
        calls["weather"] = (latitude, longitude, Path(cache_dir))
        return weather, metadata

    def fake_config(**kwargs):
        calls["config"] = kwargs
        return kwargs

    def fake_simulate(weather_frame, config):
        calls["simulate"] = (weather_frame, config)
        return SimpleNamespace(
            summary={"annual_energy_kwh": 123.4},
            hourly=pd.DataFrame({"ac_power_w": [10.0, 20.0]}, index=pd.date_range("2026-01-01", periods=2, freq="h")),
        )

    monkeypatch.setattr(sources, "fetch_weather", fake_weather)
    monkeypatch.setattr(physics, "PVConfig", fake_config)
    monkeypatch.setattr(physics, "simulate_pv", fake_simulate)

    output = tmp_path / "simulation"
    cli.main(
        [
            "simulate",
            "--latitude",
            "47.5",
            "--longitude",
            "8.2",
            "--area",
            "24.5",
            "--tilt",
            "28",
            "--azimuth",
            "175",
            "--output",
            str(output),
        ]
    )
    report = _printed_json(capsys)
    assert calls["weather"] == (47.5, 8.2, tmp_path / "data" / "cache")
    assert calls["config"] == {
        "latitude": 47.5,
        "longitude": 8.2,
        "available_area_m2": 24.5,
        "tilt_deg": 28.0,
        "azimuth_deg": 175.0,
        "weather_source": "fixture-pvgis",
        "elevation_m": 405.0,
    }
    assert calls["simulate"][1] == calls["config"]
    assert report == {"annual_energy_kwh": 123.4, "weather_metadata": metadata}
    assert (output / "hourly.csv").is_file()
    assert json.loads((output / "summary.json").read_text()) == report


def test_doctor_command_reports_environment_and_model_manifest(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    manifest = tmp_path / "artifacts" / "models" / "current.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"status": "completed", "best_weights": "best.pt"}))
    fake_torch = SimpleNamespace(
        __version__="fixture-torch",
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr("importlib.metadata.version", lambda package: "fixture-ultralytics")
    monkeypatch.setattr(runtime, "choose_device", lambda: "mps")

    cli.main(["doctor"])
    report = _printed_json(capsys)
    assert report == {
        "root": str(tmp_path),
        "torch": "fixture-torch",
        "ultralytics": "fixture-ultralytics",
        "mps_available": True,
        "cuda_available": False,
        "selected_device": "mps",
        "trained_model": {"status": "completed", "best_weights": "best.pt"},
    }


def test_cli_wraps_download_runtime_errors_as_system_exit(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "ROOT", tmp_path)

    def fail_download(raw):
        raise RuntimeError("fixture network failure")

    monkeypatch.setattr(data, "download_dataset", fail_download)
    with pytest.raises(SystemExit, match=r"^Error: fixture network failure$"):
        cli.main(["download-data", "--raw-dir", str(tmp_path / "raw")])


def test_evaluate_without_current_manifest_is_a_clear_cli_error(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    with pytest.raises(SystemExit, match=r"^Error: No completed training run\."):
        cli.main(["evaluate"])


def test_analyze_roofs_cli_forwards_exact_extent_and_controls(monkeypatch, tmp_path, capsys):
    from rooftop_pv import site_analysis
    calls = []

    def analyze(bounds, output, **kwargs):
        calls.append((bounds, output, kwargs))
        return {"status": "completed", "roof_count": 39}

    monkeypatch.setattr(site_analysis, "run_site_analysis", analyze)
    destination = tmp_path / "geneva"
    cli.main(["analyze-roofs", "--bbox", "2499900.123", "1117200.456", "2500000.123", "1117300.456",
              "--output", str(destination), "--geneva", "--device", "cpu", "--setback", "0.5", "--fill-ratio", "0.8"])
    assert calls == [([2499900.123, 1117200.456, 2500000.123, 1117300.456], destination,
                      {"include_geneva": True, "device": "cpu", "setback_m": .5, "fill_ratio": .8})]
    assert _printed_json(capsys)["roof_count"] == 39


def test_recalculate_roofs_cli_reuses_saved_predictions(tmp_path, monkeypatch, capsys):
    from rooftop_pv import site_analysis
    source, output = tmp_path / "source", tmp_path / "output"

    def replay(actual_source, actual_output):
        assert (actual_source, actual_output) == (source, output)
        return {"roof_count": 39}

    monkeypatch.setattr(site_analysis, "recalculate_site_analysis", replay)
    cli.main(["recalculate-roofs", str(source), "--output", str(output)])
    assert _printed_json(capsys)["roof_count"] == 39
