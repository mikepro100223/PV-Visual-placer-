"""Local download, training, evaluation, inference and solar simulation commands."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rooftop_pv.runtime import ROOT


def _json(value):
    print(json.dumps(value, indent=2, default=str, ensure_ascii=False))


def _current_weights() -> Path:
    manifest = ROOT / "artifacts" / "models" / "current.json"
    if not manifest.is_file():
        raise FileNotFoundError("No completed training run. Run rooftop-pv train first.")
    return Path(json.loads(manifest.read_text())["best_weights"])


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="YOLO11 Swiss rooftop PV potential")
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check environment, accelerator and local model status")
    download = sub.add_parser("download-data", help="Download official Swiss PV labels and provenance")
    download.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    prepare = sub.add_parser("prepare-data", help="Prepare a geographic train/validation/test split")
    prepare.add_argument("--archive", type=Path)
    prepare.add_argument("--output", type=Path, default=ROOT / "data" / "processed" / "swiss-pv")
    prepare.add_argument("--seed", type=int, default=42)
    obstacles_download = sub.add_parser("download-obstacles", help="Download and verify RID2 (6.96 GB)")
    obstacles_download.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw" / "rid2")
    obstacles_prepare = sub.add_parser("prepare-obstacles", help="Convert RID2 roof-centred labels")
    obstacles_prepare.add_argument("--archive", type=Path)
    obstacles_prepare.add_argument("--output", type=Path, default=ROOT / "data" / "processed" / "rid2")
    obstacles_prepare.add_argument("--seed", type=int, default=42)
    obstacles_prepare.add_argument("--max-images", type=int, help="Explicit smoke subset; default all 1819")
    train_parser = sub.add_parser("train", help="Train YOLO11 instance segmentation locally")
    train_parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train.yaml")
    train_parser.add_argument("--data", type=Path)
    for name in ("epochs", "imgsz", "batch", "workers"):
        train_parser.add_argument(f"--{name}", type=int)
    train_parser.add_argument("--device", default=None)
    train_parser.add_argument("--resume", type=Path, help="Resume an interrupted last.pt checkpoint")
    evaluation = sub.add_parser("evaluate", help="Measure segmentation, area errors and quality gates")
    evaluation.add_argument("--weights", type=Path)
    evaluation.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "swiss-pv" / "dataset.yaml")
    evaluation.add_argument("--split", choices=("val", "test"), default="test")
    evaluation.add_argument("--confidence", type=float, default=.25)
    evaluation.add_argument("--device", default="auto")
    evaluation.add_argument("--imgsz", type=int, default=512)
    evaluation.add_argument("--batch", type=int, default=8)
    inference = sub.add_parser("predict", help="Segment a real aerial image and save overlay/polygons")
    inference.add_argument("image", type=Path)
    inference.add_argument("--weights", type=Path)
    inference.add_argument("--output", type=Path, default=ROOT / "artifacts" / "predictions")
    inference.add_argument("--confidence", type=float, default=.25)
    inference.add_argument("--device", default="auto")
    inference.add_argument("--imgsz", type=int, default=512)
    source = sub.add_parser("fetch-site", help="Fetch real Sonnendach geometry and Swiss aerial image")
    source.add_argument("address", help="Swiss street address, e.g. Hauptstrasse 1 5200 Brugg")
    source.add_argument("--output", type=Path, default=ROOT / "data" / "sites" / "brugg")
    source.add_argument("--pixels", type=int, default=1024)
    source.add_argument("--weather", action="store_true")
    batch_analysis = sub.add_parser("analyze-roofs", help="Analyze every Sonnendach roof intersecting an LV95 extent")
    batch_analysis.add_argument("--bbox", type=float, nargs=4, required=True,
                                metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    batch_analysis.add_argument("--output", type=Path, required=True)
    batch_analysis.add_argument("--geneva", action="store_true", help="Include the incomplete SITG superstructure inventory")
    batch_analysis.add_argument("--device", default="auto")
    batch_analysis.add_argument("--setback", type=float, default=.3)
    batch_analysis.add_argument("--fill-ratio", type=float, default=.85)
    replay = sub.add_parser("recalculate-roofs", help="Recalculate saved roof geometry without downloading or running models")
    replay.add_argument("source", type=Path)
    replay.add_argument("--output", type=Path, required=True)
    simulation = sub.add_parser("simulate", help="Run a physical PV scenario with live/cached TMY weather")
    simulation.add_argument("--latitude", type=float, default=47.4831)
    simulation.add_argument("--longitude", type=float, default=8.2071)
    simulation.add_argument("--area", type=float, required=True, help="Usable sloped module area in m²")
    simulation.add_argument("--tilt", type=float, default=30)
    simulation.add_argument("--azimuth", type=float, default=180, help="Compass degrees, south=180")
    simulation.add_argument("--output", type=Path, default=ROOT / "artifacts" / "simulation")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "doctor":
            import importlib.metadata
            import torch
            from rooftop_pv.runtime import choose_device
            manifest = ROOT / "artifacts" / "models" / "current.json"
            _json({"root": ROOT, "torch": torch.__version__,
                   "ultralytics": importlib.metadata.version("ultralytics"),
                   "mps_available": torch.backends.mps.is_available(),
                   "cuda_available": torch.cuda.is_available(), "selected_device": choose_device(),
                   "trained_model": json.loads(manifest.read_text()) if manifest.is_file() else None})
        elif arguments.command == "download-data":
            from rooftop_pv.data import download_dataset
            _json({"archive": download_dataset(arguments.raw_dir)})
        elif arguments.command == "prepare-data":
            from rooftop_pv.data import download_dataset, prepare_dataset
            archive = arguments.archive or download_dataset(ROOT / "data" / "raw")
            _json(asdict(prepare_dataset(archive, arguments.output, seed=arguments.seed)))
        elif arguments.command == "download-obstacles":
            from rooftop_pv.obstacle_data import ensure_rid2_archive, fetch_rid2_metadata
            metadata = fetch_rid2_metadata(arguments.raw_dir)
            _json({"archive": ensure_rid2_archive(arguments.raw_dir), "metadata": metadata})
        elif arguments.command == "prepare-obstacles":
            from rooftop_pv.obstacle_data import ensure_rid2_archive, prepare_rid2_dataset
            archive = arguments.archive or ensure_rid2_archive(ROOT / "data" / "raw" / "rid2")
            _json(asdict(prepare_rid2_dataset(archive, arguments.output, seed=arguments.seed,
                                            max_images=arguments.max_images)))
        elif arguments.command == "train":
            from rooftop_pv.training import train
            overrides = {key: getattr(arguments, key) for key in
                         ("epochs", "imgsz", "batch", "workers", "device")}
            if arguments.data:
                overrides["data"] = str(arguments.data.resolve())
            _json(train(arguments.config, overrides, arguments.resume))
        elif arguments.command == "evaluate":
            from rooftop_pv.evaluation import evaluate
            report = evaluate(arguments.weights or _current_weights(), arguments.data,
                              arguments.split, arguments.confidence, arguments.device,
                              arguments.imgsz, arguments.batch)
            _json({key: value for key, value in report.items() if key not in {"rows", "worst_examples"}})
        elif arguments.command == "predict":
            from PIL import Image
            from rooftop_pv.inference import overlay, predict
            from rooftop_pv.training import sha256
            weights = arguments.weights or _current_weights()
            with Image.open(arguments.image) as source:
                image = source.convert("RGB")
            detections = predict(image, weights, confidence=arguments.confidence,
                                 device=arguments.device, imgsz=arguments.imgsz)
            arguments.output.mkdir(parents=True, exist_ok=True)
            output = arguments.output / arguments.image.stem
            overlay(image, detections).save(output.with_suffix(".overlay.jpg"))
            report = {"image": str(arguments.image.resolve()), "width": image.width,
                      "height": image.height, "weights_sha256": sha256(weights),
                      "coordinate_system": "original image pixels; x right, y down",
                      "confidence_threshold": arguments.confidence,
                      "detections": [detection.as_dict() for detection in detections]}
            output.with_suffix(".json").write_text(json.dumps(report, indent=2))
            _json({"detections": len(detections), "overlay": output.with_suffix(".overlay.jpg"),
                   "polygons": output.with_suffix(".json")})
        elif arguments.command == "fetch-site":
            from shapely.geometry import shape
            from shapely.ops import unary_union
            from rooftop_pv.sources import fetch_orthophoto, fetch_roofs, fetch_weather, geocode
            location = geocode(arguments.address)
            roofs = fetch_roofs(location.latitude, location.longitude)
            if roofs:
                bounds = unary_union([shape(roof.geometry) for roof in roofs]).bounds
                xmin, ymin, xmax, ymax = bounds
                # Preserve roughly the 10 cm/px Swiss training scale on small roofs.
                side = max(100, max(xmax - xmin, ymax - ymin) + 20)
                x, y = (xmin + xmax) / 2, (ymin + ymax) / 2
            else:
                x, y, side = location.easting, location.northing, 100
            image = fetch_orthophoto((x-side/2, y-side/2, x+side/2, y+side/2),
                                    arguments.pixels, arguments.pixels)
            arguments.output.mkdir(parents=True, exist_ok=True)
            (arguments.output / "orthophoto.jpg").write_bytes(image.image_bytes)
            report = {"location": asdict(location), "roofs": [asdict(roof) for roof in roofs],
                      "orthophoto": {k: v for k, v in asdict(image).items() if k != "image_bytes"}}
            if arguments.weather:
                weather, metadata = fetch_weather(location.latitude, location.longitude,
                                                   cache_dir=ROOT / "data" / "cache")
                weather.to_csv(arguments.output / "weather.csv")
                report["weather"] = metadata
            (arguments.output / "site.json").write_text(json.dumps(report, indent=2))
            _json({"site": arguments.output / "site.json", "roof_planes": len(roofs),
                   "orthophoto": arguments.output / "orthophoto.jpg"})
        elif arguments.command == "analyze-roofs":
            from rooftop_pv.site_analysis import run_site_analysis
            _json(run_site_analysis(arguments.bbox, arguments.output,
                                    include_geneva=arguments.geneva, device=arguments.device,
                                    setback_m=arguments.setback, fill_ratio=arguments.fill_ratio))
        elif arguments.command == "recalculate-roofs":
            from rooftop_pv.site_analysis import recalculate_site_analysis
            _json(recalculate_site_analysis(arguments.source, arguments.output))
        elif arguments.command == "simulate":
            from rooftop_pv.physics import PVConfig, simulate_pv
            from rooftop_pv.sources import fetch_weather
            weather, metadata = fetch_weather(arguments.latitude, arguments.longitude,
                                               cache_dir=ROOT / "data" / "cache")
            config = PVConfig(latitude=arguments.latitude, longitude=arguments.longitude,
                              available_area_m2=arguments.area, tilt_deg=arguments.tilt,
                              azimuth_deg=arguments.azimuth, weather_source=metadata["source"],
                              elevation_m=metadata.get("location", {}).get("elevation", 0.0))
            result = simulate_pv(weather, config)
            arguments.output.mkdir(parents=True, exist_ok=True)
            result.hourly.to_csv(arguments.output / "hourly.csv")
            report = dict(result.summary, weather_metadata=metadata)
            (arguments.output / "summary.json").write_text(json.dumps(report, indent=2))
            _json(report)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
