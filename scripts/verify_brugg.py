"""Reproduce a real, local site -> segmentation -> PV scenario smoke check.

Run fetch-site first. This uses downloaded imagery, real trained checkpoints and
cached/live PVGIS data; unlike unit tests it does not mock models or weather.
It verifies software integration, not the physical correctness of model masks.
"""

import argparse
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import mapping, shape

from rooftop_pv.inference import PV_CLASSES, pixel_to_map, predict, predict_tiled_obstacles
from rooftop_pv.physics import PVConfig, simulate_pv
from rooftop_pv.runtime import ROOT
from rooftop_pv.sources import RoofFeature, fetch_weather
from rooftop_pv.training import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=ROOT / "data/sites/brugg-pvgis53")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/demo/brugg")
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    spec = importlib.util.spec_from_file_location("dashboard_smoke", ROOT / "code/app.py")
    dashboard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dashboard)
    site = json.loads((arguments.site / "site.json").read_text())
    if not site["roofs"]:
        raise ValueError("The site has no roof geometry; this is not a complete site check.")
    roof = RoofFeature(**site["roofs"][0])
    geometry = shape(roof.geometry)
    bbox = tuple(site["orthophoto"]["bbox2056"])
    with Image.open(arguments.site / "orthophoto.jpg") as source:
        image = source.convert("RGB")
    model_status = dashboard.read_model_status(ROOT)
    obstacle_status = dashboard.read_obstacle_model_status(ROOT)
    if not model_status["available"]:
        raise ValueError("A completed PV checkpoint is required for this verification.")
    pv = [d for d in predict(image, Path(model_status["weights"]), device=arguments.device,
                            imgsz=512, confidence=.25) if d.class_name in PV_CLASSES]
    obstacles = []
    if obstacle_status["available"]:
        obstacles = predict_tiled_obstacles(image, Path(obstacle_status["weights"]), bbox=bbox,
                                            device=arguments.device, imgsz=640, confidence=.25)
    metrics, warnings = dashboard._metric_summary(
        roof_geometry=geometry, manual_area_m2=None, exclusions=[], pv_detections=pv,
        obstacle_detections=obstacles, bbox=bbox, image_width=image.width,
        image_height=image.height, tilt_deg=roof.tilt_deg, setback_m=.3, fill_ratio=.85,
        module_area_m2=1.134 * 1.722, module_efficiency=.21,
    )
    coordinates = site["location"]
    weather, metadata = fetch_weather(coordinates["latitude"], coordinates["longitude"],
                                      cache_dir=ROOT / "data/cache")
    weather = dashboard.normalise_weather(weather)
    config = PVConfig(
        latitude=coordinates["latitude"], longitude=coordinates["longitude"],
        available_area_m2=metrics["panel_area_budget_m2"], module_area_m2=1.134 * 1.722,
        module_efficiency=.21, tilt_deg=roof.tilt_deg, azimuth_deg=roof.azimuth_deg,
        weather_source=metadata["source"], elevation_m=metadata["location"]["elevation"],
    )
    result = simulate_pv(weather, config)
    monthly = dashboard._monthly_frame(result)
    assert len(weather) == 8760 and len(monthly) == 12
    assert np.isfinite(result.summary["energy_kwh"])
    assert result.summary["module_count"] == metrics["module_count"]
    assert metrics["usable_roof_plane_area_m2"] <= metrics["gross_roof_plane_area_m2"]
    assert np.isclose(monthly["Energie (kWh)"].sum(), result.summary["energy_kwh"])
    arguments.output.mkdir(parents=True, exist_ok=True)
    dashboard._scene_overlay(image, pv, obstacles, geometry, [], bbox).save(
        arguments.output / "roof-and-detections.jpg")
    features = [{"type": "Feature", "geometry": mapping(geometry),
                 "properties": {"role": "selected_roof", "source_id": roof.feature_id}}]
    for role, detections in (("existing_pv", pv), ("obstacle", obstacles)):
        for detection in detections:
            footprint = pixel_to_map(detection.polygon, bbox, image.width, image.height)
            clipped = footprint.intersection(geometry)
            if not clipped.is_empty and clipped.area > 0:
                features.append({"type": "Feature", "geometry": mapping(clipped),
                                 "properties": {"role": role, "class": detection.class_name,
                                                "confidence": detection.confidence}})
    collection = {"type": "FeatureCollection", "features": features,
                  "crs": {"type": "name", "properties": {"name": "EPSG:2056"}}}
    (arguments.output / "roof-exclusions-lv95.geojson").write_text(json.dumps(collection, indent=2))
    result.hourly.to_csv(arguments.output / "hourly.csv")
    monthly.to_csv(arguments.output / "monthly.csv", index=False)
    models = {}
    for role, status in (("pv", model_status), ("obstacles", obstacle_status)):
        models[role] = {"available": status["available"], "quality_status": status["quality_status"],
                        "weights_sha256": sha256(Path(status["weights"])) if status["available"] else None}
    report = {
        "pipeline_completed": True,
        "scope": "Software integration check using real sources; not a verified installation plan.",
        "exclusions_exhaustive": False,
        "site": str(arguments.site), "roof": asdict(roof), "models": models,
        "inference": {"device": arguments.device, "confidence": .25, "pv_imgsz": 512,
                      "obstacle_imgsz": 640, "obstacle_tile_span_m": 40.96,
                      "pv_detections": len(pv),
                      "obstacle_detections": len(obstacles)},
        "areas_and_capacity": metrics, "physics": result.summary, "weather": metadata,
        "warnings": warnings + ["No manual on-site verification of the detected or missing obstacles.",
                                  "Unmodelled local shade and roof structural suitability remain unknown."]
    }
    report_path = arguments.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"report": str(report_path), "models": models, "metrics": metrics,
                      "energy_kwh": result.summary["energy_kwh"], "monthly_rows": len(monthly)}, indent=2))


if __name__ == "__main__":
    main()
