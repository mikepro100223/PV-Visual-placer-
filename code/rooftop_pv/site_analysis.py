"""Complete image-to-Sonnendach batch analysis for an explicit bounded area.

Every intersecting roof is a target. Extra imagery covers complete target roofs;
overlapping 100 m source tiles preserve the Swiss PV training context. Outputs
are geometric candidates, never a construction approval or an accuracy certificate.
"""

import csv
import hashlib
import json
import math
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw
from shapely.geometry import box, shape
from shapely.ops import unary_union

from rooftop_pv import sources
from rooftop_pv.geneva import GenevaSuperstructureResult, fetch_superstructures, superstructure_exclusions
from rooftop_pv.inference import (
    OBSTACLE_CLASSES, PV_CLASSES, Detection, load_segmenter, pixel_to_map,
    predict_tiled_obstacles, predict_tiled_pv,
)
from rooftop_pv.runtime import ROOT
from rooftop_pv.training import sha256

GSD_M = .1
TILE_PIXELS = 1000
MAX_TILES = 64


def plan_tiles(bounds):
    """Snap a complete frame to 10 cm pixels and cover it with 100 m tiles."""
    if len(bounds) != 4 or not all(math.isfinite(v) for v in bounds):
        raise ValueError("Four finite LV95 bounds required")
    xmin, ymin, xmax, ymax = bounds
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("Bounds must have positive width and height")
    left, bottom = math.floor(xmin * 10), math.floor(ymin * 10)
    right, top = math.ceil(xmax * 10), math.ceil(ymax * 10)
    right, top = max(right, left + TILE_PIXELS), max(top, bottom + TILE_PIXELS)
    width, height = right - left, top - bottom

    def starts(length):
        last = length - TILE_PIXELS
        return sorted(set([*range(0, last + 1, 800), last]))

    columns, rows = starts(width), starts(height)
    if len(columns) * len(rows) > MAX_TILES:
        raise ValueError("Complete roof coverage requires more than 64 source tiles; choose a smaller query area")
    frames = [((left + x) / 10, (bottom + y) / 10,
               (left + x + TILE_PIXELS) / 10, (bottom + y + TILE_PIXELS) / 10)
              for y in rows for x in columns]
    return (left / 10, bottom / 10, right / 10, top / 10), (width, height), frames


def shift_detection(detection, tile_bbox, frame_bbox):
    """Map one native 10 cm tile to north-up virtual-frame pixel coordinates."""
    dx = round((tile_bbox[0] - frame_bbox[0]) / GSD_M, 6)
    dy = round((frame_bbox[3] - tile_bbox[3]) / GSD_M, 6)
    return Detection(detection.class_name, detection.confidence,
                     tuple((x + dx, y + dy) for x, y in detection.polygon))


def registered_models():
    """Fail closed on missing, substituted or wrongly classified checkpoints."""
    result = {}
    for role, filename, supported in (("pv", "current.json", PV_CLASSES),
                                       ("obstacles", "obstacles.json", OBSTACLE_CLASSES)):
        path = ROOT / "artifacts/models" / filename
        manifest = json.loads(path.read_text())
        weights = Path(manifest["best_weights"])
        if not weights.is_absolute():
            weights = ROOT / weights
        digest = sha256(weights)
        if digest != manifest.get("best_weights_sha256"):
            raise ValueError(f"{role} checkpoint hash differs from its registered evaluation")
        classes = {str(name).lower().replace(" ", "_") for name in manifest["classes"].values()}
        if not classes.intersection(supported):
            raise ValueError(f"{role} checkpoint has no supported classes")
        result[role] = {"weights": str(weights), "weights_sha256": digest,
                        "quality_status": manifest.get("quality_status", "unverified"),
                        "classes": manifest["classes"], "registry": str(path),
                        "test_evaluation": manifest.get("test_evaluation"),
                        "validation_evaluation": manifest.get("val_evaluation")}
    return result


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _export_roof_areas(output, combined, model_only):
    rows = list(combined["rows"].values())
    _write_json(output / "model-only.json", model_only)
    _write_json(output / "analysis.json", combined)
    collection = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:2056"}},
                  "features": [{"type": "Feature", "id": str(row["feature_id"]),
                                "geometry": row["usable_geometry_lv95"],
                                "properties": {key: value for key, value in row.items() if "geometry" not in key}}
                               for row in rows]}
    _write_json(output / "usable-roofs-lv95.geojson", collection)
    with (output / "roofs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (dict, list, tuple)) else value
                          for key, value in row.items()} for row in rows)


def recalculate_site_analysis(source: Path, output: Path):
    """Replay saved predictions and source vectors with the current area code only.

    Original reports and imagery are retained. This does not repeat inference,
    change its quality evidence, or fetch fresher vectors or aerial imagery.
    """
    from rooftop_pv.roof_batch import analyze_roofs

    source, output = Path(source).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    inputs = ["report.json", "sources.json", "predictions.json"]
    report = json.loads((source / "report.json").read_text())
    source_data = json.loads((source / "sources.json").read_text())
    predictions = json.loads((source / "predictions.json").read_text())
    roofs = [sources.RoofFeature(**value) for value in source_data["roofs"]]
    if sorted(str(roof.feature_id) for roof in roofs) != sorted(report["target_roof_ids"]):
        raise ValueError("Saved source roof IDs differ from the original report")
    if predictions["bbox2056"] != report["image_bbox2056"] or predictions["image_size"] != report["image_size"]:
        raise ValueError("Saved prediction coordinates differ from the original report")
    inventory = []
    if report.get("geneva_inventory_features") is not None:
        inputs.append("geneva-inventory.json")
        saved_inventory = json.loads((source / "geneva-inventory.json").read_text())
        inventory = superstructure_exclusions(GenevaSuperstructureResult(**saved_inventory))
    params = report["parameters"]
    parameters = {"bbox2056": predictions["bbox2056"], "image_size": predictions["image_size"],
                  "pv_detections": predictions["pv"], "obstacle_detections": predictions["obstacles"],
                  "setback_m": params["setback_m"], "pixel_size_m": params["gsd_m"], "fill_ratio": params["fill_ratio"]}
    model_only = analyze_roofs(roofs, **parameters)
    combined = analyze_roofs(roofs, external_exclusions=inventory, **parameters)
    rows = list(combined["rows"].values())
    output.mkdir(parents=True)
    _export_roof_areas(output, combined, model_only)
    for name in inputs[1:]:
        shutil.copy2(source / name, output / name)
    if (source / "overview.jpg").is_file():
        shutil.copy2(source / "overview.jpg", output / "overview.jpg")
    errors = sum(bool(row["errors"]) for row in rows)
    complete = sum(row["full_coverage"] is True and not row["errors"] for row in rows)
    report.update(created_utc=datetime.now(timezone.utc).isoformat(),
                  status="completed" if not errors and complete == len(roofs) else "partial",
                  roof_error_count=errors, complete_roof_count=complete,
                  replay_source=str(source), replay_input_sha256={name: sha256(source / name) for name in inputs},
                  original_analysis_source_sha256=report["analysis_source_sha256"],
                  analysis_source_sha256=sha256(Path(__file__)),
                  roof_batch_source_sha256=sha256(Path(__file__).with_name("roof_batch.py")),
                  geometry_source_sha256=sha256(Path(__file__).with_name("geometry.py")))
    _write_json(output / "report.json", report)
    return report


def _draw_geometry(canvas, geometry, frame, color):
    if geometry is None or geometry.is_empty:
        return
    if geometry.geom_type != "Polygon":
        for part in getattr(geometry, "geoms", ()):
            _draw_geometry(canvas, part, frame, color)
        return
    xmin, ymin, xmax, ymax = frame
    draw = ImageDraw.Draw(canvas)
    for ring in (geometry.exterior, *geometry.interiors):
        points = [((x - xmin) * canvas.width / (xmax - xmin),
                   (ymax - y) * canvas.height / (ymax - ymin)) for x, y, *_ in ring.coords]
        draw.line(points, fill=color, width=2)


def _agreement(roofs, predictions, inventory):
    roof_union = unary_union([shape(roof.geometry) for roof in roofs])
    ai = unary_union(predictions).intersection(roof_union)
    reference = unary_union(inventory).intersection(roof_union)
    combined = ai.union(reference)
    return {"scope": "Spatial agreement with incomplete dated SITG inventory, NOT accuracy/recall",
            "inventory_is_exhaustive": False, "imagery_temporally_aligned": False,
            "ai_area_m2": ai.area, "inventory_area_m2": reference.area,
            "intersection_m2": ai.intersection(reference).area,
            "union_iou": ai.intersection(reference).area / combined.area if combined.area else None}


def run_site_analysis(query_bbox2056, output: Path, *, device="cpu", include_geneva=False,
                      setback_m=.3, fill_ratio=.85):
    """Save every target roof, source tile, prediction and geometric area result.

    Existing output directories are never overwritten. The query covers at most
    500 m per side and the complete target roof imagery at most 64 one-megapixel
    source tiles. Model loading, requests and data errors are not replaced by zeroes.
    """
    from rooftop_pv.roof_batch import analyze_roofs

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    bbox = tuple(float(value) for value in query_bbox2056)
    if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox):
        raise ValueError("Four finite query bounds required")
    if not (2400000 <= bbox[0] < bbox[2] <= 2900000 and 1000000 <= bbox[1] < bbox[3] <= 1400000):
        raise ValueError("Query must use Swiss LV95 metres, not longitude/latitude")
    if max(bbox[2] - bbox[0], bbox[3] - bbox[1]) > 500:
        raise ValueError("Query exceeds the explicit 500 m side limit")
    if not math.isfinite(setback_m) or not 0 <= setback_m <= 20:
        raise ValueError("Setback must be finite and between 0 and 20 m")
    if not math.isfinite(fill_ratio) or not 0 < fill_ratio <= 1:
        raise ValueError("Fill ratio must be in (0, 1]")
    roofs = sources.fetch_roofs_bbox(bbox)
    if not roofs:
        raise ValueError("No Sonnendach roof planes intersect the requested area")
    target_ids = [str(roof.feature_id) for roof in roofs]
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("Duplicate target roof IDs")
    target_geometry = unary_union([shape(roof.geometry) for roof in roofs])
    extent = target_geometry.union(box(*bbox)).bounds
    frame, size, tiles = plan_tiles((extent[0] - 10, extent[1] - 10,
                                     extent[2] + 10, extent[3] + 10))
    checkpoints = registered_models()
    models = {role: load_segmenter(Path(info["weights"])) for role, info in checkpoints.items()}
    inventory_result = fetch_superstructures(frame, cache_dir=ROOT / "data/cache/geneva") if include_geneva else None
    inventory = superstructure_exclusions(inventory_result) if inventory_result else []
    output.mkdir(parents=True)
    (output / "tiles").mkdir()
    scale = min(1., 2000 / max(size))
    canvas = Image.new("RGB", (round(size[0] * scale), round(size[1] * scale)))
    detections = {"pv": [], "obstacles": []}
    tile_sources = []
    for index, tile in enumerate(tiles, 1):
        orthophoto = sources.fetch_orthophoto(tile, TILE_PIXELS, TILE_PIXELS)
        with orthophoto.open_pil() as source:
            image = source.convert("RGB")
        if image.size != (TILE_PIXELS, TILE_PIXELS) or tuple(orthophoto.bbox2056) != tile or not orthophoto.north_up:
            raise ValueError("Source image dimensions, bounds or north-up contract differ from the tile request")
        image_path = output / "tiles" / f"{index:03d}.jpg"
        image_path.write_bytes(orthophoto.image_bytes)
        tile_source = {k: v for k, v in asdict(orthophoto).items() if k != "image_bytes"}
        tile_source.update(path=str(image_path), sha256=hashlib.sha256(orthophoto.image_bytes).hexdigest(),
                           acquisition_date=None, source_data_date_meaning="WMS cache update, not flight date")
        tile_sources.append(tile_source)
        origin = shift_detection(Detection("origin", 1, ((0, 0),)), tile, frame).polygon[0]
        resized = image.resize((round(TILE_PIXELS * scale), round(TILE_PIXELS * scale)))
        canvas.paste(resized, (round(origin[0] * scale), round(origin[1] * scale)))
        for role, infer, resolution in (("pv", predict_tiled_pv, 512),
                                         ("obstacles", predict_tiled_obstacles, 640)):
            local = infer(image, model=models[role], bbox=tile, confidence=.25, device=device, imgsz=resolution)
            detections[role].extend(shift_detection(item, tile, frame) for item in local)
        print(f"Geneva/site source tile {index}/{len(tiles)}", flush=True)

    parameters = {"bbox2056": frame, "image_size": size, "pv_detections": detections["pv"],
                  "obstacle_detections": detections["obstacles"], "setback_m": setback_m,
                  "pixel_size_m": GSD_M, "fill_ratio": fill_ratio}
    model_only = analyze_roofs(roofs, **parameters)
    combined = analyze_roofs(roofs, external_exclusions=inventory, **parameters)
    rows = list(combined["rows"].values())
    if len(rows) != len(roofs):
        raise RuntimeError("Batch output lost target roof rows")
    world_predictions = {}
    for role, values in detections.items():
        world_predictions[role] = [pixel_to_map(d.polygon, frame, *size) for d in values]
        for geometry in world_predictions[role]:
            _draw_geometry(canvas, geometry, frame, "#19cf9a" if role == "pv" else "#fa5353")
    for geometry in inventory:
        _draw_geometry(canvas, geometry, frame, "#ffac32")
    for roof in roofs:
        _draw_geometry(canvas, shape(roof.geometry), frame, "#579bff")
    canvas.save(output / "overview.jpg")
    _export_roof_areas(output, combined, model_only)
    _write_json(output / "sources.json", {"roofs": [asdict(roof) for roof in roofs], "tiles": tile_sources})
    _write_json(output / "predictions.json", {"bbox2056": frame, "image_size": size,
        "coordinate_system": "virtual north-up frame pixels at 0.1 m per pixel",
        **{role: [item.as_dict() for item in values] for role, values in detections.items()}})
    if inventory_result:
        _write_json(output / "geneva-inventory.json", {"geojson": inventory_result.geojson,
                     "provenance": inventory_result.provenance, "pagination": inventory_result.pagination})
    roof_errors = sum(bool(row["errors"]) for row in rows)
    complete_roofs = sum(row["full_coverage"] is True and not row["errors"] for row in rows)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "status": "completed" if not roof_errors and complete_roofs == len(roofs) else "partial",
              "accuracy_validated": False, "scope": "Geometric area candidates for every roof intersecting query bbox",
              "query_bbox2056": bbox, "image_bbox2056": frame, "image_size": size,
              "target_roof_ids": target_ids, "roof_count": len(roofs), "tile_count": len(tiles),
              "complete_roof_count": complete_roofs, "roof_error_count": roof_errors,
              "models": checkpoints, "inference_source_sha256": sha256(Path(__file__).with_name("inference.py")),
              "analysis_source_sha256": sha256(Path(__file__)),
              "roof_batch_source_sha256": sha256(Path(__file__).with_name("roof_batch.py")),
              "geometry_source_sha256": sha256(Path(__file__).with_name("geometry.py")),
              "parameters": {"confidence": .25, "pv_imgsz": 512, "obstacle_imgsz": 640,
                             "setback_m": setback_m, "fill_ratio": fill_ratio, "gsd_m": GSD_M},
              "component_counts": {role: len(values) for role, values in detections.items()},
              "geneva_inventory_features": len(inventory_result.features) if inventory_result else None,
              "inventory_agreement": _agreement(roofs, world_predictions["obstacles"], inventory) if include_geneva else None,
              "limitations": ["Both models failed promotion gates; inspect masks before use.",
                  "No exhaustive temporally aligned Geneva image labels; overlap is not accuracy.",
                  "No structural, local shading or installation-safety approval.",
                  "Prediction holes are filled; overlapping roof planes must not be summed blindly."]}
    _write_json(output / "report.json", report)
    return report
