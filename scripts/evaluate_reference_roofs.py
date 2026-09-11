"""Run a production-parallel behavioral regression on three roof references.

The reference images do not have complete ground truth. This evaluator therefore
reports mask behavior and promotion gates, never accuracy. Model inference is
delegated to :mod:`app.inference` so its padding, tiling, confidence thresholds,
and Swiss-PV/RID-obstacle ownership stay authoritative.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import inference as app_inference
from app.detections import (
    arbitrate_detections,
    normalise,
    polygon_parts,
    world_to_pixel,
)

OUTPUT_DIR = ROOT / "artifacts/iteration-20260911/reference-regression"
PV_CONFIDENCE = 0.25
OBSTACLE_CONFIDENCE = 0.30
PV_REFERENCE_ID = "pv_reference"
LARGE_RID_MASK_FRACTION = 0.005
AREA_COMPARISON_TOLERANCE = 0.0001
EXPECTED_PRODUCTION_HASHES = {
    "rid": "ac8d2e68698f48140f8407628bb45269c39eeb95a22deaa89f1ebd95e8279a93",
    "swiss": "086c85ef1a8cc078241a56044fd60aeb0dd7e6c8bea06ba217821128c3aa83e2",
}

REFERENCES = {
    "flat_roof_small_objects": ROOT
    / "data/cache/fe175ce92e739095059657f7a3cd42fa3756cd42d51bd11f6979fb830ddb3567.jpg",
    PV_REFERENCE_ID: ROOT
    / "data/cache/0d84d4a8f7564fbc153448e3ed76d2b9a46d4eafae8a53b2eae645ab38fa7e2c.jpg",
    "hip_roof_small_objects": ROOT
    / "data/cache/fea6c0f018cc46e4e9f19ac4bbf02a343bcdc874e542d560085e3a6d6bfd08b0.jpg",
}

SYSTEMS = {
    "production": ROOT / "models/rid_best.pt",
    "rid_1024_small_object": ROOT
    / "runs/segment/artifacts/iteration-20260911/training/rid-1024-small-object/weights/best.pt",
    "rid_swiss_pv_hard_negative_epoch0": ROOT
    / "runs/segment/artifacts/iteration-20260911/training/rid-swiss-pv-hard-negative/weights/epoch0.pt",
}

PROMOTION_RULES = {
    "swiss_pv_present_on_pv_reference": (
        "At least one displayed Swiss-owned PV mask must remain on pv_reference."
    ),
    "large_rid_obstacle_masks_do_not_grow": (
        "For each image, a largest RID obstacle mask at or above 0.5% of the image "
        "must not exceed production by more than the 0.01 percentage-point numeric tolerance."
    ),
    "small_skylights_and_chimneys_do_not_disappear": (
        "Every image/class pair where production displays a skylight or chimney must "
        "retain at least one such RID mask; no exception is documented for these candidates."
    ),
    "six_roof_gate_passes": (
        "scripts/evaluate_hard_cases.py must pass at PV confidence 0.25 and obstacle "
        "confidence 0.30 for the exact checkpoint."
    ),
}

COLORS = {
    "pv_installation": (30, 110, 255, 105),
    "skylight": (255, 145, 0, 120),
    "chimney": (255, 90, 0, 135),
    "dormer": (255, 175, 30, 120),
    "other_obstacle": (235, 75, 20, 120),
    "ladder": (255, 205, 40, 120),
    "tree": (255, 120, 25, 120),
    "shadow": (150, 105, 55, 90),
    "roof": (85, 205, 120, 75),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _union(geometries):
    geometries = list(geometries)
    return unary_union(geometries) if geometries else Polygon()


def _sources(item):
    return set(item.get("sources", [item.get("source", item.get("model", "unknown"))]))


def _rounded(value):
    return round(float(value), 6)


def _overlap_metrics(detections, image_area, prefix):
    swiss_pv = _union(
        item["geometry"]
        for item in detections
        if item["label"] == "pv_installation" and "swiss" in _sources(item)
    )
    rid_obstacles = _union(
        item["geometry"]
        for item in detections
        if item["label"] not in {"pv_installation", "roof"} and "rid" in _sources(item)
    )
    overlap = swiss_pv.intersection(rid_obstacles).area
    return {
        f"{prefix}_swiss_pv_vs_rid_obstacles_pixels": _rounded(overlap),
        f"{prefix}_swiss_pv_vs_rid_obstacles_fraction": _rounded(overlap / image_area),
        f"{prefix}_overlap_relative_to_swiss_pv": _rounded(
            overlap / swiss_pv.area if swiss_pv.area else 0.0
        ),
    }


def summarise_image(raw_detections, displayed_detections, image_size):
    """Summarise post-arbitration masks and pre/post Swiss/RID overlap."""
    image_area = float(image_size[0] * image_size[1])
    classes = {}
    labels = sorted({item["label"] for item in displayed_detections if item["label"] != "roof"})
    for label in labels:
        items = [item for item in displayed_detections if item["label"] == label]
        geometries = [item["geometry"] for item in items]
        confidences = [item.get("confidence") for item in items if item.get("confidence") is not None]
        total_area = _union(geometries).area
        largest_area = max((geometry.area for geometry in geometries), default=0.0)
        sources = sorted({source for item in items for source in _sources(item)})
        classes[label] = {
            "mask_count": len(items),
            "total_mask_area_pixels": _rounded(total_area),
            "total_mask_area_fraction": _rounded(total_area / image_area),
            "largest_mask_area_pixels": _rounded(largest_area),
            "largest_mask_area_fraction": _rounded(largest_area / image_area),
            "max_confidence": _rounded(max(confidences)) if confidences else None,
            "sources": sources,
        }
    overlap = _overlap_metrics(raw_detections, image_area, "raw")
    overlap.update(_overlap_metrics(displayed_detections, image_area, "displayed"))
    return {
        "raw_detection_count": len(raw_detections),
        "displayed_detection_count": len(displayed_detections),
        "classes": classes,
        "overlap": overlap,
    }


def infer_reference_image(image, obstacle_checkpoint, confidence, obstacle_confidence):
    """Call the real application inference and arbitration with one RID override."""
    previous = os.environ.get("OBSTACLE_MODEL_PATH")
    os.environ["OBSTACLE_MODEL_PATH"] = str(Path(obstacle_checkpoint).resolve())
    bounds = (0.0, 0.0, float(image.width), float(image.height))
    try:
        raw = normalise(
            app_inference.predict(image, bounds, confidence, obstacle_confidence)
        )
        displayed = arbitrate_detections(raw)
    finally:
        if previous is None:
            os.environ.pop("OBSTACLE_MODEL_PATH", None)
        else:
            os.environ["OBSTACLE_MODEL_PATH"] = previous
    return raw, displayed


def render_overlay(image, detections, output_path, title):
    """Draw app-displayed masks; blue is Swiss PV and orange is RID obstacle."""
    canvas = image.convert("RGBA")
    bounds = (0.0, 0.0, float(image.width), float(image.height))
    for item in detections:
        color = COLORS.get(item["label"], (230, 80, 190, 110))
        for world_part in polygon_parts(item["geometry"]):
            part = world_to_pixel(world_part, bounds, image.size)
            layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(layer)
            exterior = [(round(x), round(y)) for x, y in part.exterior.coords]
            draw.polygon(exterior, fill=color, outline=color[:3] + (255,), width=2)
            for interior in part.interiors:
                hole = [(round(x), round(y)) for x, y in interior.coords]
                draw.polygon(hole, fill=(0, 0, 0, 0))
            centroid = part.representative_point()
            confidence = item.get("confidence")
            suffix = "" if confidence is None else f" {confidence:.3f}"
            draw.text(
                (round(centroid.x), round(centroid.y)),
                f"{item['label']}{suffix}",
                fill=(255, 255, 255, 255),
                stroke_width=2,
                stroke_fill=(0, 0, 0, 255),
            )
            canvas = Image.alpha_composite(canvas, layer)
    ImageDraw.Draw(canvas).text(
        (6, 6),
        title,
        fill=(255, 255, 255, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0, 255),
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_path, format="PNG")


def _largest_rid_obstacle_fraction(image_result):
    values = [
        metrics["largest_mask_area_fraction"]
        for label, metrics in image_result.get("classes", {}).items()
        if label not in {"pv_installation", "roof"} and "rid" in metrics.get("sources", [])
    ]
    return max(values, default=0.0)


def promotion_checks(production_images, candidate_images, six_roof_passed):
    """Apply concrete behavioral gates against the production reference."""
    pv_metrics = candidate_images.get(PV_REFERENCE_ID, {}).get("classes", {}).get(
        "pv_installation", {}
    )
    pv_present = pv_metrics.get("mask_count", 0) > 0 and "swiss" in pv_metrics.get(
        "sources", []
    )

    large_violations = []
    for image_id, production in production_images.items():
        baseline = _largest_rid_obstacle_fraction(production)
        candidate = _largest_rid_obstacle_fraction(candidate_images.get(image_id, {}))
        if (
            max(baseline, candidate) >= LARGE_RID_MASK_FRACTION
            and candidate > baseline + AREA_COMPARISON_TOLERANCE
        ):
            large_violations.append(
                {
                    "image": image_id,
                    "production_largest_fraction": _rounded(baseline),
                    "candidate_largest_fraction": _rounded(candidate),
                }
            )

    small_violations = []
    for image_id, production in production_images.items():
        candidate_classes = candidate_images.get(image_id, {}).get("classes", {})
        for label in ("skylight", "chimney"):
            baseline = production.get("classes", {}).get(label, {})
            if (
                baseline.get("mask_count", 0) > 0
                and "rid" in baseline.get("sources", [])
                and candidate_classes.get(label, {}).get("mask_count", 0) == 0
            ):
                small_violations.append({"image": image_id, "class": label})

    checks = {
        "swiss_pv_present_on_pv_reference": pv_present,
        "large_rid_obstacle_masks_do_not_grow": not large_violations,
        "small_skylights_and_chimneys_do_not_disappear": not small_violations,
        "six_roof_gate_passes": bool(six_roof_passed),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "violations": {
            "large_rid_obstacle_masks_do_not_grow": large_violations,
            "small_skylights_and_chimneys_do_not_disappear": small_violations,
        },
    }


def load_six_roof_gate(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    parameters = data.get("parameters", {})
    if parameters.get("confidence") != PV_CONFIDENCE:
        raise ValueError(f"{path}: six-roof PV confidence is not {PV_CONFIDENCE}")
    if parameters.get("obstacle_confidence") != OBSTACLE_CONFIDENCE:
        raise ValueError(
            f"{path}: six-roof obstacle confidence is not {OBSTACLE_CONFIDENCE}"
        )
    quality = data.get("quality_gates", {})
    return {
        "source": str(path),
        "sha256": sha256_file(path),
        "generated_at": data.get("generated_at"),
        "parameters": parameters,
        "passed": quality.get("passed") is True,
        "checks": quality.get("checks", {}),
    }


def _parse_gate_arguments(values):
    gates = {}
    for value in values:
        try:
            system, raw_path = value.split("=", 1)
        except ValueError as exc:
            raise ValueError(f"Expected SYSTEM=JSON for --six-roof-gate: {value}") from exc
        if system not in SYSTEMS:
            raise ValueError(f"Unknown system for --six-roof-gate: {system}")
        gates[system] = load_six_roof_gate(raw_path)
    return gates


def _validate_inputs():
    for path in [*REFERENCES.values(), *SYSTEMS.values(), ROOT / "models/swiss_best.pt"]:
        if not path.is_file():
            raise FileNotFoundError(path)
    actual_rid = sha256_file(SYSTEMS["production"])
    actual_swiss = sha256_file(ROOT / "models/swiss_best.pt")
    if actual_rid != EXPECTED_PRODUCTION_HASHES["rid"]:
        raise ValueError(f"Production RID hash mismatch: {actual_rid}")
    if actual_swiss != EXPECTED_PRODUCTION_HASHES["swiss"]:
        raise ValueError(f"Production Swiss hash mismatch: {actual_swiss}")


def evaluate_system(system_id, checkpoint):
    images = {}
    for image_id, image_path in REFERENCES.items():
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        raw, displayed = infer_reference_image(
            image, checkpoint, PV_CONFIDENCE, OBSTACLE_CONFIDENCE
        )
        overlay_path = OUTPUT_DIR / "overlays" / f"{system_id}--{image_id}.png"
        render_overlay(image, displayed, overlay_path, f"{system_id} / {image_id}")
        images[image_id] = {
            "image": str(image_path.relative_to(ROOT)),
            "image_sha256": sha256_file(image_path),
            "width": image.width,
            "height": image.height,
            "overlay": str(overlay_path.relative_to(ROOT)),
            **summarise_image(raw, displayed, image.size),
        }
    return {
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "images": images,
    }


def build_report(gates):
    _validate_inputs()
    systems = {
        system_id: evaluate_system(system_id, checkpoint)
        for system_id, checkpoint in SYSTEMS.items()
    }
    production_images = systems["production"]["images"]
    promotions = {}
    for system_id, result in systems.items():
        gate = gates.get(system_id)
        promotions[system_id] = promotion_checks(
            production_images,
            result["images"],
            gate is not None and gate["passed"],
        )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_type": "reference_behavioral_regression",
        "accuracy_claim": False,
        "interpretation": (
            "These three images have no complete ground truth. Results are reference/"
            "behavioral regression observations, not accuracy measurements."
        ),
        "parameters": {
            "pv_confidence": PV_CONFIDENCE,
            "obstacle_confidence": OBSTACLE_CONFIDENCE,
            "large_rid_mask_fraction": LARGE_RID_MASK_FRACTION,
            "area_comparison_tolerance": AREA_COMPARISON_TOLERANCE,
            "inference_device": os.environ.get(
                "PV_INFERENCE_DEVICE", "cuda:0_if_available_else_cpu"
            ),
        },
        "production_contract": {
            "swiss_checkpoint": "models/swiss_best.pt",
            "swiss_checkpoint_sha256": sha256_file(ROOT / "models/swiss_best.pt"),
            "pv_owner": "swiss when available; RID PV is fallback only",
            "obstacle_owner": "rid",
            "app_inference": "app/inference.py:predict",
            "app_arbitration": "app/detections.py:arbitrate_detections",
            "reference_bounds": "whole image in pixel units; no roof ground-truth clipping",
        },
        "promotion_rules": PROMOTION_RULES,
        "six_roof_gates": gates,
        "systems": systems,
        "promotion_results": promotions,
    }


def _class_rows(report):
    rows = []
    for system_id, system in report["systems"].items():
        for image_id, image in system["images"].items():
            for label, metrics in image["classes"].items():
                confidence = metrics["max_confidence"]
                rows.append(
                    "| {system} | {image} | {label} | {count} | {total:.6f} | "
                    "{largest:.6f} | {confidence} |".format(
                        system=system_id,
                        image=image_id,
                        label=label,
                        count=metrics["mask_count"],
                        total=metrics["total_mask_area_fraction"],
                        largest=metrics["largest_mask_area_fraction"],
                        confidence="n/a" if confidence is None else f"{confidence:.3f}",
                    )
                )
    return rows


def render_report_markdown(report):
    lines = [
        "# Reference roof behavioral regression",
        "",
        report["interpretation"],
        "",
        "## Reproducible contract",
        "",
        f"- PV confidence: `{PV_CONFIDENCE}`",
        f"- Obstacle confidence: `{OBSTACLE_CONFIDENCE}`",
        "- Swiss owns PV while its checkpoint exists; RID owns physical obstacles.",
        "- The evaluator calls `app.inference.predict` and then the app's detection arbitration.",
        "- Bounds span each complete 400 x 400 reference image; complete roof ground truth is unavailable.",
        "",
        "## Checkpoints",
        "",
        "| System | Checkpoint | SHA-256 | Six-roof gate | Promotion gates |",
        "| --- | --- | --- | --- | --- |",
    ]
    for system_id, system in report["systems"].items():
        gate = report["six_roof_gates"].get(system_id)
        gate_status = "not supplied" if gate is None else ("PASS" if gate["passed"] else "FAIL")
        lines.append(
            f"| {system_id} | `{system['checkpoint']}` | `{system['checkpoint_sha256']}` | "
            f"{gate_status} | "
            f"{'PASS' if report['promotion_results'][system_id]['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Per-image, per-class mask behavior",
            "",
            "Total area is the within-class union, relative to the full image. Largest area is the largest individual displayed mask.",
            "",
            "| System | Image | Class | Masks | Total area fraction | Largest fraction | Max confidence |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
            *_class_rows(report),
            "",
            "## Swiss-PV / RID-obstacle overlap",
            "",
            "| System | Image | Raw fraction | Displayed fraction | Raw / Swiss PV |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for system_id, system in report["systems"].items():
        for image_id, image in system["images"].items():
            overlap = image["overlap"]
            lines.append(
                f"| {system_id} | {image_id} | "
                f"{overlap['raw_swiss_pv_vs_rid_obstacles_fraction']:.6f} | "
                f"{overlap['displayed_swiss_pv_vs_rid_obstacles_fraction']:.6f} | "
                f"{overlap['raw_overlap_relative_to_swiss_pv']:.6f} |"
            )
    lines.extend(["", "## Promotion rules and results", ""])
    for key, rule in PROMOTION_RULES.items():
        lines.append(f"- `{key}`: {rule}")
    lines.extend(["", "| System | PV retained | Large masks | Small objects | Six roofs | Result |", "| --- | --- | --- | --- | --- | --- |"])
    for system_id, result in report["promotion_results"].items():
        checks = result["checks"]
        marker = lambda value: "PASS" if value else "FAIL"
        lines.append(
            f"| {system_id} | {marker(checks['swiss_pv_present_on_pv_reference'])} | "
            f"{marker(checks['large_rid_obstacle_masks_do_not_grow'])} | "
            f"{marker(checks['small_skylights_and_chimneys_do_not_disappear'])} | "
            f"{marker(checks['six_roof_gate_passes'])} | {marker(result['passed'])} |"
        )
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            ".venv/bin/python scripts/evaluate_reference_roofs.py \\",
            "  --six-roof-gate production=artifacts/iteration-20260911/reference-regression/six-roof-production-live.json \\",
            "  --six-roof-gate rid_1024_small_object=artifacts/iteration-20260911/hard-cases/candidate.json \\",
            "  --six-roof-gate rid_swiss_pv_hard_negative_epoch0=artifacts/iteration-20260911/hard-cases/epoch0.json",
            ".venv/bin/python -m pytest tests/test_evaluate_reference_roofs.py -q",
            ".venv/bin/python -m pytest tests -q",
            "```",
            "",
            "The exact executed verification results are recorded in `HANDOFF.md` after the final run.",
            "",
        ]
    )
    return "\n".join(lines)


def render_handoff_markdown(report):
    failed = [
        system_id
        for system_id, result in report["promotion_results"].items()
        if not result["passed"]
    ]
    lines = [
        "# Handoff: reference roof behavioral regression",
        "",
        "This is a reference/behavioral regression without complete ground truth; it is not an accuracy evaluation.",
        "",
        "## Outcome",
        "",
        f"- Production checkpoint verified: `{report['systems']['production']['checkpoint_sha256']}`.",
        f"- Swiss checkpoint verified: `{report['production_contract']['swiss_checkpoint_sha256']}`.",
        f"- Systems failing at least one promotion rule: `{', '.join(failed) if failed else 'none'}`.",
        "- No training, download, checkpoint replacement, model promotion, production-file edit, or commit was performed.",
        "",
        "## Outputs",
        "",
        "- `report.json`: complete machine-readable metrics, hashes, rules, and gate evidence.",
        "- `REPORT.md`: human-readable comparison and reproduction commands.",
        "- `overlays/`: nine traceable overlays, one per system/reference pair.",
        "",
        "## Verification log",
        "",
        "Final exact commands and results are appended after test and six-roof execution.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--six-roof-gate",
        action="append",
        default=[],
        metavar="SYSTEM=JSON",
        help="Attach an exact-checkpoint evaluate_hard_cases.py result.",
    )
    parser.add_argument(
        "--device",
        help="Set PV_INFERENCE_DEVICE explicitly; omitted uses the app default.",
    )
    args = parser.parse_args()
    if args.device:
        os.environ["PV_INFERENCE_DEVICE"] = args.device
    gates = _parse_gate_arguments(args.six_roof_gate)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report(gates)
    (OUTPUT_DIR / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "REPORT.md").write_text(
        render_report_markdown(report), encoding="utf-8"
    )
    (OUTPUT_DIR / "HANDOFF.md").write_text(
        render_handoff_markdown(report), encoding="utf-8"
    )
    print(json.dumps({"output": str(OUTPUT_DIR), "promotion_results": report["promotion_results"]}, indent=2))


if __name__ == "__main__":
    main()
