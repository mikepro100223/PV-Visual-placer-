"""Held-out instance metrics plus directly relevant occupied-area error analysis."""

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping, Sequence

import numpy as np
from PIL import Image

from rooftop_pv.inference import Detection, load_segmenter, overlay, predict, union_mask
from rooftop_pv.runtime import ROOT, choose_device
from rooftop_pv.training import dataset_paths, sha256

GATES = {"mask_map50": 0.70, "mask_map50_95": 0.45, "union_iou": 0.65}
PV_CLASS_NAMES = frozenset({"pv", "pv_panel", "photovoltaic", "solar_panel", "solar_panels", "solar-panel"})
OBSTACLE_CLASS_NAMES = frozenset({
    "chimney", "skylight", "dormer", "obstacle", "hvac", "roof_window", "superstructure",
    "tv_dish", "ladder", "balcony", "wall", "other",
})


def label_path(image: Path) -> Path:
    parts = list(image.parts)
    if "images" not in parts:
        raise ValueError(f"Image path must contain an images directory: {image}")
    parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


def class_name_for_id(class_names: Mapping[int, str] | Sequence[str] | None, class_id: int) -> str:
    """Resolve a numeric YOLO class id without relying on dict insertion order."""

    if class_names is None:
        return str(class_id)
    if isinstance(class_names, Mapping):
        if class_id not in class_names:
            raise ValueError(f"Unknown class id {class_id}; available ids: {sorted(class_names)}")
        return str(class_names[class_id])
    if class_id < 0 or class_id >= len(class_names):
        raise ValueError(f"Unknown class id {class_id}; class count is {len(class_names)}")
    return str(class_names[class_id])


def model_class_names(names: Mapping[int, str] | Sequence[str]) -> dict[int, str]:
    """Normalize model names to an explicit numeric-id mapping."""

    if isinstance(names, Mapping):
        return {int(class_id): str(name) for class_id, name in names.items()}
    return {class_id: str(name) for class_id, name in enumerate(names)}


def normalize_class_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def class_role(name: str) -> str:
    """Return a reporting role while retaining every original class separately."""

    normalized = normalize_class_name(name)
    if normalized in {normalize_class_name(value) for value in PV_CLASS_NAMES}:
        return "pv"
    if normalized in {normalize_class_name(value) for value in OBSTACLE_CLASS_NAMES}:
        return "obstacle"
    return "other"


def read_labels(
    path: Path,
    width: int,
    height: int,
    *,
    class_names: Mapping[int, str] | Sequence[str] | None = None,
) -> list[Detection]:
    if not path.is_file():
        raise ValueError(f"Missing label file: {path}; negative images need empty label files.")
    detections = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if len(values) < 7 or len(values) % 2 != 1:
            raise ValueError(f"Invalid segmentation polygon in {path}:{line_number}")
        coordinates = np.asarray(values[1:]).reshape(-1, 2)
        if not np.isfinite(coordinates).all() or (coordinates < 0).any() or (coordinates > 1).any():
            raise ValueError(f"Invalid normalized coordinates in {path}:{line_number}")
        class_id = int(values[0])
        detections.append(Detection(class_name_for_id(class_names, class_id), 1.0,
                                    tuple((float(x * width), float(y * height))
                                          for x, y in coordinates)))
    return detections


def reference_mask(
    image: Path,
    labels: list[Detection],
    width: int,
    height: int,
    *,
    class_name: str | None = None,
    use_original: bool = True,
):
    parts = list(image.parts)
    parts[len(parts) - 1 - parts[::-1].index("images")] = "masks"
    original = Path(*parts).with_suffix(".png")
    # The preserved source raster is binary PV ground truth.  For multi-class
    # evaluation it must never be reused as an obstacle or all-class mask.
    if original.is_file() and use_original and (class_name is None or class_role(class_name) == "pv"):
        with Image.open(original) as source:
            mask = np.asarray(source.convert("L")) > 0
        if mask.shape != (height, width):
            raise ValueError(f"Original mask dimensions differ from image: {original}")
        return mask, "original_source_raster"
    selected = labels if class_name is None else [label for label in labels if label.class_name == class_name]
    return union_mask(selected, width, height), "converted_yolo_polygons"


def mask_scores(truth: np.ndarray, predicted: np.ndarray) -> dict:
    if truth.shape != predicted.shape:
        raise ValueError("Prediction and reference masks have different image dimensions.")
    intersection = int(np.logical_and(truth, predicted).sum())
    union = int(np.logical_or(truth, predicted).sum())
    true_area, pred_area = int(truth.sum()), int(predicted.sum())
    return {"iou": intersection / union if union else 1.0,
            "intersection_pixels": intersection, "union_pixels": union,
            "true_area_pixels": true_area, "predicted_area_pixels": pred_area,
            "false_positive_pixels": pred_area - intersection,
            "false_negative_pixels": true_area - intersection,
            "signed_area_error_pixels": pred_area - true_area,
            "absolute_area_error_fraction_of_image": abs(pred_area - true_area) / truth.size,
            "relative_area_error": (pred_area - true_area) / true_area if true_area else None}


def _finite_metric(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def native_mask_metrics(segment_metrics: object, names: Mapping[int, str] | Sequence[str]) -> dict[str, dict]:
    """Map Ultralytics native mask metrics to model names by numeric class id.

    Ultralytics stores AP rows in ``ap_class_index`` order, which is not safe
    to zip with ``model.names.values()`` when a class is absent from a split.
    Every configured class is emitted; absent classes retain ``None`` metrics.
    """

    class_map = model_class_names(names)
    indexes = np.asarray(getattr(segment_metrics, "ap_class_index", ()), dtype=int).reshape(-1)
    precision = np.asarray(getattr(segment_metrics, "p", ()), dtype=float).reshape(-1)
    recall = np.asarray(getattr(segment_metrics, "r", ()), dtype=float).reshape(-1)
    ap50 = np.asarray(getattr(segment_metrics, "ap50", ()), dtype=float).reshape(-1)
    ap = np.asarray(getattr(segment_metrics, "ap", ()), dtype=float).reshape(-1)
    if not len(indexes) and len(class_map) == len(ap):
        # This fallback is only valid when the metric explicitly contains all
        # configured classes (e.g. a lightweight test double).
        indexes = np.arange(len(class_map), dtype=int)
    lengths = {len(indexes), len(precision), len(recall), len(ap50), len(ap)}
    nonzero_lengths = {length for length in lengths if length}
    if len(nonzero_lengths) > 1:
        raise ValueError("Ultralytics per-class mask metric arrays have inconsistent lengths")
    if len(indexes) and any(class_id not in class_map for class_id in indexes):
        raise ValueError(f"Native mask metrics reference unknown class ids: {indexes.tolist()}")
    rows: dict[str, dict] = {}
    by_id = {int(class_id): position for position, class_id in enumerate(indexes)}
    def value_at(values: np.ndarray, position: int | None) -> float | None:
        return _finite_metric(values[position]) if position is not None and position < len(values) else None

    for class_id, name in class_map.items():
        position = by_id.get(class_id)
        rows[name] = {
            "class_id": class_id,
            "role": class_role(name),
            "mask_precision": value_at(precision, position),
            "mask_recall": value_at(recall, position),
            "mask_ap50": value_at(ap50, position),
            "mask_ap50_95": value_at(ap, position),
        }
    return rows


def aggregate_pixel_scores(scores: Sequence[dict]) -> dict:
    """Aggregate image-level mask/area scores without mixing classes."""

    intersection = sum(int(score["intersection_pixels"]) for score in scores)
    union = sum(int(score["union_pixels"]) for score in scores)
    true_area = sum(int(score["true_area_pixels"]) for score in scores)
    predicted_area = sum(int(score["predicted_area_pixels"]) for score in scores)
    positives = [score for score in scores if score["true_area_pixels"] > 0]
    return {
        "images": len(scores),
        "positive_images": len(positives),
        "intersection_pixels": intersection,
        "union_pixels": union,
        "true_area_pixels": true_area,
        "predicted_area_pixels": predicted_area,
        "iou": intersection / union if union else 1.0,
        "mean_positive_image_iou": float(np.mean([score["iou"] for score in positives]))
        if positives else None,
        "signed_area_error_pixels": predicted_area - true_area,
        "mean_absolute_area_error_fraction_of_image": float(
            np.mean([score["absolute_area_error_fraction_of_image"] for score in scores])
        )
        if scores else None,
        "relative_area_error": (predicted_area - true_area) / true_area if true_area else None,
    }


def per_class_pixel_metrics(rows: Sequence[dict], class_names: Mapping[int, str] | Sequence[str]) -> dict[str, dict]:
    """Aggregate pixel IoU and occupied-area errors for each class independently."""

    names = tuple(model_class_names(class_names).values())
    return {
        name: aggregate_pixel_scores([row["class_scores"][name] for row in rows])
        for name in names
    }


def role_union_metrics(rows: Sequence[dict], role: str) -> dict:
    """Aggregate PV or obstacle unions kept separate from all other classes."""

    return aggregate_pixel_scores([row["role_scores"][role] for row in rows])


def dataset_manifest_hashes(data_yaml: Path) -> dict[str, str | None]:
    """Hash split and source manifests when the prepared dataset provides them."""

    data_yaml = Path(data_yaml).resolve()
    splits = data_yaml.parent / "splits.json"
    source_candidates = (
        data_yaml.parent / "source-metadata.json",
        data_yaml.parent.parent.parent / "raw" / "source-metadata.json",
        data_yaml.parent.parent.parent / "raw" / "source_manifest.json",
    )
    source = next((candidate for candidate in source_candidates if candidate.is_file()), None)
    return {
        "dataset_splits_sha256": sha256(splits) if splits.is_file() else None,
        "dataset_source_manifest_sha256": sha256(source) if source is not None else None,
    }


def quality_gate(metrics: dict, *, require_obstacle: bool = False) -> dict:
    gates = dict(GATES)
    if require_obstacle:
        # Obstacle-aware runs must demonstrate the same pre-declared 0.65
        # occupied-area quality for the obstacle role as for the primary union.
        # Swiss single-class PV evaluation does not have an obstacle gate.
        gates["obstacle_union_iou"] = GATES["union_iou"]
    checks = {key: {"value": metrics.get(key), "minimum": minimum,
                    "passed": isinstance(metrics.get(key), (float, int))
                    and np.isfinite(metrics[key]) and metrics[key] >= minimum}
              for key, minimum in gates.items()}
    return {"passed": all(item["passed"] for item in checks.values()), "checks": checks,
            "scope": "Held-out dataset benchmark; not a Swiss deployment certification."}


def evaluate(weights: Path, data_yaml: Path, split: str = "test", confidence: float = .25,
             device: str = "auto", imgsz: int = 512, batch: int = 8) -> dict:
    if split not in {"val", "test"}:
        raise ValueError("Evaluation must use val or test, never train.")
    images = dataset_paths(data_yaml.resolve(), split)
    output_dir = ROOT / "artifacts" / "evaluations" / (
        weights.parent.parent.name + "-" + split + "-" +
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output_dir.mkdir(parents=True)
    model = load_segmenter(weights)
    selected_device = choose_device(device)
    # AP uses the conventional low collection threshold; area uses the stated operating threshold.
    ap = model.val(data=str(data_yaml.resolve()), split=split, device=selected_device,
                   imgsz=imgsz, batch=batch, workers=0, conf=.001, plots=True,
                   project=str(output_dir), name="instance_metrics", verbose=False)
    class_names = model_class_names(model.names)
    class_name_values = tuple(class_names.values())
    pv_classes = tuple(name for name in class_name_values if class_role(name) == "pv")
    obstacle_classes = tuple(name for name in class_name_values if class_role(name) == "obstacle")
    single_pv_class = len(class_name_values) == 1 and len(pv_classes) == 1
    rows = []
    for index, image_path in enumerate(images):
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        truth_detections = read_labels(
            label_path(image_path), image.width, image.height, class_names=class_names
        )
        predictions = predict(image, model=model, confidence=confidence,
                              device=selected_device, imgsz=imgsz)
        # Keep PV and obstacle role references independent.  The primary area
        # scope below is the legacy Swiss PV raster only for single-class runs;
        # multiclass runs use a separate all-class union.
        role_truth: dict[str, np.ndarray] = {}
        role_predicted: dict[str, np.ndarray] = {}
        role_reference_types: dict[str, str] = {}
        for role, names_for_role in {"pv": pv_classes, "obstacle": obstacle_classes}.items():
            selected_truth = [item for item in truth_detections if item.class_name in names_for_role]
            selected_predictions = [item for item in predictions if item.class_name in names_for_role]
            if role == "pv" and pv_classes:
                role_truth[role], role_reference_types[role] = reference_mask(
                    image_path, selected_truth, image.width, image.height,
                    use_original=single_pv_class,
                )
            else:
                role_truth[role], role_reference_types[role] = reference_mask(
                    image_path, selected_truth, image.width, image.height,
                    class_name=None, use_original=False
                )
            role_predicted[role] = union_mask(selected_predictions, image.width, image.height)
        if single_pv_class:
            truth, reference_type = role_truth["pv"], role_reference_types["pv"]
            predicted = role_predicted["pv"]
        else:
            # Multi-class area quality is an all-class union.  A binary Swiss
            # source raster is intentionally excluded here: it contains PV
            # semantics only and cannot certify obstacle predictions.
            truth, reference_type = reference_mask(
                image_path, truth_detections, image.width, image.height, use_original=False
            )
            predicted = union_mask(predictions, image.width, image.height)
        score = mask_scores(truth, predicted)
        class_scores = {}
        for name in class_name_values:
            selected_truth = [item for item in truth_detections if item.class_name == name]
            selected_predictions = [item for item in predictions if item.class_name == name]
            class_truth, _ = reference_mask(
                image_path,
                selected_truth,
                image.width,
                image.height,
                class_name=name,
                # A binary original raster is safe for a single PV class only.
                use_original=(len(pv_classes) == 1 and name in pv_classes),
            )
            class_scores[name] = mask_scores(
                class_truth,
                union_mask(selected_predictions, image.width, image.height),
            )
        score.update(image=str(image_path), true_instances=len(truth_detections),
                     predicted_instances=len(predictions), image_pixels=truth.size,
                     reference_type=reference_type, class_scores=class_scores,
                     role_scores={role: mask_scores(role_truth[role], role_predicted[role])
                                  for role in ("pv", "obstacle")})
        rows.append(score)
        if index % 50 == 0:
            print(f"Area evaluation {index + 1}/{len(images)}", flush=True)
    positives = [row for row in rows if row["true_area_pixels"] > 0]
    negatives = [row for row in rows if row["true_area_pixels"] == 0]
    intersection = sum(row["intersection_pixels"] for row in rows)
    union = sum(row["union_pixels"] for row in rows)
    native_per_class = native_mask_metrics(ap.seg, class_names)
    pixel_per_class = per_class_pixel_metrics(rows, class_names)
    role_unions = {role: role_union_metrics(rows, role) for role in ("pv", "obstacle")}
    metrics = {"mask_map50": float(ap.seg.map50), "mask_map50_95": float(ap.seg.map),
               "mask_precision": float(ap.seg.mp), "mask_recall": float(ap.seg.mr),
               "union_iou": intersection / union if union else 1.,
               "mean_positive_image_iou": float(np.mean([row["iou"] for row in positives]))
               if positives else None,
               "mean_absolute_area_error_fraction_of_image":
                   float(np.mean([row["absolute_area_error_fraction_of_image"] for row in rows])),
               "negative_image_false_positive_rate":
                   sum(row["predicted_area_pixels"] > 0 for row in negatives) / len(negatives)
                   if negatives else None,
               "per_class": {name: {"native_mask": native_per_class[name],
                                     "pixel": pixel_per_class[name]}
                             for name in class_name_values},
               "role_unions": role_unions,
               "obstacle_union_iou": role_unions["obstacle"]["iou"],
               # Keep a flat alias for consumers that only need native AP/P/R.
               "native_per_class": native_per_class}
    slices = {}
    for name, subset in {"positive": positives, "negative": negatives,
                         "small_coverage_under_1pct": [r for r in positives
                           if r["true_area_pixels"] / r["image_pixels"] < .01],
                         "larger_coverage": [r for r in positives
                           if r["true_area_pixels"] / r["image_pixels"] >= .01]}.items():
        slices[name] = {"images": len(subset), "mean_iou":
                       float(np.mean([r["iou"] for r in subset])) if subset else None}
    failures_dir = output_dir / "review"
    failures_dir.mkdir()
    worst = sorted(rows, key=lambda row: row["iou"])[:12]
    for rank, row in enumerate(worst, 1):
        with Image.open(row["image"]) as source:
            image = source.convert("RGB")
        predictions = predict(image, model=model, confidence=confidence,
                              device=selected_device, imgsz=imgsz)
        truth = read_labels(label_path(Path(row["image"])), image.width, image.height,
                            class_names=class_names)
        canvas = Image.new("RGB", (2 * image.width, image.height))
        canvas.paste(overlay(image, truth), (0, 0))
        canvas.paste(overlay(image, predictions), (image.width, 0))
        canvas.save(failures_dir / f"{rank:02d}-{Path(row['image']).stem}.jpg")
    references = sorted({row["reference_type"] for row in rows})
    if single_pv_class and references == ["original_source_raster"]:
        area_metric_scope = "single_pv_original_source_raster"
    elif references == ["converted_yolo_polygons"]:
        area_metric_scope = "all_classes_yolo_polygons"
    else:
        area_metric_scope = "mixed_original_and_yolo_polygons"
    metrics["area_metric_scope"] = area_metric_scope
    report = {"split": split, "weights": str(weights.resolve()),
              "weights_sha256": sha256(weights), "dataset_yaml_sha256": sha256(data_yaml),
              "evaluation_source_sha256": sha256(Path(__file__)),
              "inference_source_sha256": sha256(Path(__file__).with_name("inference.py")),
              **dataset_manifest_hashes(data_yaml),
              "evaluated_utc": datetime.now(timezone.utc).isoformat(),
              "images": len(rows), "positive_images": len(positives),
              "negative_images": len(negatives), "operating_confidence": confidence,
              "imgsz": imgsz, "class_names": class_names,
              "class_roles": {name: class_role(name) for name in class_name_values},
              "area_metric_scope": area_metric_scope,
              "metrics": metrics, "slices": slices,
              "gate": quality_gate(metrics, require_obstacle=bool(obstacle_classes)),
              "rows": rows, "worst_examples": worst,
              "empty_mask_baseline_positive_iou": 0.0 if positives else None,
              "full_mask_baseline_union_iou": sum(r["true_area_pixels"] for r in rows)
                  / sum(r["image_pixels"] for r in rows),
              "area_reference": references,
              "review_layout": "Converted ground-truth polygons left, model prediction right. "
                  "Area metrics use original raster masks where provided.",
              "report_path": str(output_dir / "report.json")}
    Path(report["report_path"]).write_text(json.dumps(report, indent=2, allow_nan=False))
    for index in (ROOT / "artifacts" / "models").glob("*.json"):
        manifest = json.loads(index.read_text())
        if manifest.get("best_weights_sha256") == report["weights_sha256"]:
            manifest[f"{split}_evaluation"] = report["report_path"]
            if split == "test":
                manifest["quality_status"] = ("dataset_gates_passed" if report["gate"]["passed"]
                                              else "experimental_gates_failed")
            index.write_text(json.dumps(manifest, indent=2))
    return report
