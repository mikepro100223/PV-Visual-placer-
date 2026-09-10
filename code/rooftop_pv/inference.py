"""Original-resolution segmentation masks, overlays and georeferenced exclusions."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from shapely.validation import make_valid

from rooftop_pv.runtime import choose_device, yolo_class

PV_CLASSES = {"solar_panel", "solar_panels", "pv", "pv_panel", "photovoltaic", "solar-panel"}
OBSTACLE_CLASSES = {"chimney", "skylight", "dormer", "obstacle", "hvac", "roof_window",
                    "tv_dish", "ladder", "balcony", "wall", "other", "superstructure"}


@dataclass(frozen=True)
class Detection:
    """One predicted exterior contour in original-image pixel coordinates.

    ``polygon`` intentionally stores an exterior only.  The current public
    detection contract has no hole representation; masks with interior holes
    therefore remain an explicitly documented vectorisation limitation.
    """

    class_name: str
    confidence: float
    polygon: tuple[tuple[float, float], ...]

    def as_dict(self) -> dict:
        return {"class": self.class_name, "confidence": self.confidence,
                "polygon_pixels": self.polygon}


def load_segmenter(weights: Path):
    if not weights.is_file():
        raise FileNotFoundError(f"Trained weights missing: {weights}. Run train first.")
    model = yolo_class()(str(weights), task="segment")
    if model.task != "segment":
        raise ValueError("Only segmentation checkpoints are supported.")
    names = set(str(name).lower().replace(" ", "_") for name in model.names.values())
    if not names & (PV_CLASSES | OBSTACLE_CLASSES):
        raise ValueError(f"Checkpoint has no supported rooftop class (found {sorted(names)}). "
                         "COCO starting weights are not a trained rooftop model.")
    return model


def _mask_array(data: object) -> np.ndarray:
    """Move an Ultralytics mask tensor to a CPU NumPy array without resizing."""

    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    masks = np.asarray(data)
    if masks.ndim != 3:
        raise ValueError(f"Segmentation masks must have shape (N, H, W), got {masks.shape}.")
    return masks


def _mask_components(mask: np.ndarray) -> list[tuple[tuple[float, float], ...]]:
    """Return one exterior contour per disconnected positive mask component.

    Ultralytics' ``masks2segments(strategy='all')`` merges separate contours
    into a single path.  ``RETR_EXTERNAL`` deliberately retains each island as
    a separate exterior while documenting that interior holes are not carried
    by :class:`Detection`.
    """

    binary = np.asarray(mask > 0.5, dtype=np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) < 3 or cv2.contourArea(contour) <= 0:
            continue
        points = tuple((float(point[0][0]), float(point[0][1])) for point in contour)
        polygons.append(points)
    return sorted(polygons, key=lambda points: (min(y for _, y in points), min(x for x, _ in points)))


def predict(image: Image.Image, weights: Path | None = None, *, model=None,
            confidence: float = 0.25, device: str = "auto", imgsz: int = 512) -> list[Detection]:
    if not 0 < confidence <= 1:
        raise ValueError("Confidence must be in (0, 1].")
    if image.width * image.height > 25_000_000:
        raise ValueError("Image is too large; use a georeferenced roof crop or tiles.")
    if model is None:
        if weights is None:
            raise ValueError("Provide a trained checkpoint.")
        model = load_segmenter(weights)
    result = model.predict(image.convert("RGB"), conf=confidence, imgsz=imgsz,
                           device=choose_device(device), retina_masks=True, verbose=False)[0]
    if result.masks is None:
        return []
    masks = _mask_array(result.masks.data)
    if masks.shape[1:] != (image.height, image.width):
        raise ValueError(
            "Segmentation masks are not at original image dimensions: "
            f"{masks.shape[1:]} vs {(image.height, image.width)}."
        )
    class_ids = result.boxes.cls.cpu().tolist()
    confidences = result.boxes.conf.cpu().tolist()
    if len(masks) != len(class_ids) or len(masks) != len(confidences):
        raise ValueError(
            "Segmentation masks, class ids and confidences have inconsistent counts."
        )
    detections: list[Detection] = []
    for mask, class_id, confidence_value in zip(masks, class_ids, confidences):
        for polygon in _mask_components(mask):
            detections.append(
                Detection(str(model.names[int(class_id)]), float(confidence_value), polygon)
            )
    return detections


def _predict_tiled(
    image: Image.Image,
    weights: Path | None,
    *,
    bbox,
    model,
    confidence: float,
    device: str,
    imgsz: int,
    context_m: float,
    allowed_classes: set[str],
    tile_label: str,
    missing_model_label: str,
) -> list[Detection]:
    if len(bbox) != 4 or not np.isfinite(bbox).all():
        raise ValueError("Invalid metric image bounds.")
    xmin, ymin, xmax, ymax = bbox
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("Invalid metric image bounds.")
    if image.width * image.height > 25_000_000:
        raise ValueError("Image is too large; use a smaller crop.")
    tile_width = min(image.width, max(1, round(context_m * image.width / (xmax - xmin))))
    tile_height = min(image.height, max(1, round(context_m * image.height / (ymax - ymin))))

    def starts(length, tile):
        last = length - tile
        return sorted(set([*range(0, last + 1, max(1, round(tile * .8))), last]))

    columns, rows = starts(image.width, tile_width), starts(image.height, tile_height)
    if len(columns) * len(rows) > 64:
        raise ValueError(f"More than 64 {tile_label} tiles; select a smaller roof/image extent.")
    if model is None:
        if weights is None:
            raise ValueError(f"Provide a trained {missing_model_label} checkpoint.")
        model = load_segmenter(weights)
    detections = []
    for top in rows:
        for left in columns:
            crop = image.crop((left, top, left + tile_width, top + tile_height))
            predictions = predict(crop, model=model, confidence=confidence, device=device, imgsz=imgsz)
            detections.extend(
                Detection(prediction.class_name, prediction.confidence,
                          tuple((x + left, y + top) for x, y in prediction.polygon))
                for prediction in predictions if prediction.class_name in allowed_classes
            )
    return detections


def predict_tiled_obstacles(image: Image.Image, weights: Path | None = None, *, bbox,
                            model=None, confidence: float = .25, device: str = "auto",
                            imgsz: int = 640) -> list[Detection]:
    """Infer RID2 at its 40.96 m training context, returning original-image pixels.

    Tiles overlap by 20 percent. Their polygons may overlap and MUST be unioned
    for area accounting; returned polygon count is not a physical object count.
    This preserves source scale, but is not a Swiss-domain accuracy validation.
    """

    return _predict_tiled(
        image, weights, bbox=bbox, model=model, confidence=confidence, device=device,
        imgsz=imgsz, context_m=40.96, allowed_classes=OBSTACLE_CLASSES,
        tile_label="obstacle", missing_model_label="obstacle",
    )


def predict_tiled_pv(image: Image.Image, weights: Path | None = None, *, bbox,
                     model=None, confidence: float = .25, device: str = "auto",
                     imgsz: int = 512) -> list[Detection]:
    """Infer PV at a 100 m training context in original-image pixels.

    Tiles overlap by 20 percent and returned polygons may overlap.  Consumers
    must union them before area accounting; this is an inference-scale adapter,
    not an additional model or domain-quality claim.
    """

    return _predict_tiled(
        image, weights, bbox=bbox, model=model, confidence=confidence, device=device,
        imgsz=imgsz, context_m=100.0, allowed_classes=PV_CLASSES,
        tile_label="PV", missing_model_label="PV",
    )


def overlay(image: Image.Image, detections: list[Detection]) -> Image.Image:
    base = image.convert("RGBA")
    layer = Image.new("RGBA", base.size)
    draw = ImageDraw.Draw(layer)
    for detection in detections:
        points = list(detection.polygon)
        if len(points) < 3:
            continue
        draw.polygon(points, fill=(23, 200, 173, 80), outline=(35, 255, 215, 230), width=2)
        draw.text(points[0], f"{detection.class_name} {detection.confidence:.0%}",
                  fill=(255, 255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0, 255))
    return Image.alpha_composite(base, layer).convert("RGB")


def pixel_to_map(points, bbox: tuple[float, float, float, float], width: int, height: int):
    xmin, ymin, xmax, ymax = bbox
    if width <= 0 or height <= 0 or xmax <= xmin or ymax <= ymin:
        raise ValueError("Invalid image dimensions or projected bounding box.")
    points = np.asarray(points, dtype=float)
    if len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("Polygon must have at least three finite points.")
    if ((points[:, 0] < 0) | (points[:, 0] > width) |
            (points[:, 1] < 0) | (points[:, 1] > height)).any():
        raise ValueError("Pixel polygon is outside the original image.")
    geometry = Polygon([(xmin + x / width * (xmax - xmin),
                         ymax - y / height * (ymax - ymin)) for x, y in points])
    return make_valid(geometry)


def union_mask(detections: list[Detection], width: int, height: int) -> np.ndarray:
    mask = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for detection in detections:
        if len(detection.polygon) >= 3:
            draw.polygon(detection.polygon, fill=1)
    return np.asarray(mask, dtype=bool)
