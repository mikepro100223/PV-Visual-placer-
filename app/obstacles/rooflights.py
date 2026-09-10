"""Roof-window detection from the aerial image.

The height model finds anything standing proud of the roof, but a roof window
sits flush in the pitch and is invisible to it. In an aerial photo it is not:
glass reflects the sky, so a rooflight reads distinctly blue against clay,
concrete or bitumen, all of which are red- or brown-dominant.

Measured on a Zurich roof: the tiles sit at about -18 on the blue-minus-red
axis and the windows at +20 to +32, which is the top one percent of the roof.
So the test is a robust one against the roof's own colour, not a fixed cut,
and every candidate must also be the size and shape of an actual window.
"""

import cv2
import numpy as np
from shapely.geometry import Polygon

# A rooflight is bluer than the roof immediately around it by at least this
# much, in 0-255 units, and by this many robust deviations.
MIN_BLUE_SHIFT = 8.0
BLUE_SIGMAS = 3.0
# The surroundings are sampled over this radius, wide enough to span a window
# and its roof but narrow enough to track sun and shade across the building.
BACKGROUND_M = 3.0
# Physical size of a roof window, in square metres.
MIN_AREA_M2 = 0.25
MAX_AREA_M2 = 6.0
# Rectangular-ish: neither a sliver nor a ragged blob.
MIN_ASPECT = 0.3
MAX_ASPECT = 3.4
MIN_EXTENT = 0.55
MAX_ROOFLIGHTS = 60
# Ignore the roof's own edge, where gutters and flashing are also bluish.
EDGE_MARGIN_M = 0.4
# The bright reflection stops short of the frame, so grow the detection back.
GROW_M = 0.2


def _mask_from(polygon, shape) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    if polygon.geom_type == "MultiPolygon":
        for part in polygon.geoms:
            mask |= _mask_from(part, shape)
        return mask
    if polygon.is_empty or polygon.geom_type != "Polygon":
        return mask
    rings = [np.array(polygon.exterior.coords, np.int32)]
    cv2.fillPoly(mask, rings, 1)
    for interior in polygon.interiors:
        cv2.fillPoly(mask, [np.array(interior.coords, np.int32)], 0)
    return mask


def detect(
    image: np.ndarray,
    roof: Polygon,
    pixels_per_metre: float,
    exclude: list[Polygon] | None = None,
) -> list[dict]:
    """Return roof-window polygons in image pixels."""
    if roof.is_empty or pixels_per_metre <= 0:
        return []
    rgb = image.astype(np.float32)
    inside = _mask_from(roof, rgb.shape[:2])
    edge = max(1, int(round(EDGE_MARGIN_M * pixels_per_metre)))
    inside = cv2.erode(inside, np.ones((edge * 2 + 1, edge * 2 + 1), np.uint8))
    for other in exclude or []:
        if not other.is_empty and other.geom_type == "Polygon":
            inside[_mask_from(other, rgb.shape[:2]).astype(bool)] = 0
    core = inside.astype(bool)
    if int(core.sum()) < 64:
        return []

    blue = rgb[..., 2] - rgb[..., 0]
    # Against the local roof, not the whole building: one merged roof spans
    # sunlit and shaded faces, and that spread swamps a global threshold.
    kernel = int(round(BACKGROUND_M * pixels_per_metre)) | 1
    contrast = blue - cv2.blur(blue, (kernel, kernel))
    values = contrast[core]
    centre = float(np.median(values))
    spread = float(np.median(np.abs(values - centre)))
    threshold = centre + max(MIN_BLUE_SHIFT, BLUE_SIGMAS * 1.4826 * spread)
    hit = (core & (contrast > threshold)).astype(np.uint8)
    if not hit.any():
        return []
    hit = cv2.morphologyEx(hit, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    cell = 1.0 / (pixels_per_metre**2)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(hit, 8)
    found = []
    for index in range(1, count):
        x, y, w, h, pixels = stats[index]
        area = pixels * cell
        if area < MIN_AREA_M2 or area > MAX_AREA_M2:
            continue
        aspect = w / h if h else 0.0
        if not MIN_ASPECT <= aspect <= MAX_ASPECT:
            continue
        if pixels / float(w * h) < MIN_EXTENT:
            continue
        piece = (labels[y : y + h, x : x + w] == index).astype(np.uint8)
        contours, _ = cv2.findContours(
            piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        epsilon = max(0.5, 0.03 * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(approx) < 3:
            box = cv2.boxPoints(cv2.minAreaRect(contour))
            approx = np.array(box)
        polygon = Polygon([(float(px + x), float(py + y)) for px, py in approx])
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.geom_type == "Polygon" and not polygon.is_empty:
            polygon = polygon.buffer(GROW_M * pixels_per_metre, join_style=2)
        if polygon.geom_type != "Polygon" or polygon.is_empty:
            continue
        found.append(
            {
                "geometry": polygon,
                "area_m2": round(polygon.area * cell, 2),
                "blue_shift": round(float(contrast[labels == index].mean()), 1),
            }
        )
    found.sort(key=lambda o: -o["area_m2"])
    return found[:MAX_ROOFLIGHTS]
