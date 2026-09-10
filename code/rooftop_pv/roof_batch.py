"""Per-roof geometry accounting for one georeferenced orthophoto.

The batch function in this module deliberately has no weather or Streamlit
dependencies.  It maps segmentation contours once into LV95, clips every
source union to each Sonnendach roof, and keeps incomplete image coverage
separate from whole-roof claims.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from shapely.geometry import box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from rooftop_pv.geometry import RoofGeometry, calculate_usable_area
from rooftop_pv.inference import pixel_to_map
from rooftop_pv.sources import RoofFeature, SourceError


_MAX_BBOX_SPAN_M = 5_000.0
_COVERAGE_TOLERANCE = 1e-9


def _repair(geometry: BaseGeometry) -> BaseGeometry:
    if geometry.is_valid:
        return geometry
    try:
        from shapely.validation import make_valid

        return make_valid(geometry)
    except ImportError:  # pragma: no cover - Shapely 2 is the supported runtime
        return geometry.buffer(0)


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_bbox(bbox2056: Sequence[float]) -> tuple[float, float, float, float]:
    if isinstance(bbox2056, (str, bytes)) or not isinstance(bbox2056, Sequence) or len(bbox2056) != 4:
        raise ValueError("bbox2056 must be a four-value LV95 sequence")
    values = tuple(_finite_float(value, "bbox coordinate") for value in bbox2056)
    min_x, min_y, max_x, max_y = values
    if min_x >= max_x or min_y >= max_y:
        raise ValueError("bbox2056 must have min values smaller than max values")
    if max_x - min_x > _MAX_BBOX_SPAN_M or max_y - min_y > _MAX_BBOX_SPAN_M:
        raise ValueError(f"bbox span exceeds {_MAX_BBOX_SPAN_M:g} m")
    return values


def _validate_image_size(image_size: Sequence[int]) -> tuple[int, int]:
    if isinstance(image_size, (str, bytes)) or not isinstance(image_size, Sequence) or len(image_size) != 2:
        raise ValueError("image_size must be a (width, height) pair")
    values: list[int] = []
    for value in image_size:
        if isinstance(value, bool):
            raise ValueError("image_size dimensions must be positive integers")
        try:
            integer = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("image_size dimensions must be positive integers") from exc
        if integer <= 0 or integer != value:
            raise ValueError("image_size dimensions must be positive integers")
        values.append(integer)
    return values[0], values[1]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _geometry_json(geometry: BaseGeometry | None) -> dict[str, Any] | None:
    if geometry is None or geometry.is_empty:
        return None
    return _json_safe(mapping(geometry))


def _as_geometry(value: Any, *, name: str) -> BaseGeometry:
    try:
        geometry = value if isinstance(value, BaseGeometry) else shape(value)
    except Exception as exc:
        raise ValueError(f"{name} is not valid GeoJSON geometry") from exc
    return geometry


def _polygon_parts(geometry: BaseGeometry) -> list[BaseGeometry]:
    if geometry.geom_type == "Polygon":
        return [geometry] if geometry.area > 0 else []
    if geometry.geom_type == "MultiPolygon":
        return [part for part in geometry.geoms if part.area > 0]
    if geometry.geom_type == "GeometryCollection":
        parts: list[BaseGeometry] = []
        for child in geometry.geoms:
            parts.extend(_polygon_parts(child))
        return parts
    return []


def _normalise_polygon(value: Any, *, name: str) -> tuple[BaseGeometry, str | None]:
    geometry = _as_geometry(value, name=name)
    was_valid = geometry.is_valid
    geometry = _repair(geometry)
    caveat: str | None = None
    if geometry.geom_type == "GeometryCollection":
        parts = _polygon_parts(geometry)
        if parts:
            geometry = _repair(unary_union(parts))
            caveat = (
                f"{name} was repaired; only positive-area polygon components were kept and "
                "degenerate/non-polygon remnants were discarded"
            )
        else:
            raise ValueError(f"{name} has no positive-area polygon component after repair")
    if not was_valid and caveat is None:
        caveat = f"{name} was repaired with make_valid"
    if geometry.is_empty or geometry.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError(f"{name} must be a non-empty Polygon or MultiPolygon")
    if not math.isfinite(float(geometry.area)) or geometry.area <= 0:
        raise ValueError(f"{name} must have positive finite area")
    return geometry, caveat


def _coerce_polygon(value: Any, *, name: str) -> BaseGeometry:
    geometry, _ = _normalise_polygon(value, name=name)
    return geometry


def _coerce_external_exclusions(exclusions: Any) -> list[BaseGeometry]:
    if exclusions is None:
        return []
    if isinstance(exclusions, BaseGeometry) or isinstance(exclusions, Mapping):
        exclusions = [exclusions]
    elif isinstance(exclusions, (str, bytes)):
        raise TypeError("external_exclusions must contain geometries")
    try:
        values = list(exclusions)
    except TypeError as exc:
        raise TypeError("external_exclusions must be an iterable of geometries") from exc
    return [_coerce_polygon(value, name="external exclusion") for value in values]


def _detection_geometry(detection: Any, *, bbox: tuple[float, float, float, float], image_size: tuple[int, int]) -> BaseGeometry:
    polygon = getattr(detection, "polygon", None)
    if polygon is None and isinstance(detection, Mapping):
        polygon = detection.get("polygon", detection.get("polygon_pixels"))
    if polygon is None:
        raise TypeError("detections must expose a polygon in image pixels")
    try:
        geometry = pixel_to_map(polygon, bbox, image_size[0], image_size[1])
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"invalid detection polygon: {exc}") from exc
    geometry = _repair(geometry)
    if geometry.is_empty or geometry.area <= 0:
        raise ValueError("detection polygon must have positive area")
    return geometry


def _map_detections(detections: Any, *, bbox: tuple[float, float, float, float], image_size: tuple[int, int]) -> list[BaseGeometry]:
    if detections is None or isinstance(detections, (str, bytes)):
        raise TypeError("detections must be an iterable")
    try:
        values = list(detections)
    except TypeError as exc:
        raise TypeError("detections must be an iterable") from exc
    return [_detection_geometry(value, bbox=bbox, image_size=image_size) for value in values]


def _union_clipped(geometries: Sequence[BaseGeometry], clip_region: BaseGeometry) -> BaseGeometry | None:
    clipped: list[BaseGeometry] = []
    for geometry in geometries:
        candidate = _repair(geometry.intersection(clip_region))
        if not candidate.is_empty and candidate.area > 0:
            clipped.append(candidate)
    if not clipped:
        return None
    return _repair(unary_union(clipped))


def _empty_row(feature_id: Any) -> dict[str, Any]:
    return {
        "feature_id": _json_safe(feature_id),
        "official_area_m2": None,
        "tilt_deg": None,
        "azimuth_deg": None,
        "roof_geometry_lv95": None,
        "geometry_caveats": [],
        "observed_geometry_lv95": None,
        "usable_geometry_lv95": None,
        "observed_usable_geometry_lv95": None,
        "image_coverage_fraction": None,
        "image_partial": None,
        "full_coverage": None,
        "roof_area_planimetric_m2": None,
        "roof_area_slope_m2": None,
        "setback_loss_planimetric_m2": None,
        "setback_loss_slope_m2": None,
        "pixel_erosion_loss_planimetric_m2": None,
        "pixel_erosion_loss_slope_m2": None,
        "setback_erosion_loss_planimetric_m2": None,
        "setback_erosion_loss_slope_m2": None,
        "observed_area_planimetric_m2": None,
        "observed_area_slope_m2": None,
        "pv_area_planimetric_m2": None,
        "ai_obstacle_area_planimetric_m2": None,
        "external_exclusion_area_planimetric_m2": None,
        "combined_exclusion_area_planimetric_m2": None,
        "observed_usable_horizontal_area_m2": None,
        "observed_usable_slope_area_m2": None,
        "usable_horizontal_area_m2": None,
        "usable_slope_area_m2": None,
        "candidate_module_area_m2": None,
        "candidate_module_area_fill_ratio": None,
        "errors": [],
    }


def analyze_roofs(
    roofs: Sequence[RoofFeature],
    *,
    bbox2056: Sequence[float],
    image_size: tuple[int, int],
    pv_detections: Sequence[Any],
    obstacle_detections: Sequence[Any],
    external_exclusions: Sequence[Any] = (),
    setback_m: float = 0.3,
    pixel_size_m: float | None = None,
    fill_ratio: float = 0.85,
) -> dict[str, Any]:
    """Analyze every Sonnendach roof against one LV95 orthophoto.

    Detection polygons are interpreted in original-image pixel coordinates and
    mapped with :func:`pixel_to_map`.  ``usable_*`` fields are only populated
    when the image covers the complete roof.  For a partial image, the
    ``observed_*`` fields describe only the roof/image intersection and whole-
    roof fields remain ``None`` to avoid turning missing imagery into claimed
    usable area.
    """

    if isinstance(roofs, (str, bytes)):
        raise TypeError("roofs must be a sequence of RoofFeature objects")
    try:
        roof_values = list(roofs)
    except TypeError as exc:
        raise TypeError("roofs must be a sequence of RoofFeature objects") from exc
    bbox = _validate_bbox(bbox2056)
    dimensions = _validate_image_size(image_size)
    setback = _finite_float(setback_m, "setback_m")
    if setback < 0:
        raise ValueError("setback_m must be non-negative")
    if setback > _MAX_BBOX_SPAN_M:
        raise ValueError(f"setback_m exceeds {_MAX_BBOX_SPAN_M:g} m")
    if pixel_size_m is None:
        pixel_size = max((bbox[2] - bbox[0]) / dimensions[0], (bbox[3] - bbox[1]) / dimensions[1])
    else:
        pixel_size = _finite_float(pixel_size_m, "pixel_size_m")
        if pixel_size < 0:
            raise ValueError("pixel_size_m must be non-negative")
    fill = _finite_float(fill_ratio, "fill_ratio")
    if not 0 <= fill <= 1:
        raise ValueError("fill_ratio must be in [0, 1]")

    ids: list[str] = []
    for roof in roof_values:
        feature_id = getattr(roof, "feature_id", None)
        key = str(feature_id)
        if feature_id is None or not key.strip():
            raise ValueError("roof feature IDs must be non-empty")
        if key in ids:
            raise ValueError(f"duplicate Sonnendach feature ID: {key}")
        ids.append(key)

    image_geometry = box(bbox[0], bbox[1], bbox[2], bbox[3])
    pv_geometries = _map_detections(pv_detections, bbox=bbox, image_size=dimensions)
    ai_geometries = _map_detections(obstacle_detections, bbox=bbox, image_size=dimensions)
    external_geometries = _coerce_external_exclusions(external_exclusions)

    rows: dict[str, dict[str, Any]] = {}
    valid_roofs: list[tuple[str, BaseGeometry]] = []
    for roof, feature_id in zip(roof_values, ids):
        row = _empty_row(getattr(roof, "feature_id", feature_id))
        raw_geometry = getattr(roof, "geometry", None)
        if isinstance(raw_geometry, Mapping):
            row["roof_geometry_lv95"] = _json_safe(raw_geometry)
        errors: list[str] = row["errors"]
        try:
            tilt = _finite_float(roof.tilt_deg, "roof tilt")
            if not 0 <= tilt < 90:
                raise ValueError("roof tilt must be in [0, 90) degrees")
            row["tilt_deg"] = tilt
        except (AttributeError, TypeError, ValueError, SourceError) as exc:
            errors.append(str(exc))
        try:
            azimuth = _finite_float(roof.azimuth_deg, "roof azimuth") % 360.0
            row["azimuth_deg"] = azimuth
        except (AttributeError, TypeError, ValueError, SourceError) as exc:
            errors.append(str(exc))
        try:
            official_area = _finite_float(roof.area_m2, "official roof area")
            if official_area <= 0:
                raise ValueError("official roof area must be positive")
            row["official_area_m2"] = official_area
        except (AttributeError, TypeError, ValueError, SourceError) as exc:
            errors.append(str(exc))
        try:
            geometry, geometry_caveat = _normalise_polygon(raw_geometry, name="roof geometry")
            row["roof_geometry_lv95"] = _geometry_json(geometry)
            if geometry_caveat is not None:
                row["geometry_caveats"].append(geometry_caveat)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
        if errors:
            rows[feature_id] = row
            continue

        assert row["tilt_deg"] is not None
        tilt = float(row["tilt_deg"])
        valid_roofs.append((feature_id, geometry))
        try:
            observed = _repair(geometry.intersection(image_geometry))
            observed_area = float(observed.area) if not observed.is_empty else 0.0
            coverage = max(0.0, min(1.0, observed_area / float(geometry.area)))
            partial = coverage < 1.0 - _COVERAGE_TOLERANCE
            roof_area_slope = float(geometry.area) / math.cos(math.radians(tilt))
            row.update(
                {
                    "image_coverage_fraction": coverage,
                    "image_partial": partial,
                    "full_coverage": not partial,
                    "roof_area_planimetric_m2": float(geometry.area),
                    "roof_area_slope_m2": roof_area_slope,
                    "observed_geometry_lv95": _geometry_json(observed),
                    "observed_area_planimetric_m2": observed_area,
                    "observed_area_slope_m2": observed_area / math.cos(math.radians(tilt)),
                }
            )

            roof_without_margin = calculate_usable_area(
                RoofGeometry(polygon=geometry), tilt_deg=tilt, setback_m=0, pixel_size_m=0
            )
            roof_with_setback = calculate_usable_area(
                RoofGeometry(polygon=geometry), tilt_deg=tilt, setback_m=setback, pixel_size_m=0
            )
            roof_with_erosion = calculate_usable_area(
                RoofGeometry(polygon=geometry), tilt_deg=tilt, setback_m=setback, pixel_size_m=pixel_size
            )
            setback_loss_plan = max(
                roof_without_margin.planimetric_area_m2 - roof_with_setback.planimetric_area_m2, 0.0
            )
            pixel_loss_plan = max(
                roof_with_setback.planimetric_area_m2 - roof_with_erosion.planimetric_area_m2, 0.0
            )
            row.update(
                {
                    "setback_loss_planimetric_m2": setback_loss_plan,
                    "setback_loss_slope_m2": setback_loss_plan / math.cos(math.radians(tilt)),
                    "pixel_erosion_loss_planimetric_m2": pixel_loss_plan,
                    "pixel_erosion_loss_slope_m2": pixel_loss_plan / math.cos(math.radians(tilt)),
                    "setback_erosion_loss_planimetric_m2": setback_loss_plan + pixel_loss_plan,
                    "setback_erosion_loss_slope_m2": (setback_loss_plan + pixel_loss_plan)
                    / math.cos(math.radians(tilt)),
                }
            )

            clip_region = observed
            pv_union = _union_clipped(pv_geometries, clip_region)
            ai_union = _union_clipped(ai_geometries, clip_region)
            external_union = _union_clipped(external_geometries, clip_region)
            source_geometries = [geometry for geometry in (pv_union, ai_union, external_union) if geometry is not None]
            combined_union = _repair(unary_union(source_geometries)) if source_geometries else None
            area_result = calculate_usable_area(
                RoofGeometry(polygon=geometry),
                exclusions=combined_union,
                tilt_deg=tilt,
                setback_m=setback,
                pixel_size_m=pixel_size,
            )
            usable_full = area_result.geometry
            observed_usable = _repair(usable_full.intersection(image_geometry)) if usable_full is not None else None
            observed_usable_plan = float(observed_usable.area) if observed_usable is not None and not observed_usable.is_empty else 0.0
            observed_usable_slope = observed_usable_plan / math.cos(math.radians(tilt))
            row.update(
                {
                    "pv_area_planimetric_m2": float(pv_union.area) if pv_union is not None else 0.0,
                    "ai_obstacle_area_planimetric_m2": float(ai_union.area) if ai_union is not None else 0.0,
                    "external_exclusion_area_planimetric_m2": float(external_union.area) if external_union is not None else 0.0,
                    "combined_exclusion_area_planimetric_m2": float(combined_union.area) if combined_union is not None else 0.0,
                    "observed_usable_geometry_lv95": _geometry_json(observed_usable),
                    "observed_usable_horizontal_area_m2": observed_usable_plan,
                    "observed_usable_slope_area_m2": observed_usable_slope,
                }
            )
            if not partial:
                row.update(
                    {
                        "usable_geometry_lv95": _geometry_json(usable_full),
                        "usable_horizontal_area_m2": float(area_result.planimetric_area_m2),
                        "usable_slope_area_m2": float(area_result.tilted_area_m2),
                        "candidate_module_area_m2": float(area_result.tilted_area_m2) * fill,
                        "candidate_module_area_fill_ratio": fill,
                    }
                )
        except Exception as exc:
            errors.append(f"calculation: {exc}")
        rows[feature_id] = row

    overlaps: list[dict[str, Any]] = []
    for index, (first_id, first_geometry) in enumerate(valid_roofs):
        for second_id, second_geometry in valid_roofs[index + 1 :]:
            intersection = _repair(first_geometry.intersection(second_geometry))
            if not intersection.is_empty and intersection.area > 0:
                overlaps.append(
                    {
                        "roof_ids": [first_id, second_id],
                        "intersection_area_m2": float(intersection.area),
                    }
                )

    return _json_safe(
        {
            "image": {
                "crs": "EPSG:2056",
                "bbox2056": list(bbox),
                "image_size": list(dimensions),
                "pixel_size_m": pixel_size,
                "setback_m": setback,
                "fill_ratio": fill,
            },
            "roof_count": len(rows),
            "rows": rows,
            "overlaps": overlaps,
            "summed_area_caveat": (
                "Roof rows are independent Sonnendach features. Overlapping roof polygons are reported, "
                "but their areas and candidate module budgets must not be summed."
            ),
        }
    )


__all__ = ["analyze_roofs"]
