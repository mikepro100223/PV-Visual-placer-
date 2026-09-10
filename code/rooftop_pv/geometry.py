"""Metric roof geometry utilities for conservative PV area estimates.

The module deliberately operates on planar, metric Shapely coordinates.  It
does not georeference lon/lat polygons for the caller: pass a polygon after it
has been projected to a local metric CRS (for example a Swiss LV95 CRS).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

try:  # Shapely 2.x
    from shapely.validation import make_valid
except ImportError:  # pragma: no cover - exercised only on old Shapely
    make_valid = None


def _repair(geometry: BaseGeometry) -> BaseGeometry:
    """Return a valid geometry without changing a valid geometry."""

    if geometry.is_valid:
        return geometry
    if make_valid is not None:
        return make_valid(geometry)
    # ``buffer(0)`` is the least surprising fallback on Shapely 1.x for
    # simple self-intersections.  It can alter pathological inputs, so this
    # is intentionally only used after the validity check.
    return geometry.buffer(0)


def _coerce_geometry(value: object, *, name: str) -> BaseGeometry:
    if not isinstance(value, BaseGeometry):
        raise TypeError(f"{name} must be a Shapely geometry")
    geometry = _repair(value)
    if geometry.is_empty:
        raise ValueError(f"{name} must not be empty")
    return geometry


@dataclass(frozen=True)
class RoofGeometry:
    """A roof footprint in a planar metric coordinate system.

    Supply either ``polygon`` or ``manual_area_m2``.  Manual areas are useful
    when a survey provides a trusted usable footprint but no polygon; they are
    intentionally not eligible for geometric panel placement or obstacle
    subtraction.  ``exclusions`` are optional roof-plane obstacles (skylights,
    chimneys, HVAC, and so on) and are unioned before subtraction.
    """

    polygon: BaseGeometry | None = None
    exclusions: tuple[BaseGeometry, ...] = ()
    manual_area_m2: float | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if (self.polygon is None) == (self.manual_area_m2 is None):
            raise ValueError("provide exactly one of polygon or manual_area_m2")
        if self.polygon is not None:
            polygon = _coerce_geometry(self.polygon, name="polygon")
            if polygon.area <= 0:
                raise ValueError("polygon must have positive area")
            object.__setattr__(self, "polygon", polygon)
        else:
            area = float(self.manual_area_m2)  # type: ignore[arg-type]
            if not math.isfinite(area) or area <= 0:
                raise ValueError("manual_area_m2 must be a positive finite number")
            object.__setattr__(self, "manual_area_m2", area)

        cleaned: list[BaseGeometry] = []
        for exclusion in self.exclusions:
            geometry = _coerce_geometry(exclusion, name="exclusion")
            if geometry.area > 0:
                cleaned.append(geometry)
        object.__setattr__(self, "exclusions", tuple(cleaned))


@dataclass(frozen=True)
class AreaResult:
    """Area accounting returned by :func:`calculate_usable_area`.

    ``planimetric_area_m2`` is the map footprint area.  ``tilted_area_m2`` is
    the corresponding roof-plane area and is the default basis for estimating
    module count.  ``geometry`` is in the same planar CRS as the input.
    """

    planimetric_area_m2: float
    tilted_area_m2: float
    excluded_area_m2: float
    geometry: BaseGeometry | None
    tilt_deg: float
    roof_area_m2: float

    @property
    def usable_area_m2(self) -> float:
        """Alias for tilted roof-plane area, suitable for module-area math."""

        return self.tilted_area_m2

    def __float__(self) -> float:
        """Allow explicit ``float(result)`` for callers needing one number."""

        return self.usable_area_m2


def _normalise_exclusions(exclusions: object) -> list[BaseGeometry]:
    if exclusions is None:
        return []
    if isinstance(exclusions, BaseGeometry):
        return [_coerce_geometry(exclusions, name="exclusion")]
    try:
        values = list(exclusions)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("exclusions must be a Shapely geometry or iterable") from exc
    return [_coerce_geometry(value, name="exclusion") for value in values]


def calculate_usable_area(
    roof: RoofGeometry | BaseGeometry | float,
    exclusions: BaseGeometry | Iterable[BaseGeometry] | None = None,
    *,
    tilt_deg: float = 0.0,
    setback_m: float = 0.0,
    pixel_size_m: float = 0.0,
) -> AreaResult:
    """Subtract a union of obstacles and conservative edge setbacks.

    Parameters are in metres and degrees.  ``setback_m`` is applied to the
    roof perimeter and around the union of exclusions.  ``pixel_size_m``
    erodes the remaining geometry by half a pixel to avoid crediting an edge
    that is uncertain at the source raster resolution.  Areas are measured in
    the input CRS; callers must provide metric coordinates.

    A numeric ``roof`` is accepted as shorthand for a trusted manual
    planimetric area.  Numeric/manual inputs cannot be combined with geometric
    exclusions, setbacks, pixel erosion, or panel placement.
    """

    tilt = float(tilt_deg)
    if not math.isfinite(tilt) or not 0 <= tilt < 90:
        raise ValueError("tilt_deg must be finite and in [0, 90)")
    setback = float(setback_m)
    pixel = float(pixel_size_m)
    if not math.isfinite(setback) or setback < 0:
        raise ValueError("setback_m must be a finite non-negative number")
    if not math.isfinite(pixel) or pixel < 0:
        raise ValueError("pixel_size_m must be a finite non-negative number")

    if isinstance(roof, RoofGeometry):
        roof_geometry = roof
    elif isinstance(roof, BaseGeometry):
        roof_geometry = RoofGeometry(polygon=roof)
    elif isinstance(roof, (int, float)) and not isinstance(roof, bool):
        roof_geometry = RoofGeometry(manual_area_m2=float(roof))
    else:
        raise TypeError("roof must be RoofGeometry, Shapely geometry, or area in m²")

    all_exclusions = list(roof_geometry.exclusions) + _normalise_exclusions(exclusions)
    if roof_geometry.polygon is None:
        if all_exclusions or setback or pixel:
            raise ValueError(
                "manual_area_m2 cannot be combined with exclusions, setback_m, or pixel_size_m"
            )
        plan_area = float(roof_geometry.manual_area_m2)
        tilted = plan_area / math.cos(math.radians(tilt))
        return AreaResult(plan_area, tilted, 0.0, None, tilt, plan_area)

    original = roof_geometry.polygon
    roof_area = float(original.area)
    # Erode by the requested edge safety margin and half a source pixel.  A
    # negative buffer naturally handles concave footprints conservatively.
    erosion = setback + pixel / 2.0
    usable = original.buffer(-erosion) if erosion else original
    usable = _repair(usable)
    if all_exclusions and not usable.is_empty:
        obstacle_union = _repair(unary_union(all_exclusions))
        obstacle_buffer = obstacle_union.buffer(setback + pixel / 2.0)
        usable = _repair(usable.difference(obstacle_buffer))
    plan_area = max(float(usable.area), 0.0)
    tilted = plan_area / math.cos(math.radians(tilt))
    return AreaResult(
        planimetric_area_m2=plan_area,
        tilted_area_m2=tilted,
        excluded_area_m2=max(roof_area - plan_area, 0.0),
        geometry=None if usable.is_empty else usable,
        tilt_deg=tilt,
        roof_area_m2=roof_area,
    )


def place_panels(
    area: AreaResult,
    module_width_m: float,
    module_height_m: float,
    *,
    gap_m: float = 0.0,
) -> list[Polygon]:
    """Place axis-aligned module rectangles fully covered by usable geometry.

    This is intentionally a conservative feasibility aid, not an optimizer:
    it does not infer roof orientation, rotate modules, model row shading, or
    turn a pixel-derived footprint into a construction plan.  Every returned
    rectangle is covered by the usable polygon, including holes.
    """

    if area.geometry is None:
        raise ValueError("panel placement requires a geometric roof footprint")
    width = float(module_width_m)
    height = float(module_height_m)
    gap = float(gap_m)
    if not math.isfinite(width) or width <= 0 or not math.isfinite(height) or height <= 0:
        raise ValueError("module dimensions must be positive finite numbers")
    if not math.isfinite(gap) or gap < 0:
        raise ValueError("gap_m must be a finite non-negative number")

    min_x, min_y, max_x, max_y = area.geometry.bounds
    panels: list[Polygon] = []
    x = min_x
    # A tiny tolerance prevents floating-point accumulation from dropping a
    # final rectangle exactly on the boundary.
    tol = 1e-9
    while x + width <= max_x + tol:
        y = min_y
        while y + height <= max_y + tol:
            candidate = box(x, y, x + width, y + height)
            if area.geometry.covers(candidate):
                panels.append(candidate)
            y += height + gap
        x += width + gap
    return panels


__all__ = ["AreaResult", "RoofGeometry", "calculate_usable_area", "place_panels"]
