from __future__ import annotations

import math

import pytest
from shapely.geometry import Polygon, box

from rooftop_pv.geometry import (
    RoofGeometry,
    calculate_usable_area,
    place_panels,
)


def test_overlapping_exclusions_are_unioned_before_area_calculation():
    roof = box(0, 0, 10, 10)
    exclusions = [box(2, 2, 6, 6), box(4, 4, 8, 8)]

    result = calculate_usable_area(roof, exclusions=exclusions)

    assert result.planimetric_area_m2 == pytest.approx(72.0)
    assert result.tilted_area_m2 == pytest.approx(72.0)
    assert result.excluded_area_m2 == pytest.approx(28.0)


def test_setback_and_pixel_size_are_conservative_and_tilt_is_reported():
    result = calculate_usable_area(
        RoofGeometry(box(0, 0, 10, 10)),
        tilt_deg=60,
        setback_m=1,
        pixel_size_m=1,
    )

    # 10 m square eroded by 1 m boundary setback and half a pixel on each
    # side: 7 m x 7 m planimetric footprint.
    assert result.planimetric_area_m2 == pytest.approx(49.0)
    assert result.tilted_area_m2 == pytest.approx(98.0)
    assert result.usable_area_m2 == pytest.approx(result.tilted_area_m2)


def test_manual_area_is_supported_but_has_no_geometry_for_panel_placement():
    result = calculate_usable_area(RoofGeometry(manual_area_m2=123.4), tilt_deg=30)

    assert result.planimetric_area_m2 == pytest.approx(123.4)
    assert result.tilted_area_m2 == pytest.approx(123.4 / math.cos(math.radians(30)))
    assert result.geometry is None


def test_panel_placement_only_returns_rectangles_covered_by_usable_polygon():
    roof = Polygon([(0, 0), (5, 0), (5, 4), (3, 4), (3, 2), (0, 2)])
    result = calculate_usable_area(roof)
    panels = place_panels(result, module_width_m=1.0, module_height_m=1.0)

    assert panels
    assert all(result.geometry.covers(panel) for panel in panels)
    assert all(panel.area == pytest.approx(1.0) for panel in panels)


def test_invalid_tilt_and_non_metric_manual_area_are_rejected():
    with pytest.raises(ValueError, match="tilt"):
        calculate_usable_area(box(0, 0, 2, 2), tilt_deg=90)

    with pytest.raises(ValueError, match="manual_area"):
        RoofGeometry(manual_area_m2=0)
