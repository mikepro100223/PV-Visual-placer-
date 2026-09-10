from __future__ import annotations

import json
import math

import pytest
from shapely.geometry import Polygon, box, shape

from rooftop_pv.inference import Detection
from rooftop_pv.roof_batch import analyze_roofs
from rooftop_pv.sources import RoofFeature


def _roof(
    feature_id: int | str,
    geometry,
    *,
    tilt: float = 30.0,
    azimuth: float = 0.0,
    area: float | None = None,
) -> RoofFeature:
    return RoofFeature(
        feature_id=feature_id,
        geometry={"type": geometry.geom_type, "coordinates": _coordinates(geometry)},
        properties={
            "flaeche": geometry.area if area is None else area,
            "neigung": tilt,
            "ausrichtung": azimuth,
        },
        source_url="test://sonnendach",
        source_data_date="2024-01-01",
        source_fetched_at="2024-01-01T00:00:00+00:00",
    )


def _coordinates(geometry):
    if geometry.geom_type == "Polygon":
        rings = [list(geometry.exterior.coords)]
        rings.extend(list(interior.coords) for interior in geometry.interiors)
        return rings
    if geometry.geom_type == "MultiPolygon":
        return [[list(poly.exterior.coords), *[list(ring.coords) for ring in poly.interiors]] for poly in geometry.geoms]
    raise ValueError(f"unsupported test geometry: {geometry.geom_type}")


def _pixels_for_map(points, bbox=(0.0, 0.0, 22.0, 10.0), image_size=(220, 100)):
    min_x, min_y, max_x, max_y = bbox
    width, height = image_size
    return tuple(
        ((x - min_x) / (max_x - min_x) * width, (max_y - y) / (max_y - min_y) * height)
        for x, y in points
    )


def _detection(map_polygon, *, class_name="solar_panel", bbox=(0.0, 0.0, 22.0, 10.0), image_size=(220, 100)):
    return Detection(class_name, 0.9, _pixels_for_map(list(map_polygon.exterior.coords), bbox, image_size))


def test_two_roofs_are_individually_clipped_and_source_unions_do_not_double_count():
    roofs = [_roof(101, box(0, 0, 10, 10), azimuth=-90, area=120.0), _roof(202, box(12, 0, 22, 10), tilt=20)]
    pv = [_detection(box(2, 2, 5, 5)), _detection(box(2, 2, 5, 5))]
    obstacles = [_detection(box(4, 4, 7, 7), class_name="chimney")]
    external = [box(6, 6, 8, 8), box(7, 7, 9, 9)]

    result = analyze_roofs(
        roofs,
        bbox2056=(0, 0, 22, 10),
        image_size=(220, 100),
        pv_detections=pv,
        obstacle_detections=obstacles,
        external_exclusions=external,
        setback_m=0,
        pixel_size_m=0,
        fill_ratio=0.5,
    )

    rows = result["rows"]
    assert list(rows) == ["101", "202"]
    first = rows["101"]
    assert first["official_area_m2"] == pytest.approx(120.0)
    assert first["tilt_deg"] == pytest.approx(30.0)
    assert first["azimuth_deg"] == pytest.approx(90.0)
    assert first["image_coverage_fraction"] == pytest.approx(1.0)
    assert first["full_coverage"] is True
    assert first["pv_area_planimetric_m2"] == pytest.approx(9.0)
    assert first["ai_obstacle_area_planimetric_m2"] == pytest.approx(9.0)
    assert first["external_exclusion_area_planimetric_m2"] == pytest.approx(7.0)
    assert first["combined_exclusion_area_planimetric_m2"] < 25.0
    assert first["usable_horizontal_area_m2"] == pytest.approx(
        100.0 - first["combined_exclusion_area_planimetric_m2"]
    )
    assert first["usable_slope_area_m2"] == pytest.approx(
        first["usable_horizontal_area_m2"] / math.cos(math.radians(30.0))
    )
    assert first["candidate_module_area_m2"] == pytest.approx(first["usable_slope_area_m2"] * 0.5)
    assert rows["202"]["pv_area_planimetric_m2"] == pytest.approx(0.0)
    assert rows["202"]["usable_horizontal_area_m2"] == pytest.approx(100.0)
    json.dumps(result, allow_nan=False)


def test_overlapping_roofs_report_a_caveat_instead_of_claiming_summed_area():
    roofs = [_roof("left", box(0, 0, 10, 10)), _roof("right", box(5, 0, 15, 10))]

    result = analyze_roofs(
        roofs,
        bbox2056=(0, 0, 15, 10),
        image_size=(150, 100),
        pv_detections=[],
        obstacle_detections=[],
        setback_m=0,
        pixel_size_m=0,
    )

    assert len(result["overlaps"]) == 1
    assert result["overlaps"][0]["roof_ids"] == ["left", "right"]
    assert result["overlaps"][0]["intersection_area_m2"] == pytest.approx(50.0)
    assert "must not be summed" in result["summed_area_caveat"]


def test_coordinate_transform_and_source_clipping_are_in_lv95():
    roof = _roof("roof", box(10, 20, 20, 30), area=111.0)
    detection = _detection(
        box(0, 20, 30, 50),
        bbox=(0, 10, 40, 50),
        image_size=(400, 400),
    )

    result = analyze_roofs(
        [roof],
        bbox2056=(0, 10, 40, 50),
        image_size=(400, 400),
        pv_detections=[detection],
        obstacle_detections=[],
        setback_m=0,
        pixel_size_m=0,
    )
    row = result["rows"]["roof"]

    assert row["pv_area_planimetric_m2"] == pytest.approx(100.0)
    assert row["roof_geometry_lv95"]["type"] == "Polygon"
    assert shape(row["roof_geometry_lv95"]).bounds == pytest.approx((10, 20, 20, 30))
    assert result["image"]["crs"] == "EPSG:2056"


def test_partial_image_reports_observed_area_but_never_whole_roof_usable_area():
    roof = _roof("partial", box(0, 0, 10, 10))

    result = analyze_roofs(
        [roof],
        bbox2056=(0, 0, 5, 10),
        image_size=(50, 100),
        pv_detections=[],
        obstacle_detections=[],
        setback_m=0,
        pixel_size_m=0,
    )
    row = result["rows"]["partial"]

    assert row["image_partial"] is True
    assert row["full_coverage"] is False
    assert row["image_coverage_fraction"] == pytest.approx(0.5)
    assert row["observed_area_planimetric_m2"] == pytest.approx(50.0)
    assert row["observed_usable_horizontal_area_m2"] == pytest.approx(50.0)
    assert row["usable_horizontal_area_m2"] is None
    assert row["usable_slope_area_m2"] is None
    assert row["candidate_module_area_m2"] is None
    assert row["usable_geometry_lv95"] is None
    assert row["observed_usable_geometry_lv95"] is not None


def test_setback_and_pixel_erosion_losses_are_explicit():
    result = analyze_roofs(
        [_roof("losses", box(0, 0, 10, 10), tilt=0)],
        bbox2056=(0, 0, 10, 10),
        image_size=(100, 100),
        pv_detections=[],
        obstacle_detections=[],
        setback_m=1,
        pixel_size_m=2,
    )
    row = result["rows"]["losses"]

    assert row["setback_loss_planimetric_m2"] == pytest.approx(36.0)
    assert row["pixel_erosion_loss_planimetric_m2"] == pytest.approx(28.0)
    assert row["setback_erosion_loss_planimetric_m2"] == pytest.approx(64.0)
    assert row["setback_erosion_loss_slope_m2"] == pytest.approx(64.0)


def test_holes_are_preserved_and_default_pixel_size_is_explicit():
    roof_geometry = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
        holes=[[(4, 4), (6, 4), (6, 6), (4, 6), (4, 4)]],
    )
    result = analyze_roofs(
        [_roof("hole", roof_geometry)],
        bbox2056=(0, 0, 10, 10),
        image_size=(10, 10),
        pv_detections=[],
        obstacle_detections=[],
        setback_m=0,
        pixel_size_m=None,
    )
    row = result["rows"]["hole"]

    assert result["image"]["pixel_size_m"] == pytest.approx(1.0)
    assert len(row["roof_geometry_lv95"]["coordinates"]) == 2
    assert 72.0 < row["usable_horizontal_area_m2"] < 72.5


def test_repaired_polygon_keeps_positive_components_and_records_caveat():
    invalid_bowtie = Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)])
    result = analyze_roofs(
        [_roof("repaired", invalid_bowtie, area=2.0)],
        bbox2056=(0, 0, 2, 2),
        image_size=(20, 20),
        pv_detections=[],
        obstacle_detections=[],
        setback_m=0,
        pixel_size_m=0,
    )
    row = result["rows"]["repaired"]

    assert not row["errors"]
    assert row["geometry_caveats"]
    assert row["roof_area_planimetric_m2"] == pytest.approx(2.0)

    touching_hole = Polygon(
        [(4, 0), (8, 0), (8, 4), (4, 4), (4, 0)],
        holes=[[(4, 1), (6, 1), (6, 2), (4, 2), (4, 1)]],
    )
    collection_result = analyze_roofs(
        [_roof("collection", touching_hole, area=16.0)],
        bbox2056=(4, 0, 8, 4),
        image_size=(40, 40),
        pv_detections=[],
        obstacle_detections=[],
        setback_m=0,
        pixel_size_m=0,
    )
    collection_row = collection_result["rows"]["collection"]
    assert not collection_row["errors"]
    assert "degenerate/non-polygon remnants" in collection_row["geometry_caveats"][0]


def test_invalid_geometry_or_tilt_keeps_one_explicit_error_row_per_input():
    invalid_tilt = _roof("bad-tilt", box(0, 0, 2, 2), tilt=None)  # type: ignore[arg-type]
    vertical_tilt = _roof("vertical", box(3, 0, 5, 2), tilt=90)
    zero_area = _roof("zero-area", Polygon([(6, 0), (7, 0), (8, 0), (6, 0)]))
    invalid_geometry = RoofFeature(
        feature_id="bad-geometry",
        geometry={"type": "Point", "coordinates": [1, 1]},
        properties={"flaeche": 4.0, "neigung": 30.0, "ausrichtung": 0.0},
        source_url="test://sonnendach",
        source_data_date=None,
        source_fetched_at="2024-01-01T00:00:00+00:00",
    )

    result = analyze_roofs(
        [invalid_tilt, vertical_tilt, zero_area, invalid_geometry],
        bbox2056=(0, 0, 8, 2),
        image_size=(20, 20),
        pv_detections=[],
        obstacle_detections=[],
        setback_m=0,
        pixel_size_m=0,
    )

    assert set(result["rows"]) == {"bad-tilt", "vertical", "zero-area", "bad-geometry"}
    assert result["rows"]["bad-tilt"]["errors"]
    assert result["rows"]["vertical"]["errors"]
    assert result["rows"]["zero-area"]["errors"]
    assert result["rows"]["bad-geometry"]["errors"]
    assert result["rows"]["bad-geometry"]["official_area_m2"] == pytest.approx(4.0)
    assert result["rows"]["bad-geometry"]["tilt_deg"] == pytest.approx(30.0)
    assert result["rows"]["bad-geometry"]["usable_horizontal_area_m2"] is None


def test_one_roof_calculation_error_is_kept_without_dropping_other_rows(monkeypatch):
    import rooftop_pv.roof_batch as roof_batch

    original = roof_batch.calculate_usable_area

    def fail_for_left(roof, *args, **kwargs):
        if roof.polygon.bounds[0] == 0:
            raise ValueError("synthetic calculation failure")
        return original(roof, *args, **kwargs)

    monkeypatch.setattr(roof_batch, "calculate_usable_area", fail_for_left)
    result = analyze_roofs(
        [_roof("left", box(0, 0, 2, 2)), _roof("right", box(3, 0, 5, 2))],
        bbox2056=(0, 0, 5, 2),
        image_size=(50, 20),
        pv_detections=[],
        obstacle_detections=[],
        setback_m=0,
        pixel_size_m=0,
    )

    assert len(result["rows"]) == 2
    assert any("synthetic calculation failure" in error for error in result["rows"]["left"]["errors"])
    assert result["rows"]["right"]["usable_horizontal_area_m2"] == pytest.approx(4.0)


def test_duplicate_ids_and_unsupported_parameters_are_rejected():
    roof = _roof(1, box(0, 0, 1, 1))
    with pytest.raises(ValueError, match="duplicate"):
        analyze_roofs(
            [roof, _roof("1", box(2, 0, 3, 1))],
            bbox2056=(0, 0, 3, 1),
            image_size=(30, 10),
            pv_detections=[],
            obstacle_detections=[],
        )

    common = dict(
        roofs=[roof],
        bbox2056=(0, 0, 1, 1),
        image_size=(10, 10),
        pv_detections=[],
        obstacle_detections=[],
    )
    with pytest.raises(ValueError, match="bbox"):
        analyze_roofs(**{**common, "bbox2056": (0, 0, 0, 1)})
    with pytest.raises(ValueError, match="span"):
        analyze_roofs(**{**common, "bbox2056": (0, 0, 5_001, 1)})
    with pytest.raises(ValueError, match="image_size"):
        analyze_roofs(**{**common, "image_size": (0, 0)})
    with pytest.raises(ValueError, match="setback"):
        analyze_roofs(**{**common, "setback_m": -1})
    with pytest.raises(ValueError, match="pixel_size"):
        analyze_roofs(**{**common, "pixel_size_m": -1})
    with pytest.raises(ValueError, match="fill_ratio"):
        analyze_roofs(**{**common, "fill_ratio": 1.1})
    with pytest.raises(ValueError, match="non-empty"):
        analyze_roofs(
            [_roof(None, box(0, 0, 1, 1))],
            bbox2056=(0, 0, 1, 1),
            image_size=(10, 10),
            pv_detections=[],
            obstacle_detections=[],
        )
