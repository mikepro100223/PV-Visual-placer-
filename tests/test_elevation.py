"""Superstructure detection against a synthetic roof with a known chimney."""

import numpy as np
import pytest
from shapely.geometry import box

from backend.services import elevation_service as es

MINX, MAXY = 2600000.0, 1200020.0


def roof(pitch_per_cell=0.0, size=40):
    """A roof face as a height grid, optionally sloping across the frame."""
    rows, cols = np.mgrid[0:size, 0:size]
    return (500.0 + pitch_per_cell * cols).astype(np.float32) + 0 * rows


def facet(size=40):
    return box(MINX, MAXY - size * es.DSM_STEP_M, MINX + size * es.DSM_STEP_M, MAXY)


def test_finds_a_chimney_on_a_flat_roof():
    heights = roof()
    heights[18:22, 18:22] += 2.0  # a 2 m x 2 m chimney rising 2 m
    found = es.detect(heights, MINX, MAXY, [facet()])
    assert len(found) == 1
    assert found[0]["height_m"] == pytest.approx(2.0, abs=0.05)
    assert found[0]["area_m2"] == pytest.approx(4.0, abs=1.5)


def test_a_slanted_roof_is_not_itself_an_obstacle():
    # 0.15 m of rise per 0.5 m cell is a ~17 degree pitch: far above the
    # detection threshold, so a naive height cut would flag the whole face.
    heights = roof(pitch_per_cell=0.15)
    assert heights.max() - heights.min() > 5
    assert es.detect(heights, MINX, MAXY, [facet()]) == []


def test_finds_a_chimney_on_a_slanted_roof():
    heights = roof(pitch_per_cell=0.15)
    heights[18:22, 18:22] += 2.0
    found = es.detect(heights, MINX, MAXY, [facet()])
    assert len(found) == 1
    assert found[0]["height_m"] == pytest.approx(2.0, abs=0.1)


def test_classifies_by_footprint():
    heights = roof()
    heights[10:12, 10:12] += 2.5  # 1 m x 1 m chimney
    heights[24:32, 20:28] += 1.2  # 4 m x 4 m dormer-sized block
    kinds = {o["kind"] for o in es.detect(heights, MINX, MAXY, [facet()])}
    assert kinds == {"chimney", "other_obstacle"}


def test_ignores_speckle_and_shallow_texture():
    rng = np.random.default_rng(7)
    heights = roof() + rng.normal(0, 0.05, (40, 40)).astype(np.float32)
    heights[5, 5] += 3.0  # a single stray cell is not a structure
    assert es.detect(heights, MINX, MAXY, [facet()]) == []


def test_a_gap_in_the_height_model_is_survivable():
    heights = roof()
    heights[:, :] = np.nan
    assert es.detect(heights, MINX, MAXY, [facet()]) == []


def test_tile_origin_reads_the_swisstopo_kilometre_name():
    from pathlib import Path

    path = Path("swisssurface3d-raster_2018_2683-1247_0.5_2056_5728.tif")
    assert es.tile_origin(path) == (2683000.0, 1247000.0)


def test_cache_prunes_to_its_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(es, "CACHE", tmp_path)
    monkeypatch.setattr(es, "CACHE_LIMIT_BYTES", 3000)
    for index in range(5):
        tile = tmp_path / f"tile{index}.tif"
        tile.write_bytes(b"x" * 1000)
        import os, time

        os.utime(tile, (time.time() + index, time.time() + index))
    es.prune_cache()
    left = sorted(p.name for p in tmp_path.glob("*.tif"))
    assert left == ["tile2.tif", "tile3.tif", "tile4.tif"]


def test_plane_settles_on_the_deck_not_the_plant_room():
    # A third of this flat roof carries a 3 m plant room. A symmetric fit is
    # pulled up between deck and plant until neither stands out.
    heights = roof()
    heights[:, 26:] += 3.0
    found = es.detect(heights, MINX, MAXY, [facet()])
    assert len(found) == 1
    assert found[0]["height_m"] == pytest.approx(3.0, abs=0.1)


def test_finds_a_low_rooflight_kerb():
    heights = roof()
    heights[16:24, 16:24] += 0.32  # a flat-roof rooflight, 32 cm proud
    found = es.detect(heights, MINX, MAXY, [facet()])
    assert len(found) == 1


def test_multipolygon_facet_is_handled():
    from shapely.geometry import MultiPolygon

    heights = roof()
    heights[18:22, 4:8] += 2.0
    pair = MultiPolygon(
        [
            box(MINX, MAXY - 20, MINX + 5, MAXY),
            box(MINX + 6, MAXY - 20, MINX + 20, MAXY),
        ]
    )
    assert es.facet_mask(pair, heights.shape, MINX, MAXY).any()
    assert len(es.detect(heights, MINX, MAXY, [pair])) == 1


def test_thickness_survives_lv95_coordinates():
    # float32 quantises eastings near 2.68 million to 0.25 m, which would
    # report this 0.3 m sliver as wider than it is.
    sliver = box(2683000.0, 1247000.0, 2683000.3, 1247020.0)
    assert es.thickness(sliver) == pytest.approx(0.3, abs=0.01)


def test_edge_slivers_are_not_structures():
    heights = roof()
    heights[:, 20:21] += 2.0  # a one-cell strip, as a taller neighbour gives
    assert es.detect(heights, MINX, MAXY, [facet()]) == []


def test_compact_two_cell_vent_is_not_lost_as_a_degenerate_line():
    heights = roof()
    heights[18,18:20] += 1.2
    found = es.detect(heights, MINX, MAXY, [facet()])
    assert len(found) == 1
    assert found[0]["area_m2"] == pytest.approx(.5)
    assert found[0]["height_m"] == pytest.approx(1.2,abs=.01)


def test_narrow_tall_neighbour_edge_is_still_rejected():
    heights = roof()
    heights[18,10:25] += 3
    assert es.detect(heights, MINX, MAXY, [facet()]) == []


def test_terrain_exclusion_with_missing_residual_has_json_safe_unknown_height():
    import json
    heights = roof()
    residual = np.zeros(heights.shape, dtype=np.float32)
    residual[10:20, 10:20] = np.nan
    above_ground = np.full(heights.shape, 10.)
    above_ground[10:20, 10:20] = 0.
    found = es.detect(heights, MINX, MAXY, [facet()],
                      fits=[{"residual": residual}], above_ground=above_ground)
    assert len(found) == 1 and found[0]["below_roof"]
    assert found[0]["height_m"] is None
    json.dumps([{k: v for k, v in item.items() if k != "geometry"} for item in found], allow_nan=False)


def test_ground_far_below_the_face_is_not_roof():
    # A Sonnendach face can span a block and take in its courtyard. Panels were
    # being packed onto the garden inside it.
    heights = roof()
    heights[8:32, 8:32] -= 6.0  # a courtyard, six metres down
    found = es.detect(heights, MINX, MAXY, [facet()])
    assert len(found) == 1
    assert found[0]["below_roof"] is True
    assert found[0]["area_m2"] > 100


def test_a_shallow_valley_is_still_roof():
    heights = roof()
    heights[16:24, 16:24] -= 0.6  # roof texture, not a storey
    assert es.detect(heights, MINX, MAXY, [facet()]) == []


def test_a_chimney_is_not_reported_as_ground():
    heights = roof()
    heights[18:22, 18:22] += 2.0
    found = es.detect(heights, MINX, MAXY, [facet()])
    assert len(found) == 1
    assert found[0]["below_roof"] is False


def test_terrain_tells_a_yard_from_a_roof():
    # The courtyard is the flatter, larger surface here, so a consensus fit on
    # the surface model alone settles on the ground and calls the roof an
    # obstacle. Height above terrain settles it.
    heights = roof()
    heights[:, :24] -= 8.0  # half the face is actually the yard below
    above = np.full(heights.shape, 9.0, np.float32)
    above[:, :24] = 0.2
    found = es.detect(heights, MINX, MAXY, [facet()], None, above)
    ground = [o for o in found if o["below_roof"]]
    assert ground, "the yard should be excluded"
    assert max(o["area_m2"] for o in ground) > 50


def test_a_roof_well_above_ground_is_left_alone():
    heights = roof()
    above = np.full(heights.shape, 9.0, np.float32)
    assert es.detect(heights, MINX, MAXY, [facet()], None, above) == []


def test_terrain_still_lets_a_chimney_through():
    heights = roof()
    heights[18:22, 18:22] += 2.0
    above = np.full(heights.shape, 9.0, np.float32)
    found = es.detect(heights, MINX, MAXY, [facet()], None, above)
    assert len(found) == 1 and found[0]["below_roof"] is False
