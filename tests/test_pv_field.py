"""Existing arrays found by colour, and the roofs that must not be mistaken for one."""

import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon, box

from backend.services import pv_field_service as pv

PPM = 10.0
SIZE = 240
ROOF = box(10, 10, SIZE - 10, SIZE - 10)


def roof_image(red=150, green=140, blue=132, noise=4.0, seed=3):
    """A grey-brown roof: red-dominant, lightly textured."""
    rng = np.random.default_rng(seed)
    image = np.zeros((SIZE, SIZE, 3), np.float32)
    for channel, level in enumerate((red, green, blue)):
        image[..., channel] = level + rng.normal(0, noise, (SIZE, SIZE))
    return np.clip(image, 0, 255).astype(np.uint8)


def put_array(image, x0, y0, x1, y1, seed=5):
    """A module field: strongly blue, and textured by its cell and frame lines."""
    rng = np.random.default_rng(seed)
    patch = image[y0:y1, x0:x1].astype(np.float32)
    patch[..., 0] = 95 + rng.normal(0, 3, patch.shape[:2])
    patch[..., 1] = 105 + rng.normal(0, 3, patch.shape[:2])
    patch[..., 2] = 140 + rng.normal(0, 3, patch.shape[:2])
    patch[::11, :, :] -= 45   # frame lines between module rows
    patch[:, ::11, :] -= 45
    image[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)
    return image


def test_a_module_field_is_found():
    image = put_array(roof_image(), 60, 60, 180, 170)
    found = pv.detect(image, ROOF, PPM)
    assert len(found) == 1
    assert found[0]["area_m2"] == pytest.approx(120 * 110 / PPM**2, rel=0.25)
    assert found[0]["blue_shift"] > pv.MIN_ABSOLUTE_BLUE


def test_a_bare_roof_claims_nothing():
    # Otsu always splits; the guards must stop that becoming a detection.
    assert pv.detect(roof_image(), ROOF, PPM) == []


def test_sun_and_shade_on_a_bare_roof_is_not_an_array():
    # Shaded roof is bluer as well as darker, which is the obvious false positive.
    image = roof_image().astype(np.float32)
    image[:, : SIZE // 2] *= 0.55
    image[:, : SIZE // 2, 2] += 6
    assert pv.detect(np.clip(image, 0, 255).astype(np.uint8), ROOF, PPM) == []


def test_a_ragged_edge_band_is_rejected():
    # The Oerlikon failure: a bluish parapet ring, solid enough in colour but
    # nothing like the compact block a module field forms.
    image = roof_image()
    ring = np.zeros((SIZE, SIZE), np.uint8)
    cv2.rectangle(ring, (14, 14), (SIZE - 14, SIZE - 14), 1, 7)
    image[ring.astype(bool)] = (95, 105, 145)
    assert pv.detect(image, ROOF, PPM) == []


def test_smooth_blue_sheeting_is_rejected_on_texture():
    image = roof_image()
    image[60:180, 60:170] = (95, 105, 145)  # flat colour, no module lines
    assert pv.detect(image, ROOF, PPM) == []


def test_something_far_too_small_is_not_an_array():
    image = put_array(roof_image(), 100, 100, 112, 112)
    assert pv.detect(image, ROOF, PPM) == []


def test_regions_already_known_are_not_reported_again():
    image = put_array(roof_image(), 60, 60, 180, 170)
    assert pv.detect(image, ROOF, PPM, [box(50, 50, 190, 180)]) == []


def test_an_empty_roof_is_survivable():
    assert pv.detect(roof_image(), Polygon(), PPM) == []
    assert pv.detect(roof_image(), ROOF, 0) == []


def test_the_split_reports_its_own_separation():
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(-5, 2, 500), rng.normal(30, 2, 500)])
    threshold, bright, separation, share, dark = pv.split_threshold(values)
    assert -5 < threshold < 30, threshold
    assert bright == pytest.approx(30, abs=1)
    assert separation == pytest.approx(35, abs=2)
    assert share == pytest.approx(0.5, abs=0.05)
    assert dark == pytest.approx(-5, abs=1)


def test_a_roof_that_is_all_array_is_still_found():
    """Otsu splits the array into brighter and darker modules, not array from
    roof, and the separation guard would then report an empty roof."""
    image = roof_image()
    image = put_array(image, 20, 20, SIZE - 20, SIZE - 20)
    found = pv.detect(image, ROOF, PPM)
    assert found, "an entirely covered roof must not read as bare"
    covered = sum(f["area_m2"] for f in found)
    assert covered > 0.6 * ROOF.area / PPM**2


def test_holes_inside_an_array_are_closed_but_large_gaps_survive():
    image = put_array(roof_image(), 40, 40, 200, 200)
    # A vent inside the field, and a genuine bare courtyard.
    image[110:118, 110:118] = (150, 140, 132)
    found = pv.detect(image, ROOF, PPM)
    assert len(found) >= 1
    biggest = max(found, key=lambda f: f["area_m2"])
    # The small vent is absorbed rather than splitting the array in four.
    assert biggest["area_m2"] > 150


def test_filling_leaves_holes_that_are_too_big():
    mask = np.ones((100, 100), np.uint8)
    mask[30:70, 30:70] = 0          # 1600 px hole
    filled = pv.fill_small_holes(mask, 400)
    assert filled[50, 50] == 0
    assert pv.fill_small_holes(mask, 4000)[50, 50] == 1


def test_filling_ignores_the_area_outside_the_shape():
    mask = np.zeros((100, 100), np.uint8)
    mask[40:60, 40:60] = 1
    # The surrounding background touches the frame and must not be filled in.
    assert pv.fill_small_holes(mask, 100000)[0, 0] == 0


def red_tile_roof(noise=4.0, seed=11):
    """A clay roof: strongly red-dominant, so nothing on it reads as blue."""
    rng = np.random.default_rng(seed)
    image = np.zeros((SIZE, SIZE, 3), np.float32)
    for channel, level in enumerate((165, 120, 100)):
        image[..., channel] = level + rng.normal(0, noise, (SIZE, SIZE))
    return np.clip(image, 0, 255).astype(np.uint8)


def put_dark_array(image, x0, y0, x1, y1, seed=12):
    """All-black modules: darker than the tiles, and only relatively bluer."""
    rng = np.random.default_rng(seed)
    patch = image[y0:y1, x0:x1].astype(np.float32)
    patch[..., 0] = 42 + rng.normal(0, 3, patch.shape[:2])
    patch[..., 1] = 44 + rng.normal(0, 3, patch.shape[:2])
    patch[..., 2] = 52 + rng.normal(0, 3, patch.shape[:2])
    patch[::11, :, :] += 30
    patch[:, ::11, :] += 30
    image[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)
    return image


def test_deep_shade_is_never_reported_as_an_array():
    """The error that matters: shade reported as PV removes a usable roof.

    A shaded half of a roof was returned as an existing array, which takes it
    out of the estimate altogether.
    """
    image = roof_image().astype(np.float32)
    image[:, : SIZE // 2] *= 0.35          # deep shade
    image[:, : SIZE // 2, 2] += 14         # sky-lit, so bluer as well as darker
    assert pv.detect(np.clip(image, 0, 255).astype(np.uint8), ROOF, PPM) == []


def test_a_dark_array_on_a_red_roof_is_deliberately_not_claimed():
    """An all-black array and deep shade look alike in one aerial frame.

    Nothing here separates them, so the darker case is left to the trained
    model and to manual marking rather than guessed at from brightness.
    """
    image = put_dark_array(red_tile_roof(), 60, 60, 180, 170)
    assert pv.detect(image, ROOF, PPM) == []


def test_a_bare_red_roof_claims_nothing():
    assert pv.detect(red_tile_roof(), ROOF, PPM) == []


def test_a_shaded_patch_beside_a_real_array_is_left_out():
    image = put_array(roof_image(), 60, 60, 180, 170)
    # A dark but featureless patch elsewhere: shade, not modules.
    image[190:215, 30:90] = (70, 68, 66)
    found = pv.detect(image, ROOF, PPM)
    assert found, "the real array must still be found"
    for face in found:
        assert face["geometry"].centroid.y < 190


def test_a_modestly_blue_array_is_still_found():
    """The gate was fitted to one very blue roof and rejected most real ones.

    Measured arrays on nine Zurich roofs sit between +8 and +15 on the blue
    axis, not the +30 of the warehouse the threshold came from.
    """
    image = roof_image(red=124, green=120, blue=116)
    image = put_array(image, 60, 60, 180, 170)
    # Bring the modules down to about +12 on the blue axis, the middle of the
    # range measured on real roofs, without making them dark enough to read as
    # shade - that is a separate test.
    image[60:170, 60:180, 2] = np.clip(
        image[60:170, 60:180, 2].astype(np.int16) - 33, 0, 255).astype(np.uint8)
    found = pv.detect(image, ROOF, PPM)
    assert found, "an array only ten points bluer than its roof was missed"


def test_the_outline_does_not_wander_off_the_modules():
    """Simplification is a distance, not a fraction of the perimeter.

    As a fraction it grew with the region, so a roof-sized array was traced
    with a five-metre tolerance and its outline cut across bare roof.
    """
    image = put_array(roof_image(), 30, 30, 210, 210)
    found = pv.detect(image, ROOF, PPM)
    assert found
    traced = found[0]["geometry"]
    truth = box(30, 30, 210, 210)
    assert traced.difference(truth).area / truth.area < 0.05


def test_a_courtyard_inside_an_array_is_not_claimed():
    """A large gap ringed by modules is roof, and must stay available."""
    image = put_array(roof_image(), 40, 40, 200, 200)
    image[100:150, 100:150] = roof_image()[100:150, 100:150]
    found = pv.detect(image, ROOF, PPM)
    assert found
    assert not found[0]["geometry"].contains(box(110, 110, 140, 140)), \
        "the courtyard was claimed as array"


def test_two_arrays_joined_by_a_walkway_are_reported_as_blocks():
    """Merged blocks are split, not discarded.

    A thin bridge between two module fields made one sprawling region that
    failed the compactness test, and thousands of square metres of genuine
    array were dropped for it.
    """
    image = roof_image()
    image = put_array(image, 30, 30, 100, 210)
    image = put_array(image, 140, 30, 210, 210)
    image = put_array(image, 100, 110, 140, 125)   # the bridge
    found = pv.detect(image, ROOF, PPM)
    assert sum(a["area_m2"] for a in found) > 20.0, \
        "the joined arrays were thrown away instead of split"


def test_splitting_leaves_a_single_solid_block_alone():
    image = put_array(roof_image(), 60, 60, 180, 170)
    part = np.zeros((SIZE, SIZE), bool)
    part[60:170, 60:180] = True
    assert pv.split_blocks(part, 100, 1.5 * PPM) == []


def test_an_outline_fits_through_the_api():
    """A tighter tolerance produced outlines the API refused, failing the
    whole analysis with a validation error rather than losing one array."""
    rng = np.random.default_rng(7)
    image = roof_image()
    image = put_array(image, 20, 20, 220, 220)
    # Chew a ragged edge into the field so its outline is genuinely complex.
    for _ in range(400):
        x, y = rng.integers(20, 215, 2)
        image[y:y + 4, x:x + 4] = roof_image()[y:y + 4, x:x + 4]
    for array in pv.detect(image, ROOF, PPM):
        ring = array["geometry"]
        assert len(ring.exterior.coords) <= pv.MAX_VERTICES + 1
        for hole in ring.interiors:
            assert len(hole.coords) <= pv.MAX_VERTICES + 1


def test_a_ring_of_modules_does_not_reclaim_its_courtyard():
    """The analysis carries one ring per object and no interiors, so a hole
    would be silently dropped and the courtyard billed as array."""
    ring = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)],
                   [[(30, 30), (70, 30), (70, 70), (30, 70)]])
    pieces = pv.without_holes(ring)
    assert pieces
    assert all(not piece.interiors for piece in pieces)
    courtyard = box(35, 35, 65, 65)
    assert all(not piece.intersects(courtyard) for piece in pieces)
    kept = sum(piece.area for piece in pieces)
    assert abs(kept - ring.area) / ring.area < 0.05


def test_a_solid_block_is_left_whole():
    solid = box(0, 0, 50, 50)
    assert pv.without_holes(solid) == [solid]


def test_glazing_is_not_claimed_as_an_array():
    """Rooflights reflect the sky, so they are blue, and their frames make them
    as rough as a module field. What separates them is that glass photographs
    brighter than the roof and a module never does."""
    image = roof_image(red=120, green=118, blue=116)
    glass = image[60:170, 60:180].astype(np.float32)
    glass[..., 0] = 190
    glass[..., 1] = 200
    glass[..., 2] = 215
    glass[::11, :, :] -= 60      # glazing bars
    glass[:, ::11, :] -= 60
    image[60:170, 60:180] = np.clip(glass, 0, 255).astype(np.uint8)
    assert pv.detect(image, ROOF, PPM) == []


def test_a_real_array_survives_the_brightness_ceiling():
    image = put_array(roof_image(), 60, 60, 180, 170)
    assert pv.detect(image, ROOF, PPM)
