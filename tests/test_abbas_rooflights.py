"""Roof-window detection against synthetic aerial imagery."""

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from app.obstacles import rooflights as rs

PPM = 10.0
SIZE = 200
ROOF = box(10, 10, SIZE - 10, SIZE - 10)


def tiles(red=140, green=125, blue=118, noise=3.0, seed=1):
    """A brown tiled roof: red-dominant, like clay or concrete."""
    rng = np.random.default_rng(seed)
    image = np.zeros((SIZE, SIZE, 3), np.float32)
    for channel, level in enumerate((red, green, blue)):
        image[..., channel] = level + rng.normal(0, noise, (SIZE, SIZE))
    return np.clip(image, 0, 255).astype(np.uint8)


def put_window(image, cx, cy, width_m=1.0, height_m=1.2, rgb=(98, 101, 118)):
    half_w = int(width_m * PPM / 2)
    half_h = int(height_m * PPM / 2)
    image[cy - half_h : cy + half_h, cx - half_w : cx + half_w] = rgb
    return image


def test_finds_a_row_of_roof_windows():
    image = tiles()
    for i in range(4):
        put_window(image, 60 + i * 22, 100)
    found = rs.detect(image, ROOF, PPM)
    assert len(found) == 4
    assert all(0.5 <= f["area_m2"] <= 4.0 for f in found)


def test_a_bare_roof_stays_bare():
    assert rs.detect(tiles(), ROOF, PPM) == []


def test_shading_across_the_roof_does_not_invent_windows():
    # One merged roof spans sunlit and shaded faces; a global colour cut would
    # flag the whole darker half.
    image = tiles().astype(np.float32)
    image[:, : SIZE // 2] *= 0.45
    assert rs.detect(image.astype(np.uint8), ROOF, PPM) == []


def test_a_window_on_the_shaded_face_is_still_found():
    image = tiles().astype(np.float32)
    image[:, : SIZE // 2] *= 0.45
    image = image.astype(np.uint8)
    put_window(image, 50, 100, rgb=(44, 46, 58))
    assert len(rs.detect(image, ROOF, PPM)) == 1


def test_something_far_too_big_is_not_a_window():
    image = tiles()
    image[40:160, 40:160] = (98, 101, 118)  # a 12 m x 12 m blue expanse
    assert rs.detect(image, ROOF, PPM) == []


def test_a_thin_blue_line_is_not_a_window():
    image = tiles()
    image[100:102, 30:170] = (98, 101, 118)  # ridge flashing
    assert rs.detect(image, ROOF, PPM) == []


def test_known_obstacles_are_not_reported_twice():
    image = tiles()
    put_window(image, 100, 100)
    already = [box(90, 90, 110, 110)]
    assert rs.detect(image, ROOF, PPM, already) == []


def test_detections_cover_the_whole_window_not_its_bright_core():
    image = tiles()
    put_window(image, 100, 100, width_m=1.0, height_m=1.2)
    found = rs.detect(image, ROOF, PPM)
    assert len(found) == 1
    # The 1.0 x 1.2 m window must not come back materially undersized.
    assert found[0]["area_m2"] >= 1.2


def test_roof_edge_is_left_alone():
    image = tiles()
    put_window(image, 12, 100)  # right on the boundary, where flashing lives
    assert rs.detect(image, ROOF, PPM) == []


def test_empty_roof_is_survivable():
    assert rs.detect(tiles(), Polygon(), PPM) == []
