"""Metre-accurate packing of complete modules into individual roof planes."""
from dataclasses import dataclass
import math

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

@dataclass(frozen=True)
class Module:
    name: str = 'Trina Vertex S+ TSM-NEG9RC.27 450 W'
    length_m: float = 1.762
    width_m: float = 1.134
    power_w: int = 450
    source: str = 'https://www.trinasolar.com/en-glb/NEG9RC.27/'

MODULE = Module()

def roof_transform(roof, slope, azimuth):
    """Return reciprocal map->roof and roof->map affine transforms.

    Azimuth is clockwise from north. Roof v follows the horizontal downslope
    direction, stretched by 1/cos(pitch); u follows the contour direction.
    """
    if not (0 <= slope < 80):
        raise ValueError('Roof pitch must be between 0 and 80 degrees')
    origin = roof.centroid
    if slope < 1:
        corners = list(roof.minimum_rotated_rectangle.exterior.coords)
        dx, dy = corners[1][0]-corners[0][0], corners[1][1]-corners[0][1]
        azimuth = math.degrees(math.atan2(-dy, dx))
    a, c = math.radians(azimuth), math.cos(math.radians(slope))
    matrix = np.array([[math.cos(a), -math.sin(a)], [math.sin(a)/c, math.cos(a)/c]])
    offset = -matrix @ np.array([origin.x, origin.y])
    forward = [*matrix[0], *matrix[1], *offset]
    inverse = np.linalg.inv(matrix)
    backward = [*inverse[0], *inverse[1], origin.x, origin.y]
    return forward, backward

def pack_panels(roof, obstacles, slope, azimuth, setback=0.3, obstacle_gap=0.25, module=MODULE):
    if setback < 0 or obstacle_gap < 0:
        raise ValueError('Clearances cannot be negative')
    if roof.is_empty or roof.area < 1:
        return [], dict(usable_area_m2=0, surface_area_m2=0)
    forward, backward = roof_transform(roof, slope, azimuth)
    plane = affinity.affine_transform(roof, forward)
    exclusions = [affinity.affine_transform(p, forward).buffer(obstacle_gap) for p in obstacles if not p.is_empty]
    usable = plane.buffer(-setback)
    if exclusions:
        usable = usable.difference(unary_union(exclusions))
    stats = dict(usable_area_m2=round(usable.area, 2), surface_area_m2=round(plane.area, 2))
    if usable.is_empty:
        return [], stats
    minx, miny, maxx, maxy = usable.bounds
    if maxx-minx > 300 or maxy-miny > 300:
        raise ValueError('Roof is too large for this local prototype')
    best = []
    for width, height in [(module.width_m,module.length_m),(module.length_m,module.width_m)]:
        step_x, step_y = width+0.02, height+0.02
        for ox in [0,0.25,0.5,0.75]:
            for oy in [0,0.25,0.5,0.75]:
                candidate = []
                for x in np.arange(minx+ox*step_x, maxx-width+1e-8, step_x):
                    for y in np.arange(miny+oy*step_y, maxy-height+1e-8, step_y):
                        panel = box(x,y,x+width,y+height)
                        if usable.covers(panel):
                            candidate.append(panel)
                if len(candidate) > len(best):
                    best = candidate
    return [affinity.affine_transform(p, backward) for p in best], stats

def esri_polygon(geometry):
    # Symmetric difference handles disjoint exteriors and interior rings without
    # assuming ring orientation supplied by an upstream API.
    result = Polygon()
    for ring in geometry.get('rings', []):
        polygon = Polygon([(p[0],p[1]) for p in ring]).buffer(0)
        result = result.symmetric_difference(polygon)
    return result
