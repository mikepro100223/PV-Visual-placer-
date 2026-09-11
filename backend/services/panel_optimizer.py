"""Deterministic physical grid search. Coordinates are pixels; dimensions are metres.

A pitched roof mounts modules flush against it, so they sit shoulder to shoulder
with a couple of centimetres between them. A flat roof does not: modules go on
tilted racks, and a rack shades the one behind it. Rows are therefore spaced by
the shadow the design sun casts, which on the Swiss plateau at the winter
solstice is a wider gap than the module itself.
"""

import math
import numpy as np
import shapely
from shapely.affinity import rotate

MAX_CANDIDATES = 120_000
OFFSET_STEPS = 4


def boundary_offsets(aligned, minimum, stride, extent, axis):
    """Keep uniform phases and also align modules to real roof/obstacle edges."""
    points = shapely.get_coordinates(aligned)[:, axis]
    values = np.concatenate([(points-minimum) % stride,
                             (points-minimum-extent) % stride])
    # Repeated coordinates describe long straight edges. Add their phases to
    # the old search, so this can never discard a previously tested layout.
    frequencies = {}
    for value in values:
        key = round(float(value / stride), 6)
        frequencies.setdefault(key, [0, float(value)])
        frequencies[key][0] += 1
    extra = sorted(frequencies.values(), key=lambda pair: (-pair[0], pair[1]))[:4]
    return sorted(set([float(v*stride/OFFSET_STEPS) for v in range(OFFSET_STEPS)]
                      + [pair[1] for pair in extra]))


def row_gap(panel, tilt_deg, module_length) -> float:
    """Ground gap that keeps a rack clear of its neighbour's shadow.

    The row behind clears the one in front when the horizontal shadow of a
    tilted module fits in the gap: length * sin(tilt) / tan(sun altitude).
    """
    tilt = math.radians(max(0.0, tilt_deg))
    altitude = math.radians(max(1.0, panel.design_sun_altitude_deg))
    if tilt <= 1e-6:
        return 0.0
    return module_length * math.sin(tilt) / math.tan(altitude)


def flat_roof_layout(panel, module_length) -> tuple[float, float]:
    """Footprint depth and row gap for a tilted rack, both in metres."""
    tilt = math.radians(max(0.0, panel.flat_roof_tilt_deg))
    depth = module_length * math.cos(tilt)
    gap = (panel.row_gap_m if panel.row_gap_m is not None
           else row_gap(panel, panel.flat_roof_tilt_deg, module_length))
    return depth, gap


def optimise_panels(usable, ppm, panel, angle=0, diagnostics=None, tilted=False):
    if diagnostics is not None:
        diagnostics.update(candidate_layouts_tested=0, candidate_panels_tested=0)
    if usable.is_empty:
        return [], "portrait"
    origin = usable.centroid.coords[0]
    aligned = rotate(usable, -angle, origin=origin)
    minx, miny, maxx, maxy = aligned.bounds
    best, orientation = [], "portrait"
    for name, w, h in [
        ("portrait", panel.width, panel.height),
        ("landscape", panel.height, panel.width),
    ]:
        # On a rack the module leans back, so its footprint is shorter than the
        # module and the next row starts a shadow's length further on.
        depth, gap_between_rows = (flat_roof_layout(panel, h) if tilted
                                   else (h, panel.gap))
        width, height = w * ppm, depth * ppm
        dx, dy = (w + panel.gap) * ppm, (depth + gap_between_rows) * ppm
        count = math.ceil((maxx - minx) / max(dx, 1e-6)) * math.ceil(
            (maxy - miny) / max(dy, 1e-6))
        if count > MAX_CANDIDATES:
            raise ValueError(
                "Scale creates too many candidate panels. Check your measurement or select a smaller roof."
            )
        for ox in boundary_offsets(aligned, minx, dx, width, 0):
            for oy in boundary_offsets(aligned, miny, dy, height, 1):
                if diagnostics is not None:
                    diagnostics["candidate_layouts_tested"] += 1
                xs = np.arange(minx + ox, maxx - width + 1e-7, dx)
                ys = np.arange(miny + oy, maxy - height + 1e-7, dy)
                if not len(xs) or not len(ys):
                    continue
                xx, yy = np.meshgrid(xs, ys)
                candidates = shapely.box(
                    xx.ravel(), yy.ravel(), xx.ravel() + width, yy.ravel() + height
                )
                valid = candidates[shapely.covers(aligned, candidates)]
                if diagnostics is not None:
                    diagnostics["candidate_panels_tested"] += len(candidates)
                if len(valid) > len(best):
                    best, orientation = list(valid), name
    return [
        list(rotate(p, angle, origin=origin).exterior.coords)[:-1] for p in best
    ], orientation
