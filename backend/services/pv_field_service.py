"""Existing PV arrays found by colour, to back up the segmentation model.

The trained checkpoint is good on its own test split but misses large arrays on
real captures: on a Binzstrasse warehouse whose roof is half covered in modules
it returned three small patches, and 396 new modules were proposed straight on
top of the existing array.

Silicon under anti-reflective coating is strongly blue where clay, concrete,
gravel and bitumen are red- or brown-dominant, so an array separates from its
own roof on the blue-minus-red axis. The split is found with Otsu rather than a
fixed cut, because roofs differ, and the result is only believed when the bright
class is blue in absolute terms as well as relative ones - otherwise the bluer
half of a plain grey roof would be called an array.

Unlike the rooflight test this compares against the whole roof, not a local
neighbourhood: an array is metres across, and a local background subtraction
cancels exactly the large uniform regions being looked for.
"""

import cv2
import numpy as np
from shapely.geometry import Polygon

# The bright class must be genuinely blue, not merely the bluer half of grey.
# How blue an array photographs varies far more than a single roof suggested:
# the Binzstrasse warehouse this was first tuned on sits at +30, but measured
# arrays on nine other Zurich roofs run from +8 (Gruenmattstrasse 40, a full
# roof of modules) to +15. At +18 the gate rejected every one of them, so the
# colour route was switched off almost everywhere it was needed. Roofs with no
# array split at or below zero - a Ruemlang tile roof at -3, an Uetlibergstrasse
# roof at -1 - so the useful line lies just above zero, not near the brightest
# example. Shade and solidity, not this, are what keep false arrays out.
MIN_ABSOLUTE_BLUE = 5.0
# ...and the two classes must actually be distinct populations.
MIN_SEPARATION = 14.0
# When even the darker class is this blue, both sides of the split are array.
# Measured on a roof region that is all modules: the darker modules sit at +17
# against a plain roof's zero, and the two classes are only 11 apart - which the
# separation guard would otherwise read as "no array here" and return nothing.
COVERED_DARK_BLUE = 12.0
# A roof that is nearly all "bright" has no contrast to learn from; it is more
# likely blue-grey sheeting than a fully covered array. A real full-coverage
# roof is refused here, which is the safer of the two mistakes.
MAX_BRIGHT_SHARE = 0.92
# An array is at least a couple of modules.
MIN_AREA_M2 = 3.0
MAX_ARRAYS = 30
EDGE_MARGIN_M = 0.3
# Modules carry cell and frame lines, so an array is never smoother than the
# roof it sits on. This rejects smooth blue-grey metal sheeting.
MIN_TEXTURE_RATIO = 1.0
# Glass rooflights are blue too - they reflect the sky - and their frames make
# them as rough as a module field, so colour and texture alone let a bank of
# them be claimed as an array. What separates them is brightness: a module
# absorbs light and never photographs much brighter than the roof it sits on,
# while glazing does. Real arrays measured 0.55 to 1.15 against their roof;
# the glazing on a Hardhof sawtooth measured 1.69.
BRIGHT_CEILING = 1.25

# Modules are laid in rectangular blocks, so a real array is compact. Where the
# colour split spills along walkways it produces the opposite: a thin-walled
# blob wrapping around plant rooms. Rather than discard such a component - which
# on a roof-sized field threw away thousands of square metres of genuine array
# merely because separate blocks had touched - open it by this much and judge
# the pieces. A walkway two metres wide disappears; a block of modules does not.
SPLIT_M = 1.5
# An array is a compact block of modules. The same Oerlikon roof produced a
# ragged edge band at 0.45; the real arrays measured 0.61 and above.
MIN_SOLIDITY = 0.55
# Deep shade is dark, and so is an all-black module, and nothing in a single
# aerial frame reliably tells those two apart. Reporting shade as an existing
# array is the worse error - it wipes a usable roof off the estimate - so a
# component well below its roof's own brightness is refused unless it is blue
# in absolute terms, which shade never is. This keeps its own floor: it asks
# "is this blue enough to be a module despite being dark", which is a stronger
# question than "is there an array on this roof at all", and it must not follow
# that gate down.
SHADE_GREY_RATIO = 0.70
SHADE_BLUE_FLOOR = 18.0
# Gaps between module rows, and the odd vent standing inside an array, should
# not break one field into fragments.
CLOSE_M = 1.2
MAX_HOLE_M2 = 15.0
# How far the traced outline may depart from the pixels it describes. This has
# to be a real distance: as a fraction of perimeter it grew with the component,
# so a roof-sized array on Hardstrasse got a five-metre tolerance and its
# outline cut straight across bare gravel, claiming roof that is free to build
# on. A quarter of a metre follows the edge of a module.
SIMPLIFY_M = 0.25
# ...but an outline still has to fit through the API, which takes at most 200
# vertices per polygon. A roof-sized array traced at a quarter of a metre runs
# to several hundred, and the whole analysis was rejected for it. Where that
# happens the tolerance is relaxed until the outline fits, which costs a little
# precision on the largest arrays and nothing at all on ordinary ones.
MAX_VERTICES = 180


def _mask_from(polygon, shape) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    if polygon.geom_type == "MultiPolygon":
        for part in polygon.geoms:
            mask |= _mask_from(part, shape)
        return mask
    if polygon.is_empty or polygon.geom_type != "Polygon":
        return mask
    cv2.fillPoly(mask, [np.array(polygon.exterior.coords, np.int32)], 1)
    for interior in polygon.interiors:
        cv2.fillPoly(mask, [np.array(interior.coords, np.int32)], 0)
    return mask


def split_threshold(values: np.ndarray) -> tuple[float, float, float, float, float]:
    """Otsu split of the roof's blue-minus-red values, in original units.

    Returns the threshold, the mean of each class, their separation and the
    bright class's share. The dark class matters: if it is array-blue too, the
    split fell inside one array rather than between array and roof.
    """
    low, high = (float(v) for v in np.percentile(values, [1, 99]))
    span = max(high - low, 1e-6)
    scaled = np.clip((values - low) / span * 255, 0, 255).astype(np.uint8)
    level, _ = cv2.threshold(
        scaled.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold = low + level / 255.0 * span
    bright = values[values >= threshold]
    dark = values[values < threshold]
    if bright.size == 0 or dark.size == 0:
        return threshold, 0.0, 0.0, 0.0, 0.0
    return (threshold, float(bright.mean()), float(bright.mean() - dark.mean()),
            float(bright.size) / values.size, float(dark.mean()))


def fill_small_holes(mask: np.ndarray, max_pixels: int) -> np.ndarray:
    """Close interior gaps an array should not be broken by, keeping real ones."""
    if max_pixels <= 0:
        return mask
    holes = (mask == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(holes, 4)
    filled = mask.copy()
    border = set(labels[0].tolist()) | set(labels[-1].tolist())
    border |= set(labels[:, 0].tolist()) | set(labels[:, -1].tolist())
    for index in range(1, count):
        # A hole touching the frame is outside the array, not inside it.
        if index in border or stats[index, 4] > max_pixels:
            continue
        filled[labels == index] = 1
    return filled


def without_holes(polygon: Polygon) -> list[Polygon]:
    """Partition a valid mask without filling courtyards or deleting a slit."""
    from shapely import make_valid
    from shapely.geometry import LineString
    from shapely.ops import split, triangulate

    def polygons(geometry):
        if geometry.geom_type == "Polygon":
            return [geometry] if not geometry.is_empty else []
        return [p for child in getattr(geometry, "geoms", []) for p in polygons(child)]

    pending, out = polygons(make_valid(polygon)), []
    while pending:
        current = pending.pop()
        if not current.interiors:
            out.append(current)
            continue
        hole = max(current.interiors, key=lambda ring: Polygon(ring).area)
        point = Polygon(hole).representative_point()
        minx, _, maxx, _ = current.bounds
        pieces = polygons(split(current, LineString([(minx-1,point.y),(maxx+1,point.y)])))
        if len(pieces) <= 1:
            # An exact partition is safer than ever turning the hole into PV.
            pieces = [piece for triangle in triangulate(current)
                      for piece in polygons(triangle.intersection(current))]
        pending.extend(pieces)
    return [p for p in out if p.area > 0]


def fit_vertices(polygon: Polygon, tolerance: float) -> Polygon | None:
    """Bring an outline within the vertex budget the analysis will accept.

    Slitting a ring open folds the hole's boundary into the exterior, so two
    outlines that each fit become one that does not, and the analysis is
    rejected outright rather than losing a single array.
    """
    simplified = polygon
    while len(simplified.exterior.coords) - 1 > MAX_VERTICES:
        tolerance *= 1.5
        simplified = polygon.simplify(tolerance, preserve_topology=True)
        if simplified.geom_type != "Polygon" or simplified.is_empty:
            return None
        if tolerance > 1e4:
            return None
    return simplified


def _components(mask: np.ndarray, min_pixels: float):
    """Connected regions of a binary mask, as (boolean mask, pixel count)."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8)
    return [(labels == i, int(stats[i, 4])) for i in range(1, count)
            if stats[i, 4] >= min_pixels]


def split_blocks(part: np.ndarray, min_pixels: float, width_px: float):
    """Break a sprawling region at its narrow waists and return the blocks.

    Opening removes anything thinner than the kernel, so walkways and the
    single-module bridges that weld separate arrays together fall away while
    the blocks themselves survive. The pieces are then grown back so each block
    keeps its true extent rather than the eroded one.
    """
    size = max(3, int(round(width_px)) | 1)
    kernel = np.ones((size, size), np.uint8)
    cores = cv2.morphologyEx(part.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    pieces = _components(cores, min_pixels)
    if len(pieces) < 2:
        return []
    return [(cv2.dilate(piece.astype(np.uint8), kernel).astype(bool) & part,
             None) for piece, _ in pieces]


def detect(
    image: np.ndarray,
    roof: Polygon,
    pixels_per_metre: float,
    exclude: list[Polygon] | None = None,
) -> list[dict]:
    """Return existing-array polygons in image pixels."""
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
    if int(core.sum()) < int(MIN_AREA_M2 * pixels_per_metre**2):
        return []

    blue = rgb[..., 2] - rgb[..., 0]
    grey = rgb.mean(2)
    roof_blue = float(np.median(blue[core]))
    roof_grey = float(np.median(grey[core]))
    threshold, bright_mean, separation, share, dark_mean = split_threshold(blue[core])
    covered = dark_mean >= COVERED_DARK_BLUE
    blue_route = covered or not (
        bright_mean < MIN_ABSOLUTE_BLUE or separation < MIN_SEPARATION
        or share > MAX_BRIGHT_SHARE)

    if not blue_route:
        return []
    if covered:
        # Both classes are array-blue, so Otsu divided one array into its
        # brighter and darker modules instead of separating it from the roof.
        # On a face that is all array the two are 11 apart, the separation guard
        # rejects it, and every module is missed. Take the whole blue field; the
        # texture and solidity checks still have to agree it is an array.
        threshold = float(np.percentile(blue[core], 2))

    texture = np.abs(cv2.Laplacian(grey, cv2.CV_32F, ksize=3))
    hit = (core & (blue >= threshold)).astype(np.uint8)
    # Open first to drop speckle, then close the gaps between module rows so an
    # array reads as one region rather than a comb of stripes.
    hit = cv2.morphologyEx(hit, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # Close across a module row, so the gaps between rows do not split one array
    # into a comb of stripes.
    span = max(3, int(round(CLOSE_M * pixels_per_metre)) | 1)
    hit = cv2.morphologyEx(hit, cv2.MORPH_CLOSE, np.ones((span, span), np.uint8))
    hit = fill_small_holes(hit, int(MAX_HOLE_M2 * pixels_per_metre**2))
    if not hit.any():
        return []

    roof_texture = float(np.median(texture[core])) or 1.0
    cell = 1.0 / (pixels_per_metre**2)
    found = []
    floor_px = MIN_AREA_M2 * pixels_per_metre**2
    # Regions still to judge. A region that fails only because separate blocks
    # have merged is split once and its blocks judged in its place.
    queue = [(part, pixels, True) for part, pixels in _components(hit, floor_px)]
    while queue:
        part, pixels, may_split = queue.pop()
        if pixels is None:
            pixels = int(part.sum())
        if pixels < floor_px:
            continue
        if float(texture[part].mean()) / roof_texture < MIN_TEXTURE_RATIO:
            continue
        # Shade is the false positive that matters: a shaded half of a roof
        # reported as an existing array removes it from the estimate entirely.
        # Modules photographed in daylight are not markedly darker than the
        # roof they sit on; shade is.
        mean_blue = float(blue[part].mean())
        mean_grey = float(grey[part].mean())
        if (mean_grey < SHADE_GREY_RATIO * roof_grey
                and mean_blue < SHADE_BLUE_FLOOR):
            continue
        # ...and the opposite mistake: glazing, which is bright.
        if mean_grey > BRIGHT_CEILING * roof_grey:
            continue
        contours, hierarchy = cv2.findContours(
            part.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        epsilon = max(1.0, SIMPLIFY_M * pixels_per_metre)

        def traced(contour, tolerance):
            approx = cv2.approxPolyDP(contour, tolerance, True).reshape(-1, 2)
            return [(float(x), float(y)) for x, y in approx]

        outer = max(range(len(contours)), key=lambda k: cv2.contourArea(contours[k]))
        shell = traced(contours[outer], epsilon)
        while len(shell) > MAX_VERTICES:
            epsilon *= 1.5
            shell = traced(contours[outer], epsilon)
        if len(shell) < 3:
            continue
        # Courtyards and plant rooms standing inside a ring of modules are not
        # array, and a filled outline would take that roof out of the estimate.
        holes = [traced(c, epsilon) for k, c in enumerate(contours)
                 if hierarchy is not None and hierarchy[0][k][3] == outer
                 and cv2.contourArea(c) > (MIN_AREA_M2 * pixels_per_metre**2)]
        polygon = Polygon(shell, [h for h in holes
                                  if 3 <= len(h) <= MAX_VERTICES])
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.geom_type != "Polygon" or polygon.is_empty:
            continue
        # Measured from the component's own pixels, not the traced outline:
        # findContours drops holes, so a hollow ring would otherwise trace as a
        # filled rectangle and score a perfect solidity.
        hull = polygon.convex_hull.area
        if hull <= 0 or pixels / hull < MIN_SOLIDITY:
            if may_split:
                queue.extend((piece, None, False) for piece, _ in
                             split_blocks(part, floor_px,
                                          SPLIT_M * pixels_per_metre))
            continue
        for piece in without_holes(polygon):
            if piece.area * cell < MIN_AREA_M2:
                continue
            piece = fit_vertices(piece, epsilon)
            if piece is None:
                continue
            found.append(
                {
                    "geometry": piece,
                    "area_m2": round(piece.area * cell, 2),
                    "blue_shift": round(float(blue[part].mean()), 1),
                    "darker_than_roof": round(roof_grey - float(grey[part].mean()), 1),
                }
            )
    found.sort(key=lambda o: -o["area_m2"])
    return found[:MAX_ARRAYS]
