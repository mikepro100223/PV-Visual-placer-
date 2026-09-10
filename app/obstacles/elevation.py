"""Roof superstructure detection from the swisstopo swissSURFACE3D height model.

The shipped YOLO checkpoint is trained on Swiss masks that label PV only, so it
cannot find chimneys. The national 0.5 m surface model can: it measures the roof
surface itself, and anything standing proud of a roof face is a superstructure.
Each Sonnendach facet gets its own least-squares plane, so a pitched roof is
measured against its own slope instead of a single average height.

Flush features - roof windows set into the pitch, flat vents - do not rise above
the surface and are invisible here. They still need marking by hand.
"""

import asyncio
import math
import uuid
from pathlib import Path

import cv2
import httpx
import numpy as np
from PIL import Image
from pyproj import Transformer
from shapely.geometry import Polygon
from shapely import contains_xy

STAC = "https://data.geo.admin.ch/api/stac/v0.9/collections"
COLLECTION = "ch.swisstopo.swisssurface3d-raster"
TERRAIN_COLLECTION = "ch.swisstopo.swissalti3d"
TO_WGS84 = Transformer.from_crs(2056, 4326, always_xy=True)
CACHE = Path(__file__).resolve().parents[2] / "data" / "cache" / "dsm"
DSM_STEP_M = 0.5

# A superstructure must clear the roof face by this much to count. Flat-roof
# rooflight kerbs sit around 0.3 m, so the floor is set just under that.
MIN_HEIGHT_M = 0.28
# Consensus plane fit: a cell within this distance of a trial plane counts as
# lying on it, and this many trials are drawn.
FIT_TOLERANCE_M = 0.25
FIT_ITERATIONS = 200
FIT_SEED = 20260908
# Share of the residual spread taken as clear deck when sizing its roughness.
BASE_PERCENTILE = 50.0
# ...and the cut-off rises with the roughness of that deck.
NOISE_SIGMAS = 4.0
# ...and be at least this large, so single noisy cells are ignored.
MIN_AREA_M2 = 0.35
MAX_AREA_FRACTION = 0.5
MAX_OBSTACLES = 40
# A chimney or vent is small and tall; anything broader reads as a dormer.
CHIMNEY_MAX_AREA_M2 = 2.5
# Nothing on a roof is thinner than this across; anything that is comes from
# the roof outline running along a taller neighbour, not from a structure.
MIN_THICKNESS_M = 0.7
# A face can extend over ground that is not roof at all: Sonnendach outlines
# sometimes span a block and swallow its courtyard. Roof texture and valleys
# stay well within a metre, so anything this far below the fitted face is not
# part of it.
MIN_DROP_M = 1.5
MIN_DROP_AREA_M2 = 4.0
MAX_DROP_FRACTION = 0.95
# Surface minus terrain. A roof stands at least this far over the ground it
# covers, so anything lower inside an official outline is a yard, not a roof.
MIN_ROOF_HEIGHT_M = 2.0
# Strip sampled between two faces when asking whether building joins them.
BRIDGE_PROBE_M = 1.0
BRIDGE_MIN_CELLS = 6
BRIDGE_MIN_SHARE = 0.6
PAD_M = 2.0
MAX_TILES = 4
# Each tile is ~13 MB and covers a square kilometre; keep the cache bounded.
CACHE_LIMIT_BYTES = 2_000_000_000

Image.MAX_IMAGE_PIXELS = 80_000_000


class ElevationUnavailable(Exception):
    """The height model could not be read; the caller carries on without it."""


async def tile_hrefs(client: httpx.AsyncClient, bounds, collection=None) -> list[str]:
    minx, miny, maxx, maxy = bounds
    west, south = TO_WGS84.transform(minx, miny)
    east, north = TO_WGS84.transform(maxx, maxy)
    response = await client.get(
        f"{STAC}/{collection or COLLECTION}/items",
        params={"bbox": f"{west},{south},{east},{north}", "limit": 100},
    )
    response.raise_for_status()
    hrefs = []
    covered = set()
    features = sorted(response.json().get("features", []),
                      key=lambda f: str(f.get("properties", {}).get("datetime") or f.get("id", "")), reverse=True)
    for feature in features:
        for asset in feature.get("assets", {}).values():
            href = asset.get("href", "")
            if asset.get("type", "").startswith("image/tiff") and href.endswith(".tif"):
                name = href.rsplit("/", 1)[-1]
                # swissALTI3D also publishes a 2 m grid; keep the half-metre one
                # so terrain lines up cell for cell with the surface model.
                if f"_{DSM_STEP_M}_" not in name:
                    continue
                origin = tile_origin(Path(name))
                if origin in covered:
                    continue
                covered.add(origin)
                hrefs.append(href)
    return hrefs


def prune_cache() -> None:
    """Drop the least recently used tiles once the cache outgrows its budget."""
    tiles = sorted(CACHE.glob("*.tif"), key=lambda p: p.stat().st_mtime)
    total = sum(p.stat().st_size for p in tiles)
    while tiles and total > CACHE_LIMIT_BYTES:
        oldest = tiles.pop(0)
        total -= oldest.stat().st_size
        oldest.unlink(missing_ok=True)


async def cached_tile(client: httpx.AsyncClient, href: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / href.rsplit("/", 1)[-1]
    if path.exists() and path.stat().st_size > 0:
        path.touch()
        return path
    response = await client.get(href, follow_redirects=True, timeout=180)
    response.raise_for_status()
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".part")
    temporary.write_bytes(response.content)
    temporary.replace(path)
    prune_cache()
    return path


def tile_origin(path: Path) -> tuple[float, float]:
    """swisstopo names each tile by its lower-left LV95 kilometre."""
    for part in path.stem.split("_"):
        east, _, north = part.partition("-")
        if east.isdigit() and north.isdigit() and len(east) == 4:
            return float(east) * 1000, float(north) * 1000
    raise ElevationUnavailable("Cannot place height tile " + path.name)


def mosaic(paths: list[Path], bounds) -> tuple[np.ndarray, float, float]:
    """Cut the requested window out of one or more tiles, in LV95 metres."""
    minx, miny, maxx, maxy = bounds
    minx, miny = math.floor(minx / DSM_STEP_M) * DSM_STEP_M, math.floor(miny / DSM_STEP_M) * DSM_STEP_M
    maxx, maxy = math.ceil(maxx / DSM_STEP_M) * DSM_STEP_M, math.ceil(maxy / DSM_STEP_M) * DSM_STEP_M
    width = int(round((maxx - minx) / DSM_STEP_M))
    height = int(round((maxy - miny) / DSM_STEP_M))
    if width < 4 or height < 4:
        raise ElevationUnavailable("Roof is too small for the 0.5 m height model")
    out = np.full((height, width), np.nan, dtype=np.float32)
    for path in paths:
        try:
            with Image.open(path) as image:
                tile = np.asarray(image, dtype=np.float32)
        except (OSError, ValueError) as exc:
            raise ElevationUnavailable("Height tile could not be read") from exc
        ox, oy = tile_origin(path)
        rows, cols = tile.shape[:2]
        top = oy + rows * DSM_STEP_M
        # Overlap of this tile with the requested window, in output cells.
        col0 = max(0, int(round((ox - minx) / DSM_STEP_M)))
        row0 = max(0, int(round((maxy - top) / DSM_STEP_M)))
        src_col0 = max(0, int(round((minx - ox) / DSM_STEP_M)))
        src_row0 = max(0, int(round((top - maxy) / DSM_STEP_M)))
        take_w = min(width - col0, cols - src_col0)
        take_h = min(height - row0, rows - src_row0)
        if take_w <= 0 or take_h <= 0:
            continue
        patch = tile[src_row0 : src_row0 + take_h, src_col0 : src_col0 + take_w]
        target = out[row0 : row0 + take_h, col0 : col0 + take_w]
        np.copyto(target, patch, where=np.isnan(target))
    if bool(np.isnan(out).all()):
        raise ElevationUnavailable("No height data covers this roof")
    return out, minx, maxy


def facet_mask(polygon, shape, minx: float, maxy: float) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if polygon.geom_type == "MultiPolygon":
        for part in polygon.geoms:
            mask |= facet_mask(part, shape, minx, maxy).astype(np.uint8)
        return mask.astype(bool)
    if polygon.geom_type != "Polygon" or polygon.is_empty:
        return mask.astype(bool)
    rings = [polygon.exterior] + list(polygon.interiors)
    for index, ring in enumerate(rings):
        points = np.array(
            [
                [(x - minx) / DSM_STEP_M, (maxy - y) / DSM_STEP_M]
                for x, y in ring.coords
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [points], 0 if index else 1)
    return mask.astype(bool)


def fit_plane(heights: np.ndarray, mask: np.ndarray) -> dict | None:
    """Fit the roof face to its own slope, then measure what stands off it.

    Found by consensus rather than by trimming, because trimming cannot survive
    both tails. A rooftop plant room pushes cells up; a courtyard inside an
    over-large official outline drops them a whole storey. Anchoring to the low
    side lets the courtyard capture the plane, anchoring to the median lets an
    edge-aligned plant room tilt it - a sloped plane fits a step almost as well
    as a flat one does. Consensus takes the largest genuinely planar population
    wherever it sits, which is the roof.
    """
    valid = mask & np.isfinite(heights)
    if int(valid.sum()) < 12:
        return None
    rows, cols = np.nonzero(valid)
    z = heights[valid].astype(np.float64)
    design = np.column_stack(
        [cols.astype(np.float64), rows.astype(np.float64), np.ones(z.size)]
    )
    count = z.size
    # Seeded, so the same roof always returns the same plane.
    rng = np.random.default_rng(FIT_SEED)
    best_inliers, best_count = None, -1
    for _ in range(FIT_ITERATIONS):
        sample = rng.choice(count, 3, replace=False)
        corner = design[sample]
        if abs(float(np.linalg.det(corner))) < 1e-9:
            continue
        try:
            trial = np.linalg.solve(corner, z[sample])
        except np.linalg.LinAlgError:
            continue
        inliers = np.abs(z - design @ trial) <= FIT_TOLERANCE_M
        found = int(inliers.sum())
        if found > best_count:
            best_inliers, best_count = inliers, found
    keep = best_inliers if best_inliers is not None else np.ones(count, dtype=bool)
    if int(keep.sum()) < 8:
        keep = np.ones(count, dtype=bool)
    residual = np.zeros(count)
    solution = np.zeros(3)
    for _ in range(3):
        solution, *_ = np.linalg.lstsq(design[keep], z[keep], rcond=None)
        residual = z - design @ solution
        refined = np.abs(residual) <= FIT_TOLERANCE_M
        if int(refined.sum()) < 8:
            break
        keep = refined
    out = np.full(heights.shape, np.nan, dtype=np.float32)
    out[valid] = residual.astype(np.float32)
    deck = residual[keep]
    return {"residual": out, "coefficients": solution.tolist(),
            "point_count": int(count), "inlier_count": int(keep.sum()),
            "rmse_m": float(np.sqrt(np.mean(deck ** 2))) if deck.size else 0.0,
            "rank": int(np.linalg.matrix_rank(design[keep]))}


def plane_residual(heights: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    fitted = fit_plane(heights, mask)
    return fitted["residual"] if fitted else None


def rise_threshold(residual: np.ndarray, mask: np.ndarray) -> float:
    """How far above the fitted face a cell must sit before it counts.

    Never below MIN_HEIGHT_M, and lifted on a rough or poorly fitted face so
    that noise does not become a rooftop full of imaginary chimneys.
    """
    base = residual[mask & np.isfinite(residual)]
    if base.size == 0:
        return MIN_HEIGHT_M
    deck = base[base <= np.percentile(base, BASE_PERCENTILE)]
    if deck.size < 8:
        return MIN_HEIGHT_M
    spread = float(np.median(np.abs(deck - np.median(deck))))
    return max(MIN_HEIGHT_M, NOISE_SIGMAS * 1.4826 * spread)


def thickness(polygon: Polygon) -> float:
    """Width of the narrowest side of the tightest enclosing rectangle.

    Uses shapely rather than cv2.minAreaRect: LV95 eastings are around 2.68
    million, and float32 quantises that to 0.25 m, so a sliver measures wider
    than it is and slips through.
    """
    rectangle = polygon.minimum_rotated_rectangle
    if rectangle.geom_type != "Polygon":
        return 0.0
    coords = list(rectangle.exterior.coords)
    sides = [
        math.dist(coords[i], coords[i + 1]) for i in range(min(4, len(coords) - 1))
    ]
    return min(sides) if sides else 0.0


def detect(
    heights: np.ndarray, minx: float, maxy: float, facets: list[Polygon], fits=None,
    above_ground: np.ndarray | None = None,
) -> list[dict]:
    """Return superstructure polygons in LV95 metres, with height and class."""
    raised = np.zeros(heights.shape, dtype=np.uint8)
    dropped = np.zeros(heights.shape, dtype=np.uint8)
    depth = np.full(heights.shape, np.nan, dtype=np.float32)
    kernel = np.ones((3, 3), np.uint8)
    for index, facet in enumerate(facets):
        mask = facet_mask(facet, heights.shape, minx, maxy)
        if not mask.any():
            continue
        residual = (fits[index]["residual"] if fits[index] else None) if fits is not None else plane_residual(heights, mask)
        if residual is None:
            continue
        # A cell on a ridge straddles two faces, so it stands proud of both and
        # would be flagged on either. Fit on the whole face, judge only its core,
        # or every ridge line reads as a structure and welds them into one blob.
        core = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
        threshold = rise_threshold(residual, mask)
        hit = core & np.isfinite(residual) & (residual > threshold)
        raised[hit] = 1
        depth[hit] = np.fmax(depth[hit], residual[hit])
        below = core & np.isfinite(residual) & (residual < -MIN_DROP_M)
        if above_ground is not None:
            # Ground inside the outline is not roof at all, whatever its height
            # relative to the fitted face.
            below |= core & np.isfinite(above_ground) & (above_ground < MIN_ROOF_HEIGHT_M)
        dropped[below] = 1
        # A terrain-supported exclusion can have no fitted residual (e.g. a
        # courtyard at the edge of a face). Preserve its footprint, but never
        # let missing heights poison measured depths from overlapping faces.
        depth[below] = np.fmax(depth[below], np.abs(residual[below]))
    if not raised.any() and not dropped.any():
        return []
    # Close pinholes inside a chimney. Deliberately no opening: a 3x3 erosion
    # deletes a 1 m chimney outright, and the area filter below removes speckle.
    raised = cv2.morphologyEx(raised, cv2.MORPH_CLOSE, kernel)
    dropped = cv2.morphologyEx(dropped, cv2.MORPH_CLOSE, kernel)
    roof_area = sum(f.area for f in facets) or 1.0
    found = []
    for mask, below_roof in ((raised, False), (dropped, True)):
      if not mask.any():
          continue
      contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      for contour in contours:
          # Scale simplification to the structure: a fixed epsilon flattens a
          # 1 m chimney's four-cell contour into a line and loses it entirely.
          epsilon = min(0.4, 0.02 * cv2.arcLength(contour, True))
          approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
          if len(approx) < 3:
              x, y, w, h = cv2.boundingRect(contour)
              approx = np.array(
                  [[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]]
              )
          patch = np.zeros(mask.shape, np.uint8)
          cv2.drawContours(patch, [contour], -1, 1, -1)
          rises = depth[patch.astype(bool)]
          rises = rises[np.isfinite(rises)]
          rise = float(rises.max()) if rises.size else None
          ring = [
              (minx + (c + 0.5) * DSM_STEP_M, maxy - (r + 0.5) * DSM_STEP_M)
              for c, r in approx
          ]
          polygon = Polygon(ring)
          if not polygon.is_valid:
              polygon = polygon.buffer(0)
          # The contour traces cell centres, so it stops half a cell short of the
          # structure on every side. Grow it back, which also errs on the safe side.
          polygon = polygon.buffer(DSM_STEP_M / 2, join_style=2)
          if polygon.geom_type != "Polygon" or not polygon.is_valid:
              continue
          area = polygon.area
          floor = MIN_DROP_AREA_M2 if below_roof else MIN_AREA_M2
          ceiling = MAX_DROP_FRACTION if below_roof else MAX_AREA_FRACTION
          if area < floor or area > roof_area * ceiling:
              continue
          if thickness(polygon) < MIN_THICKNESS_M:
              continue
          found.append(
              {
                  "geometry": polygon,
                  "height_m": round(rise, 2) if rise is not None else None,
                  "area_m2": round(area, 2),
                  "below_roof": below_roof,
                  "kind": "other_obstacle" if below_roof
                      else ("chimney" if area <= CHIMNEY_MAX_AREA_M2 else "other_obstacle"),
              }
          )
    found.sort(key=lambda o: -o["area_m2"])
    return found[:MAX_OBSTACLES]


