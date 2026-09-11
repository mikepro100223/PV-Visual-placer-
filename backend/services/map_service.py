"""Swiss roof lookup and metric aerial capture. No screenshot/zoom scale guessing."""

import base64
import html
import io
import math
import hashlib
import uuid
import json
import re
from datetime import datetime, timezone

import httpx
from PIL import Image
from pyproj import Transformer
from shapely.geometry import Point, Polygon, mapping, shape
from shapely.ops import transform, unary_union

from backend.schemas.map import MapSelection
import numpy as np

from backend.services.elevation_service import ElevationUnavailable, roof_model
from backend.services.roof_plane import RoofPlane, official_plane
from backend.services.runtime_cache import captures, prepared, geodata, imagery
from backend.services.roof_graph import TOUCH_TOLERANCE_M, rejected_summary, select_connected
from backend.services import buildings3d_service as buildings3d
from backend.services.image_alignment import estimate_shift
from backend.services.rooflight_service import detect as detect_rooflights
from backend.services.pv_field_service import detect as detect_pv_fields
from backend.services.hvac_service import detect as detect_hvac
from backend.services.pv_register_service import RegisterUnavailable, registered_pv
from backend.services import vintage_service
from backend.services.geneva_service import superstructures, surveyed_polygons

API = "https://api3.geo.admin.ch/rest/services/ech"
ROOF_LAYER = "ch.bfe.solarenergie-eignung-daecher"
IMAGERY_LAYER = "ch.swisstopo.swissimage"
TO_SWISS = Transformer.from_crs(4326, 2056, always_xy=True)
TO_WGS84 = Transformer.from_crs(2056, 4326, always_xy=True)
MAX_IMAGE_SIZE = 1280
TARGET_PPM = 10.0
CAPTURE_PADDING_M = 8.0
GAP_CLOSE_M = 0.5
MAX_VERTICES = 200
MAX_CANDIDATES = 48
MIN_PLANE_AREA_M2 = 0.25
MIN_CANDIDATE_AREA_M2 = 2.0
MIN_HOLE_AREA_M2 = 0.25
MIN_DRAWN_AREA_M2 = 1.0
# Faces this close in orientation, and this close together, are one surface.
COPLANAR_PITCH_DEG = 6.0
COPLANAR_AZIMUTH_DEG = 12.0
COPLANAR_GAP_M = 0.6
FLAT_PITCH_DEG = 5.0


def enclosing_roof_features(features, click):
    """Select through an enclosed roof opening, never snap across a street."""
    groups = {}
    for feature in features:
        properties = feature.get("properties") or feature.get("attributes") or {}
        identifier = properties.get("building_id")
        if identifier is not None:
            groups.setdefault(identifier, []).append(feature)
    matches = []
    for group in groups.values():
        planes = feature_planes(group, click)
        if not planes:
            continue
        geometry = unary_union([p["geometry"] for p in planes])
        parts = geometry.geoms if geometry.geom_type == "MultiPolygon" else [geometry]
        for part in parts:
            if part.geom_type == "Polygon" and Polygon(part.exterior).covers(click):
                matches.append((part.area, group))
    return min(matches, key=lambda pair: pair[0])[1] if matches else []


class MapServiceError(Exception):
    pass


async def get_json(client: httpx.AsyncClient, path: str, params: dict) -> dict:
    key = (path, json.dumps(params, sort_keys=True))
    cached = geodata.get(key)
    if cached is not None:
        return json.loads(json.dumps(cached))
    response = await client.get(API + path, params=params)
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise MapServiceError("The Swiss map service could not complete this lookup.")
    geodata.put(key, data)
    return json.loads(json.dumps(data))


async def building_address(client, geometry, click, egids):
    """Only label verified registry matches, never an arbitrary nearby house."""
    bounds = geometry.bounds if geometry is not None else click.buffer(15).bounds
    bbox = ",".join(str(v + (-5 if i < 2 else 5)) for i, v in enumerate(bounds))
    try:
        data = await get_json(client, "/MapServer/identify", {
            "geometry": bbox, "geometryType": "esriGeometryEnvelope",
            "layers": "all:ch.bfs.gebaeude_wohnungs_register", "sr": 2056,
            "geometryFormat": "geojson", "returnGeometry": "true", "tolerance": 0,
            "mapExtent": bbox, "imageDisplay": "1000,1000,96", "limit": 100})
        known = {str(e) for e in egids if e is not None}
        matches = []
        parts = ([] if geometry is None else
                 list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry])
        for feature in data.get("results", []):
            props = feature.get("properties") or feature.get("attributes") or {}
            street = props.get("strname_deinr")
            if not street:
                continue
            identity = str(props.get("egid")) in known
            try:
                point = Point(float(props["gkode"]), float(props["gkodn"]))
                inside = any(Polygon(part.exterior).covers(point) for part in parts)
            except (KeyError, TypeError, ValueError):
                point, inside = click, False
            if identity or inside:
                label = f"{street}, {props.get('dplz4', '')} {props.get('dplzname', '')}".strip()
                matches.append((not identity, point.distance(click), label, props.get("egid")))
        if matches:
            matches.sort()
            return {"label": matches[0][2], "egid": matches[0][3],
                    "source": "Federal Register of Buildings and Dwellings (GWR)"}
    except (httpx.HTTPError, ValueError, MapServiceError):
        pass
    return None


async def search_locations(query: str) -> list[dict]:
    async with httpx.AsyncClient(timeout=20) as client:
        data = await get_json(
            client,
            "/SearchServer",
            {
                "searchText": query,
                "type": "locations",
                "sr": 4326,
                "limit": 7,
                "lang": "en",
            },
        )
    return [
        {
            "label": html.unescape(re.sub(r"<[^>]+>", "", item["attrs"]["label"])),
            "latitude": item["attrs"]["lat"],
            "longitude": item["attrs"]["lon"],
            "is_address": item["attrs"].get("origin") == "address",
        }
        for item in data.get("results", [])
        if "lat" in item.get("attrs", {})
    ]


def bounded_simplify(polygon: Polygon) -> Polygon | None:
    """Preserve topology while keeping the editable vertex count workable."""
    simplified = polygon.simplify(0.025, preserve_topology=True)
    tolerance = 0.05
    while len(simplified.exterior.coords) > MAX_VERTICES and tolerance <= 1:
        simplified = polygon.simplify(tolerance, preserve_topology=True)
        tolerance *= 2
    return simplified if len(simplified.exterior.coords) <= MAX_VERTICES else None


def drawn_plane(selection: MapSelection) -> dict:
    """Turn an outline drawn on the map into a roof, bypassing the official lookup."""
    ring = [
        TO_SWISS.transform(point.longitude, point.latitude)
        for point in selection.polygon
    ]
    polygon = Polygon(ring)
    if not polygon.is_valid or polygon.area < MIN_DRAWN_AREA_M2:
        raise MapServiceError(
            "Draw a non-crossing outline of at least 1 m². Click each corner, then finish."
        )
    simplified = bounded_simplify(polygon)
    if simplified is None:
        raise MapServiceError("That outline has too many points. Draw a simpler shape.")
    return {
        "id": "drawn",
        "geometry": simplified,
        "properties": {},
        "contains_click": True,
        "distance": 0.0,
    }


def connected_roof_members(planes: list[dict], bridge=None) -> list[dict]:
    """Faces physically attached to the clicked one; see roof_graph."""
    return select_connected(planes, bridge)[0]


def roof_selection(planes: list[dict], bridge=None):
    """Accepted faces plus the decision record behind each candidate."""
    return select_connected(planes, bridge)


async def expand_connected_roofs(client, features, click, warnings):
    """Discover neighbouring roof records and fetch all faces of each section."""
    initial = feature_planes(features, click)
    # Only the clicked record's siblings have been fetched so far. Other
    # records can already appear at the click where roofs overlap.
    loaded = {initial[0]["properties"].get("building_id")} if initial else set()
    for _ in range(4):
        current = connected_roof_members(feature_planes(features, click))
        if not current:
            break
        bounds = unary_union([p["geometry"] for p in current]).bounds
        minx, miny, maxx, maxy = bounds
        nearby = await get_json(client, "/MapServer/identify", {
            "geometry": f"{minx-1},{miny-1},{maxx+1},{maxy+1}",
            "geometryType": "esriGeometryEnvelope", "layers": "all:" + ROOF_LAYER,
            "sr": 2056, "geometryFormat": "geojson", "returnGeometry": "true",
            "tolerance": 0, "mapExtent": f"{minx-10},{miny-10},{maxx+10},{maxy+10}",
            "imageDisplay": "1000,1000,96", "limit": 1000, "lang": "en"})
        candidates = feature_planes(features + nearby.get("results", []), click)
        connected = connected_roof_members(candidates)
        building_ids = {p["properties"].get("building_id") for p in connected} - loaded - {None}
        if not building_ids and {p["id"] for p in connected} == {p["id"] for p in current}:
            break
        accepted_ids = {p["id"].split(":")[0] for p in connected}
        features += [f for f in nearby.get("results", [])
                     if str(f.get("featureId", f.get("id"))) in accepted_ids]
        if len(loaded | building_ids) > 32:
            warnings.append("Connected roof complex exceeds the lookup limit. Check the boundary or select a specific roof section.")
            break
        for building in sorted(building_ids, key=str):
            siblings = await get_json(client, "/MapServer/find", {
                "layer": ROOF_LAYER, "searchField": "building_id", "searchText": str(building),
                "contains": "false", "sr": 2056, "geometryFormat": "geojson",
                "returnGeometry": "true", "lang": "en", "limit": 1000})
            features += siblings.get("results", [])
        loaded.update(building_ids)
    else:
        warnings.append("Connected roof lookup reached its limit. Check that the full roof is outlined.")
    return features


def merge_building(planes: list[dict], click: Point, bridge=None) -> dict | None:
    """Union every Sonnendach plane of the clicked building into one roof outline.

    Sonnendach splits a roof into one facet per pitch/azimuth - 58 of them on a
    Zurich block. Analysing only the clicked facet reports a fraction of the roof,
    so the whole building is merged and the facets stay available as candidates.
    """
    if not planes:
        return None
    anchor = next((p for p in planes if p.get("contains_click")), planes[0])
    building = anchor["properties"].get("building_id")
    members, decisions = roof_selection(planes, bridge)
    summary = rejected_summary(decisions)
    if len(members) == 1:
        # Carry the record even for a lone face: an empty member list used to
        # fall back to "every plane with this building_id", which is exactly
        # the selection this graph exists to prevent.
        return {**anchor, "member_ids": [anchor["id"]], "merged_planes": 1,
                "facets": [anchor["geometry"]], "decisions": decisions,
                "selection_summary": summary,
                "source_building_ids": [building] if building is not None else []}
    geometries = [p["geometry"] for p in members]
    merged = unary_union(geometries)
    if merged.geom_type != "Polygon":
        # Facets that only touch at a corner need a hairline close to join.
        closed = unary_union([g.buffer(GAP_CLOSE_M) for g in geometries]).buffer(
            -GAP_CLOSE_M
        )
        if closed.geom_type == "Polygon" and closed.area >= merged.area * 0.95:
            merged = closed
    if merged.geom_type != "Polygon":
        parts = [g for g in merged.geoms if g.geom_type == "Polygon"]
        covering = [g for g in parts if g.covers(click)]
        merged = max(covering or parts, key=lambda g: g.area)
    simplified = bounded_simplify(merged)
    if simplified is None or simplified.area < anchor["geometry"].area:
        return anchor
    return {
        "id": f"building:{building}",
        "geometry": simplified,
        "properties": {
            **anchor["properties"],
            # Pitch and azimuth belong to a single facet, not to the merged roof.
            "neigung": None,
            "ausrichtung": None,
        },
        "contains_click": True,
        "distance": 0.0,
        "merged_planes": len(members),
        "member_ids": [p["id"] for p in members],
        "decisions": decisions,
        "selection_summary": summary,
        "source_building_ids": sorted({p["properties"].get("building_id") for p in members
                                       if p["properties"].get("building_id") is not None}, key=str),
        "facets": geometries,
    }


def merge_coplanar(members: list[dict]) -> list[dict]:
    """Join touching faces that share an orientation into one physical plane.

    Sonnendach splits a roof by sub-area as well as by geometry, so one plane
    can arrive as several narrow strips. Packed separately, each is charged a
    full edge setback along a boundary that is not an edge at all: a villa in
    Seefeld arrived as 6.5 x 1.61 m strips at 1244 kWh/m2, which leaves 0.71 m
    once both margins are taken and fits no module, so a sunny roof returned
    nothing. Merged, the strips are one surface and pack normally.
    """
    if len(members) < 2:
        return members
    parents = list(range(len(members)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def orientation(plane):
        pitch = plane["properties"].get("neigung")
        azimuth = plane["properties"].get("ausrichtung")
        return (None if pitch is None else float(pitch),
                None if azimuth is None else float(azimuth))

    for a in range(len(members)):
        pitch_a, azimuth_a = orientation(members[a])
        for b in range(a + 1, len(members)):
            pitch_b, azimuth_b = orientation(members[b])
            if pitch_a is None or pitch_b is None:
                continue
            # Flat faces are left alone. Equal pitch and aspect says nothing
            # about height when both are zero, so joining them can weld two
            # levels of a stepped roof into one plane that exists nowhere - it
            # cost a 2,800 m2 industrial roof every one of its 412 modules.
            if min(pitch_a, pitch_b) <= FLAT_PITCH_DEG:
                continue
            if abs(pitch_a - pitch_b) > COPLANAR_PITCH_DEG:
                continue
            if azimuth_a is None or azimuth_b is None:
                continue
            turn = abs(azimuth_a - azimuth_b) % 360
            if min(turn, 360 - turn) > COPLANAR_AZIMUTH_DEG:
                continue
            if members[a]["geometry"].dwithin(members[b]["geometry"], COPLANAR_GAP_M):
                parents[root(a)] = root(b)

    groups: dict[int, list[int]] = {}
    for index in range(len(members)):
        groups.setdefault(root(index), []).append(index)
    merged = []
    for indices in groups.values():
        if len(indices) == 1:
            merged.append(members[indices[0]])
            continue
        parts = [members[i] for i in indices]
        union = unary_union([p["geometry"] for p in parts])
        if union.geom_type != "Polygon":
            merged.extend(parts)
            continue
        simplified = bounded_simplify(union)
        if simplified is None:
            merged.extend(parts)
            continue
        lead = max(parts, key=lambda p: p["geometry"].area)
        properties = dict(lead["properties"])
        # Sonnendach's per-face totals are additive; its irradiation is a mean,
        # so it has to be re-weighted or the baseline comparison drifts.
        total_area = 0.0
        weighted = 0.0
        for key in ("flaeche", "stromertrag", "flaeche_kollektoren"):
            values = [p["properties"].get(key) for p in parts]
            numbers = [float(v) for v in values if isinstance(v, (int, float))]
            if numbers:
                properties[key] = sum(numbers)
        for part in parts:
            area = part["properties"].get("flaeche") or part["geometry"].area
            irradiation = part["properties"].get("mstrahlung")
            if isinstance(irradiation, (int, float)):
                total_area += float(area)
                weighted += float(area) * float(irradiation)
        if total_area:
            properties["mstrahlung"] = round(weighted / total_area)
        merged.append({**lead, "geometry": simplified, "properties": properties,
                       "merged_faces": len(parts)})
    merged.sort(key=lambda p: p["id"])
    return merged


def feature_planes(features: list[dict], click: Point) -> list[dict]:
    planes = []
    seen = set()
    for feature in features:
        if not feature.get("geometry"):
            continue
        geometry = shape(feature["geometry"])
        parts = (
            list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
        )
        props = feature.get("properties", feature.get("attributes", {}))
        for index, part in enumerate(parts):
            if (
                part.geom_type != "Polygon"
                or not part.is_valid
                or part.area < MIN_PLANE_AREA_M2
            ):
                continue
            identifier = f"{feature.get('featureId', feature.get('id'))}:{index}"
            if identifier in seen:
                continue
            seen.add(identifier)
            simplified = bounded_simplify(part)
            if simplified is None:
                continue
            planes.append(
                {
                    "id": identifier,
                    "geometry": simplified,
                    "properties": {**props, "_source_projected_area": geometry.area},
                    "contains_click": part.covers(click),
                    "distance": part.distance(click),
                }
            )
    return sorted(
        planes,
        key=lambda p: (not p["contains_click"], p["distance"], p["geometry"].area),
    )


def public_plane(plane: dict) -> dict:
    props = plane["properties"]
    return {
        "id": plane["id"],
        "geometry": mapping(transform(TO_WGS84.transform, plane["geometry"])),
        "projected_area_m2": round(plane["geometry"].area, 2),
        "pitch_deg": props.get("neigung"),
        "azimuth_deg": props.get("ausrichtung"),
        "plane_number": props.get("df_nummer"),
        "building_id": props.get("building_id"),
        "contains_click": plane["contains_click"],
    }


def capture_grid(geometry: Polygon | None, center: Point, span_m=64) -> dict:
    if geometry is not None:
        minx, miny, maxx, maxy = geometry.bounds
        width = maxx - minx + CAPTURE_PADDING_M * 2
        height = maxy - miny + CAPTURE_PADDING_M * 2
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    else:
        width = height = span_m
        cx, cy = center.x, center.y
    if max(width, height) > 1000:
        raise MapServiceError(
            "This roof is too large for one capture. Select a smaller roof plane."
        )
    ppm = min(TARGET_PPM, MAX_IMAGE_SIZE / max(width, height))
    pixel_width = max(256, math.ceil(width * ppm))
    pixel_height = max(256, math.ceil(height * ppm))
    half_w, half_h = pixel_width / ppm / 2, pixel_height / ppm / 2
    return {
        "bbox": [cx - half_w, cy - half_h, cx + half_w, cy + half_h],
        "width": pixel_width,
        "height": pixel_height,
        "pixels_per_metre": ppm,
    }


def pixel_ring(coords, grid: dict) -> list[list[float]]:
    minx, _, _, maxy = grid["bbox"]
    ppm = grid["pixels_per_metre"]
    # The photograph shows a building leaning away from the camera, so map
    # geometry has to move with it. Zero until the capture has been aligned.
    shift_x, shift_y = grid.get("shift_px", (0.0, 0.0))
    points = [
        [round((x - minx) * ppm + shift_x, 4), round((maxy - y) * ppm + shift_y, 4)]
        for x, y, *_ in coords
    ]
    return points[:-1] if points[0] == points[-1] else points


def alignment(geometry: Polygon) -> float:
    coords = list(geometry.minimum_rotated_rectangle.exterior.coords)
    start, end = max(
        zip(coords, coords[1:]),
        key=lambda pair: Point(pair[0]).distance(Point(pair[1])),
    )
    angle = math.degrees(math.atan2(-(end[1] - start[1]), end[0] - start[0]))
    return round((angle + 90) % 180 - 90, 2)


async def prepare_capture(selection: MapSelection) -> dict:
    key = selection.model_dump_json()
    cached = prepared.get(key)
    if cached is not None and captures.get(cached["capture_id"]) is not None:
        return cached
    return prepared.put(key, await _prepare_capture(selection))


async def _prepare_capture(selection: MapSelection) -> dict:
    x, y = TO_SWISS.transform(selection.longitude, selection.latitude)
    click = Point(x, y)
    warnings = []
    planes = []
    whole = None
    selected = None
    async with httpx.AsyncClient(timeout=httpx.Timeout(35, connect=10)) as client:
        if selection.polygon is not None:
            selected = drawn_plane(selection)
            warnings.append(
                "Using the outline you drew. Scale comes from the map, so area and capacity stay metric."
            )
        else:
            try:
                data = await get_json(
                    client,
                    "/MapServer/identify",
                    {
                        "geometry": f"{x},{y}",
                        "geometryType": "esriGeometryPoint",
                        "layers": "all:" + ROOF_LAYER,
                        "sr": 2056,
                        "geometryFormat": "geojson",
                        "returnGeometry": "true",
                        "tolerance": 0,
                        "lang": "en",
                        "mapExtent": f"{x - 100},{y - 100},{x + 100},{y + 100}",
                        "imageDisplay": "1000,1000,96",
                    },
                )
                features = data.get("results", [])
                found = feature_planes(features, click)
                if not found:
                    # A centre click can land in an atrium excluded from the roof
                    # polygon. Look for an enclosing building instead of picking
                    # whichever neighbouring roof happens to be nearest.
                    nearby = await get_json(client, "/MapServer/identify", {
                        "geometry": f"{x-25},{y-25},{x+25},{y+25}",
                        "geometryType": "esriGeometryEnvelope", "layers": "all:" + ROOF_LAYER,
                        "sr": 2056, "geometryFormat": "geojson", "returnGeometry": "true",
                        "tolerance": 0, "mapExtent": f"{x-50},{y-50},{x+50},{y+50}",
                        "imageDisplay": "1000,1000,96", "limit": 1000, "lang": "en"})
                    features = enclosing_roof_features(nearby.get("results", []), click)
                    found = feature_planes(features, click)
                    if found:
                        warnings.append("Selected the building around the roof opening you clicked; the opening remains excluded.")
            except (httpx.HTTPError, ValueError, MapServiceError):
                features, found = [], []
                warnings.append("Sonnendach lookup unavailable; checking measured 3D building geometry.")
            if found:
                building = found[0]["properties"].get("building_id")
                if building is not None:
                    try:
                        siblings = await get_json(
                            client,
                            "/MapServer/find",
                            {
                                "layer": ROOF_LAYER,
                                "searchField": "building_id",
                                "searchText": str(building),
                                "contains": "false",
                                "sr": 2056,
                                "geometryFormat": "geojson",
                                "returnGeometry": "true",
                                "lang": "en",
                                "limit": 100,
                            },
                        )
                        features += siblings.get("results", [])
                    except (httpx.HTTPError, ValueError, MapServiceError):
                        warnings.append(
                            "Other roof planes could not be loaded; the clicked plane is available."
                        )
            if found and (not selection.roof_id or selection.roof_id.startswith("building:")):
                try:
                    features = await expand_connected_roofs(client, features, click, warnings)
                except (httpx.HTTPError, ValueError, MapServiceError):
                    warnings.append("Connected roof sections could not all be checked. Verify the outline before using the result.")
            planes = feature_planes(features, click)
            whole = merge_building(planes, click)
            if selection.roof_id:
                selected = (
                    whole
                    if whole is not None and whole["id"] == selection.roof_id
                    else next((p for p in planes if p["id"] == selection.roof_id), None)
                )
                if selected is None:
                    raise MapServiceError(
                        "That roof plane is no longer available. Click the building again."
                    )
            else:
                selected = whole
            if selected and len(selected.get("source_building_ids", [])) > 1:
                warnings.append("Included connected roof sections stored under different Sonnendach IDs. "
                                "This is a physical roof outline, not a verified land-property boundary; check attached neighbouring structures.")
            if selected is not None and selected.get("merged_planes", 1) > 1:
                warnings.append(
                    f"Loaded {selected['merged_planes']} individual Sonnendach roof faces. "
                    "Each face is optimised independently."
                )
        geometry = selected["geometry"] if selected else None
        members = []
        # swissBUILDINGS3D decides the physical roof where it can. Sonnendach
        # stays for irradiation and suitability, matched face to face by
        # overlap. Any failure falls through to the Sonnendach geometry below.
        measured = None
        if not selection.roof_id and not selection.polygon:
            try:
                building = await buildings3d.building_at(
                    client, selection.latitude, selection.longitude)
                candidates = buildings3d.as_roof_planes(building, click, planes)
                if candidates and building["footprint"].buffer(1.5).covers(click):
                    measured = {"building": building, "members": candidates}
                elif candidates and any(Polygon(p.exterior).covers(click) for p in
                        (building["footprint"].geoms if building["footprint"].geom_type == "MultiPolygon" else [building["footprint"]])):
                    measured = {"building": building, "members": candidates}
            except (buildings3d.Buildings3DUnavailable, httpx.HTTPError,
                    ValueError, KeyError, OSError, ImportError) as exc:
                warnings.append(
                    f"Measured 3D building geometry unavailable ({exc}); "
                    + ("the available Sonnendach faces were retained." if selected else "No automatic outline is available from this source.")
                )
        building3d = None
        if measured is not None:
            building = measured["building"]
            outline = building["footprint"]
            attributes = building.get("attributes") or {}
            single = (outline if outline.geom_type == "Polygon"
                      else max(outline.geoms, key=lambda g: g.area))
            # The measured outline traces every corner of the solid; the
            # analysis schema accepts 200 points.
            single = bounded_simplify(single) or single
            building3d = {"outline": outline, "display": single,
                          "attributes": attributes,
                          "faces": measured["members"]}
            if selected is None:
                # Measured geometry can stand alone; solar attributes remain
                # unknown rather than borrowing a neighbour's irradiation.
                members = measured["members"]
                selected = {**members[0], "geometry": single,
                            "geometry_source": "swissbuildings3d",
                            "member_ids": [p["id"] for p in members],
                            "merged_planes": len(members)}
                whole = selected
                geometry = single
                warnings.append("Roof found in swissBUILDINGS3D despite missing Sonnendach coverage. Annual solar yield is unavailable unless you supply it.")

        if selected is None:
            warnings.append("No automatic roof boundary found in Sonnendach or swissBUILDINGS3D. Scale is calibrated; draw the roof in the editor.")

        if selected is not None and not members:
            building = selected["properties"].get("building_id")
            member_ids = set(selected.get("member_ids", []))
            members = ([p for p in planes if p["id"] in member_ids]
                       if planes and member_ids and
                       (not selection.roof_id or selection.roof_id.startswith("building:"))
                       else [selected])
        members.sort(key=lambda p: p["id"])
        if building3d is not None and members:
            # The measured footprint says how far the building reaches, so any
            # official face inside it belongs to this roof even when the
            # connectivity walk never reached its record. A Ruemlang building
            # lost a whole 150 m2 wing that way.
            outline = building3d["outline"]
            known = {p["id"] for p in members}
            minx, miny, maxx, maxy = outline.bounds
            try:
                extra = await get_json(client, "/MapServer/identify", {
                    "geometry": f"{minx-2},{miny-2},{maxx+2},{maxy+2}",
                    "geometryType": "esriGeometryEnvelope",
                    "layers": "all:" + ROOF_LAYER, "sr": 2056,
                    "geometryFormat": "geojson", "returnGeometry": "true",
                    "tolerance": 0,
                    "mapExtent": f"{minx-20},{miny-20},{maxx+20},{maxy+20}",
                    "imageDisplay": "1000,1000,96", "limit": 500, "lang": "en"})
                for plane in feature_planes(extra.get("results", []), click):
                    if plane["id"] in known:
                        continue
                    covered = plane["geometry"].intersection(outline).area
                    if covered >= max(MIN_PLANE_AREA_M2,
                                      0.6 * plane["geometry"].area):
                        members.append(plane)
                        known.add(plane["id"])
            except (httpx.HTTPError, ValueError, MapServiceError):
                warnings.append(
                    "Roof faces inside the measured footprint could not all be "
                    "fetched; part of the building may be missing.")
            members.sort(key=lambda p: p["id"])
        if building3d is not None and members and selected.get("geometry_source") != "swissbuildings3d":
            # swissBUILDINGS3D decides how far the building reaches; Sonnendach
            # faces are kept for their shape and solar record but cut to it.
            # The measured faces themselves are too finely triangulated to pack
            # against - on one block that cost 199 modules of 199 - so they set
            # the boundary rather than the packing surface.
            clipped = []
            for member in members:
                piece = member["geometry"].intersection(building3d["outline"])
                if piece.geom_type == "MultiPolygon":
                    piece = max(piece.geoms, key=lambda g: g.area)
                if piece.geom_type != "Polygon" or piece.area < MIN_PLANE_AREA_M2:
                    continue
                simplified = bounded_simplify(piece) or piece
                clipped.append({**member, "geometry": simplified,
                                "properties": {**member["properties"],
                                               "_clipped_to_buildings3d": True}})
            if clipped and any(p["contains_click"] for p in clipped):
                dropped = len(members) - len(clipped)
                members = clipped
                selected = {**selected, "geometry": building3d["display"],
                            "geometry_source": "swissbuildings3d",
                            "member_ids": [p["id"] for p in members]}
                geometry = selected["geometry"]
                whole = {**(whole or selected), "geometry": geometry}
                egid = building3d["attributes"].get("EGID")
                warnings.append(
                    "Building extent measured from swissBUILDINGS3D"
                    + (f" (EGID {egid})" if egid else "")
                    + f": {len(building3d['faces'])} roof surface(s), "
                    f"{building3d['outline'].area:.0f} m² footprint."
                    + (f" {dropped} official face(s) fell outside it."
                       if dropped else "")
                )
        before = len(members)
        members = merge_coplanar(members)
        if len(members) < before:
            warnings.append(
                f"Joined {before} official roof faces into {len(members)} physical surfaces. "
                "Sonnendach splits a plane by sub-area, and packing the strips "
                "separately charges an edge setback to boundaries that are not edges."
            )
        all_geometry = unary_union([p["geometry"] for p in members]) if members else geometry
        if building3d is not None and all_geometry is not None:
            # The capture has to cover what is drawn: the measured outline can
            # reach past the official faces clipped inside it.
            all_geometry = unary_union([all_geometry, selected["geometry"]])
        if len(members) > 1:
            selected = {**selected, "merged_planes": len(members)}
            whole = {**whole, "geometry": all_geometry}
        grid = capture_grid(all_geometry, click, selection.span_m)
        model_planes = [official_plane(p["geometry"], p["properties"]) for p in members]
        sunlight = [{"available": False, "reason": "Surrounding height data unavailable."} for _ in members]
        image_key = json.dumps(grid, sort_keys=True)
        image_content = imagery.get(image_key)
        if image_content is None:
            response = await client.get(
                "https://wms.geo.admin.ch/",
                params={
                    "SERVICE": "WMS",
                    "REQUEST": "GetMap",
                    "VERSION": "1.3.0",
                    "LAYERS": IMAGERY_LAYER,
                    "STYLES": "",
                    "CRS": "EPSG:2056",
                    "BBOX": ",".join(map(str, grid["bbox"])),
                    "WIDTH": grid["width"],
                    "HEIGHT": grid["height"],
                    "FORMAT": "image/jpeg",
                },
            )
            response.raise_for_status()
            if not response.headers.get("content-type", "").startswith("image/"):
                raise MapServiceError(
                    "The aerial-image service did not return an image. Try again shortly."
                )
            image_content = imagery.put(image_key, response.content)
        detected = []
        if geometry is not None:
            # The height model sees chimneys and dormers the PV-only model cannot.
            facets = [p["geometry"] for p in members]
            try:
                model = await roof_model(client, facets, all_geometry.bounds, [p["properties"] for p in members])
                detected, model_planes = model["obstacles"], model["planes"]
                sunlight = model["sunlight"]
                # Geometry chose these faces before any elevation was loaded.
                # Now that the surface model is here, ask it whether building
                # actually bridges each join, and withdraw the ones it does not
                # support. The window is the one already fetched, so this costs
                # no further download.
                bridge = model.get("bridge")
                if bridge is not None and len(members) > 1:
                    kept, checks = select_connected(members, bridge)
                    kept_ids = {p["id"] for p in kept}
                    if len(kept) < len(members):
                        dropped = [p["id"] for p in members if p["id"] not in kept_ids]
                        keep = [i for i, p in enumerate(members) if p["id"] in kept_ids]
                        members = [members[i] for i in keep]
                        model_planes = [model_planes[i] for i in keep]
                        sunlight = [sunlight[i] for i in keep]
                        facets = [p["geometry"] for p in members]
                        all_geometry = unary_union(facets)
                        selected = {**selected, "bridge_checked": True,
                                    "member_ids": sorted(kept_ids),
                                    "decisions": checks,
                                    "selection_summary": rejected_summary(checks)}
                        why = {}
                        for check in checks:
                            if not check["accepted"]:
                                why[check["reason"].split(" (")[0]] = True
                        warnings.append(
                            f"Withdrew {len(dropped)} roof face(s) after the height "
                            "check: " + "; ".join(sorted(why)) + "."
                        )
                    else:
                        selected = {**selected, "bridge_checked": True}
            except ElevationUnavailable as exc:
                warnings.append(
                    f"Roof superstructures could not be measured ({exc}). "
                    "Mark chimneys and roof windows by hand."
                )
        if building3d is not None and members:
            # Draw what was analysed, and only after the height check has had
            # its say. The measured solid can span two dwellings under separate
            # identifiers - a Ruemlang pair shares one roof under EGIDs 36499
            # and 36500 - and outlining a neighbour's half while proposing
            # nothing on it is the wrong half of the truth.
            analysed = unary_union([p["geometry"] for p in members])
            if analysed.geom_type == "MultiPolygon":
                analysed = max(analysed.geoms, key=lambda g: g.area)
            if analysed.geom_type == "Polygon" and not analysed.is_empty:
                geometry = bounded_simplify(analysed) or analysed
                selected = {**selected, "geometry": geometry}
                whole = {**(whole or selected), "geometry": geometry}
        image = Image.open(io.BytesIO(image_content)).convert("RGB")
        if image.size != (grid["width"], grid["height"]):
            raise MapServiceError(
                "The map image size did not match its scale. Please retry."
            )
        # Everything below converts between map and image, so settle the offset
        # between the two before any of it runs.
        alignment_result = {"applied": False, "shift_m": (0.0, 0.0),
                            "reason": "no roof outline to align"}
        if geometry is not None:
            alignment_result = estimate_shift(
                np.asarray(image), pixel_ring(geometry.exterior.coords, grid),
                grid["pixels_per_metre"])
            if alignment_result["applied"]:
                grid["shift_px"] = alignment_result["shift_px"]
                east, north = alignment_result["shift_m"]
                warnings.append(
                    f"Roof geometry moved {abs(east):.1f} m east/west and "
                    f"{abs(north):.1f} m north/south to match the photograph. "
                    "Aerial images are corrected for terrain, not for building "
                    "height, so a tall roof is drawn away from its coordinates."
                )
        encoded = io.BytesIO()
        image.save(encoded, format="JPEG", quality=95)
        # How old each source is. Five datasets answer for one roof and none
        # were surveyed on the same day.
        west, south = TO_WGS84.transform(grid["bbox"][0], grid["bbox"][1])
        east, north = TO_WGS84.transform(grid["bbox"][2], grid["bbox"][3])
        surface_years = [
            vintage_service.tile_year(href)
            for plane in model_planes
            for href in (plane.diagnostics.get("height_tiles") or [])
        ]
        vintage = {
            "imagery_year": await vintage_service.imagery_year(
                client, (west, south, east, north)
            ),
            "surface_year": max([y for y in surface_years if y], default=None),
            "roof_data_updated": (selected["properties"].get("datum_aenderung")
                                  if selected else None),
            "register_updated": "monthly",
        }
        # What Switzerland already records as installed, before any detection.
        register = {"known": False, "plant_count": 0, "total_power_kw": None,
                    "plants": [], "egids": [],
                    "basis": "SFOE register of electricity production plants.",
                    "coverage_note": "Not consulted."}
        egids = [p["properties"].get("gwr_egid") for p in members] if members else []
        if egids:
            try:
                register = await registered_pv(client, egids)
            except RegisterUnavailable as exc:
                warnings.append(f"Registered PV could not be checked ({exc}).")
        address = await building_address(client, all_geometry, click, egids)
        geneva = []
        if all_geometry is not None:
            try:
                key = "geneva:" + json.dumps(grid["bbox"])
                geneva = geodata.get(key)
                if geneva is None:
                    geneva = await superstructures(client, grid["bbox"])
                    geodata.put(key, geneva)
            except (httpx.HTTPError, ValueError):
                geneva = []
                warnings.append("Geneva surveyed superstructures unavailable; image and height detection remain active.")
    props = selected["properties"] if selected else {}
    roof_pixels = (
        pixel_ring(geometry.exterior.coords, grid) if geometry is not None else []
    )
    objects = []
    if all_geometry is not None:
        for part, attributes in surveyed_polygons(geneva, all_geometry):
            objects.append({"polygon": pixel_ring(part.exterior.coords, grid),
                            "kind": "other_obstacle", "source": "map"})
        if geneva:
            warnings.append("Included surveyed Geneva rooftop superstructures (SITG). Survey dates may differ from the aerial image; the catalogue is not exhaustive.")
    if geometry is not None:
        for ring in geometry.interiors:
            hole = Polygon(ring).simplify(0.05, preserve_topology=True)
            if hole.geom_type != "Polygon" or hole.area < MIN_HOLE_AREA_M2:
                continue
            polygon = pixel_ring(hole.exterior.coords, grid)
            # Rounding to pixels can collapse a sliver into a degenerate ring.
            if len(polygon) < 3 or not Polygon(polygon).is_valid:
                continue
            objects.append(
                {
                    "polygon": polygon,
                    "kind": "other_obstacle",
                    "source": "map",
                }
            )
    if geometry is not None and roof_pixels:
        # Flush roof windows never reach the height model; the photo shows them.
        shift_x, shift_y = grid.get("shift_px", (0.0, 0.0))
        outline = transform(
            lambda x, y: ((np.asarray(x)-grid["bbox"][0])*grid["pixels_per_metre"] + shift_x,
                          (grid["bbox"][3]-np.asarray(y))*grid["pixels_per_metre"] + shift_y),
            all_geometry)
        if outline.is_valid:
            already = [Polygon(o["polygon"]) for o in objects]
            already = [p for p in already if p.is_valid]
            # Arrays first: the segmentation model misses large ones, and a
            # module field would otherwise be read as a field of rooflights.
            for array in detect_pv_fields(
                np.asarray(image), outline, grid["pixels_per_metre"], already
            ):
                clipped = array["geometry"].intersection(outline)
                if clipped.geom_type != "Polygon" or clipped.is_empty:
                    continue
                ring = [[round(x, 4), round(y, 4)] for x, y in clipped.exterior.coords]
                ring = ring[:-1] if ring[0] == ring[-1] else ring
                if len(ring) < 3:
                    continue
                objects.append(
                    {"polygon": ring, "kind": "existing_pv", "source": "image"}
                )
            pv_regions = [Polygon(o["polygon"]) for o in objects if o["kind"] == "existing_pv"]
            voids = [Polygon(o["polygon"]) for o in objects if o["source"] == "map" and o["kind"] == "other_obstacle"]
            equipment_roof = outline.difference(unary_union(voids)) if voids else outline
            for equipment in detect_hvac(np.asarray(image), equipment_roof, grid["pixels_per_metre"], pv_regions):
                objects.append({"polygon": list(equipment["geometry"].exterior.coords)[:-1],
                                "kind": "other_obstacle", "source": "image"})
            already = [Polygon(o["polygon"]) for o in objects]
            already = [p for p in already if p.is_valid]
            for light in detect_rooflights(
                np.asarray(image), outline, grid["pixels_per_metre"], already
            ):
                clipped = light["geometry"].intersection(outline)
                if clipped.geom_type != "Polygon" or clipped.is_empty:
                    continue
                ring = [[round(x, 4), round(y, 4)] for x, y in clipped.exterior.coords]
                ring = ring[:-1] if ring[0] == ring[-1] else ring
                if len(ring) < 3:
                    continue
                objects.append(
                    {
                        "polygon": ring,
                        "kind": "skylight",
                        "source": "image",
                    }
                )
    for obstacle in detected if geometry is not None else []:
        clipped = obstacle["geometry"].intersection(all_geometry)
        if clipped.geom_type != "Polygon" or clipped.area < MIN_HOLE_AREA_M2:
            continue
        clipped = bounded_simplify(clipped) or clipped.convex_hull
        polygon = pixel_ring(clipped.exterior.coords, grid)
        if len(polygon) < 3 or not Polygon(polygon).is_valid:
            continue
        objects.append(
            {
                "polygon": polygon,
                "kind": obstacle["kind"],
                "source": "terrain" if obstacle.get("below_roof") else "elevation",
                "height_m": obstacle["height_m"],
            }
        )
    ground = [o for o in objects if o["source"] == "terrain"]
    if ground:
        warnings.append(
            f"Excluded {len(ground)} area(s) lying more than 1.5 m below the roof face. "
            "An official roof outline can span a whole block and take in its courtyard."
        )
    raised = [o for o in objects if o["source"] == "elevation"]
    lights = [o for o in objects if o["source"] == "image" and o["kind"] == "skylight"]
    equipment = [o for o in objects if o["source"] == "image" and o["kind"] == "other_obstacle"]
    if equipment:
        warnings.append(f"Excluded {len(equipment)} likely fan-bank housing(s) from repeated circular features in the image. This is image evidence, not a surveyed or trained equipment classification; verify the footprint.")
    arrays = [o for o in objects if o["source"] == "image" and o["kind"] == "existing_pv"]
    if arrays:
        warnings.append(
            f"Found {len(arrays)} existing PV area(s) by their colour against the roof. "
            "These are excluded from the new layout; check them against the image."
        )
    if raised:
        warnings.append(
            f"Measured {len(raised)} raised roof structures from the swisstopo height model."
        )
    if lights:
        warnings.append(
            f"Found {len(lights)} likely roof windows by their reflection in the aerial photo. "
            "Check them against the image and mark anything missed."
        )
    if not raised and not lights:
        warnings.append(
            "No roof structures found. That is not proof the roof is clear - "
            "mark any chimneys or roof windows yourself."
        )
    if register.get("known"):
        power = register.get("total_power_kw")
        warnings.append(
            f"The federal register records {register['plant_count']} PV installation(s) "
            f"on this building" + (f", {power:g} kW in total" if power else "")
            + ". The register holds capacity, not position, so where the panels sit "
            "still comes from the image."
        )
    summary = (selected or {}).get("selection_summary") or {}
    if summary.get("rejected_sharing_building_id"):
        warnings.append(
            f"{summary['rejected_sharing_building_id']} roof polygon(s) share this "
            "building's Sonnendach identifier but are not physically attached to the "
            "roof you clicked, so they are excluded. Open the face list to see why."
        )
    for note in vintage_service.findings(vintage, register):
        warnings.append(note)
    capture_id = uuid.uuid4().hex
    faces = [{**p, "plane": plane, "sunlight": sun} for p, plane, sun in zip(members, model_planes, sunlight)]
    # Hash the decoded JPEG, exactly as the analyse endpoint receives it.
    decoded = Image.open(io.BytesIO(encoded.getvalue())).convert("RGB")
    captures.put(capture_id, {"faces": faces, "grid": grid, "pv_register": register,
                 "latitude": selection.latitude, "longitude": selection.longitude,
                 "vintage": vintage,
                 "image_hash": hashlib.sha256(decoded.tobytes()).hexdigest(),
                 "warnings": list(warnings), "default_angle": alignment(geometry) if geometry is not None else 0})
    return {
        "capture_id": capture_id,
        "pv_register": register,
        "vintage": vintage,
        "roof_faces": [{**public_plane(p), "roof": pixel_ring(p["geometry"].exterior.coords, grid),
                        "plane": plane.describe()} for p, plane in zip(members, model_planes)],
        "alignment": alignment_result,
        "roof_selection": {
            **((selected or {}).get("selection_summary") or {}),
            "tolerance_m": TOUCH_TOLERANCE_M,
            "physical_check": ("swissSURFACE3D above swissALTI3D terrain"
                               if (selected or {}).get("bridge_checked")
                               else "geometry only"),
            "candidates": (selected or {}).get("decisions") or [],
        },
        "image_base64": base64.b64encode(encoded.getvalue()).decode(),
        "mime_type": "image/jpeg",
        "roof": roof_pixels,
        "objects": objects,
        "pixels_per_metre": grid["pixels_per_metre"],
        "angle": alignment(geometry) if geometry is not None else 0,
        "candidates": [
            public_plane(p)
            for p in members
            if p["geometry"].area >= MIN_CANDIDATE_AREA_M2
        ][:MAX_CANDIDATES],
        "building_outline": public_plane(whole) if whole else None,
        "selected_roof_id": selected["id"] if selected else None,
        "warnings": warnings,
        "provenance": {
            "imagery": "SWISSIMAGE / swisstopo",
            "address": address,
            "roof_source": (
                "Drawn on the map by you"
                if selection.polygon is not None
                else "swissBUILDINGS3D / swisstopo"
                if selected and selected.get("geometry_source") == "swissbuildings3d"
                else "Sonnendach / Swiss Federal Office of Energy"
                if selected
                else None
            ),
            "crs": "EPSG:2056",
            "bbox": grid["bbox"],
            "image_width": grid["width"],
            "image_height": grid["height"],
            "pixels_per_metre": grid["pixels_per_metre"],
            "latitude": selection.latitude,
            "longitude": selection.longitude,
            "feature_id": selected["id"] if selected else None,
            "building_id": props.get("building_id"),
            "source_building_ids": selected.get("source_building_ids", [props.get("building_id")]) if selected else [],
            "selection_basis": "Connected official roof sections; different known EGIDs remain separate" if selection.polygon is None else "User-drawn outline",
            "merged_planes": selected.get("merged_planes", 1) if selected else 0,
            "roof_area_m2": round(all_geometry.area, 2) if all_geometry is not None else None,
            "pitch_deg": props.get("neigung"),
            "azimuth_deg": props.get("ausrichtung"),
            "roof_data_updated": props.get("datum_aenderung"),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "scale_basis": "LV95 map metres, independent of display zoom. Each reliable DSM face is packed in true surface metres; unavailable faces use a labelled projected fallback.",
            "source_url": "https://map.geo.admin.ch/",
        },
    }
