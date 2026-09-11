from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from shapely import make_valid

MODE_FACTORS = {"conservative": 1.5, "recommended": 1.0, "maximum": 0.5}


def polygon_from_points(points, *, strict=False):
    polygon = Polygon(points)
    if strict and (not polygon.is_valid or polygon.area < 4):
        raise ValueError(
            "Draw a non-crossing roof polygon with at least three distinct points."
        )
    return make_valid(polygon)


def build_usable(roof, detections, ppm, settings):
    factor = MODE_FACTORS[settings.mode]
    inner = roof.buffer(-settings.edge_margin * factor * ppm, join_style=2)
    exclusions = []
    for item in detections:
        geometry = polygon_from_points(item["polygon"])
        is_rwa = "rwa" in [item["kind"], *item.get("kinds", [])]
        # Buffer the full object before intersecting the usable face. A chimney
        # just across a face boundary still requires clearance on this side.
        margin = (
            settings.pv_margin
            if item["kind"] == "existing_pv"
            else settings.obstacle_margin
        )
        # A union of PV and structural evidence must retain the larger clearance.
        for kind in item.get("kinds", []):
            margin = max(margin, settings.pv_margin if kind == "existing_pv" else settings.obstacle_margin)
        clearance = margin * factor
        if is_rwa:
            clearance = max(clearance, 2.0)
        exclusions.append(geometry.buffer(clearance * ppm, join_style=2))
    usable = inner.difference(unary_union(exclusions)) if exclusions else inner
    return usable, roof.difference(usable)


def geojson(geometry):
    return mapping(geometry)
