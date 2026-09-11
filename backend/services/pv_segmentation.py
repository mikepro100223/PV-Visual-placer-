"""Overlapping native-resolution views preserve small arrays on large captures."""
from shapely.geometry import Polygon
from shapely.ops import unary_union
import math


def corroborated_pv(base, candidates):
    """Let another orientation complete an array already seen in the first pass.

    Do not admit new, unsupported objects from test-time augmentation: rotated
    roof edges and glazing can otherwise become false arrays. Require at least
    a quarter of each candidate to agree with the original PV evidence.
    """
    evidence = unary_union([Polygon(o["polygon"]).buffer(0) for o in base
                            if o["kind"] == "existing_pv"])
    accepted = []
    for obj in candidates:
        if obj["kind"] != "existing_pv":
            continue
        geometry = Polygon(obj["polygon"]).buffer(0)
        if geometry.area > 0 and geometry.intersection(evidence).area >= .25 * geometry.area:
            accepted.append(obj)
    return accepted


def detection_views(image, tile=640, overlap=160):
    yield image, 0, 0
    if max(image.size) <= 768:
        return
    tile = max(tile, math.ceil(max(image.size)/3))
    overlap = tile // 4
    def starts(length):
        return sorted(set([*range(0, max(1, length-tile+1), tile-overlap), max(0, length-tile)]))
    for y in starts(image.height):
        for x in starts(image.width):
            yield image.crop((x, y, min(image.width, x+tile), min(image.height, y+tile))), x, y


def merge_detections(objects):
    """Combine overlapping same-class masks from full image and tile passes."""
    merged = []
    for obj in sorted(objects, key=lambda o: -o["confidence"]):
        polygon = Polygon(obj["polygon"]).buffer(0)
        if polygon.is_empty or polygon.geom_type != "Polygon":
            continue
        hits = [i for i, old in enumerate(merged) if old["kind"] == obj["kind"]
                and polygon.intersection(old["geometry"]).area / min(polygon.area, old["geometry"].area) > .5]
        if hits:
            polygon = unary_union([polygon] + [merged[i]["geometry"] for i in hits])
            obj = {**obj, "confidence": max([obj["confidence"]] + [merged[i]["confidence"] for i in hits])}
            merged = [old for i, old in enumerate(merged) if i not in hits]
        merged.append({**obj, "geometry": polygon})
    # The API represents one exterior per object. Partition rings instead of
    # dropping their interiors, which would turn a courtyard into occupied PV.
    from .pv_field_service import without_holes
    return [{**{k: v for k, v in obj.items() if k != "geometry"},
             "polygon": list(part.exterior.coords)[:-1]}
            for obj in merged for part in without_holes(obj["geometry"])]
