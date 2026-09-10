"""Bounded access to Geneva's official roof-superstructure footprints.

SITG publishes this layer as 2-D projections of superstructures derived from
3-D building digitisation.  It is useful as a generic geometric exclusion for
roof-area calculations, but it is not treated as an exhaustive image-label
dataset.  The optional YOLO conversion therefore requires an explicit,
human-reviewed temporal/spatial alignment declaration and never writes an
empty annotation as a negative example.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import requests
from shapely.geometry import Polygon, box, shape
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid

SERVICE_URL = "https://vector.sitg.ge.ch/arcgis/rest/services/CAD_BATIMENT_HORSOL_TOIT_SP/FeatureServer"
LAYER_URL = f"{SERVICE_URL}/0"
QUERY_URL = f"{LAYER_URL}/query"
CATALOG_URL = "https://sitg.ge.ch/donnees/cad-batiment-horsol-toit-sp"
STDL_GROUND_TRUTH_URL = "https://tech.stdl.ch/PROJ-ROOFTOPS/#24-ground-truth"
LAYER_ID = 0
SPATIAL_REFERENCE = "EPSG:2056"
GENEVA_CLASS = "superstructure"
QUERY_FIELDS = ("OBJECTID", "EGID", "ALTITUDE_MIN", "ALTITUDE_MAX", "DATE_LEVE")
GENEVA_EXTENT_2056 = (2486335.8291, 1110440.8135, 2512590.2891, 1135438.9663)
MAX_BBOX_WIDTH_M = 5_000.0
MAX_BBOX_HEIGHT_M = 5_000.0
DEFAULT_PAGE_SIZE = 1_000
MAX_FEATURES = 20_000


class GenevaSourceError(RuntimeError):
    """Raised when the official service returns an unusable response."""


class GenevaAlignmentError(ValueError):
    """Raised when vector labels are used without explicit imagery alignment."""


@dataclass(frozen=True)
class GenevaSuperstructureResult:
    """GeoJSON response plus source provenance and pagination evidence."""

    geojson: dict[str, Any]
    provenance: dict[str, Any]
    pagination: dict[str, Any]
    cache_path: Path | None = None

    @property
    def features(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.geojson.get("features", ()))


@dataclass(frozen=True)
class GenevaImagery:
    """One georeferenced image frame eligible for opt-in label conversion."""

    image_id: str
    bounds2056: tuple[float, float, float, float]
    size_px: tuple[int, int]
    acquired_at: str | date | datetime
    crs_epsg: int = 2056

    def __post_init__(self) -> None:
        if not str(self.image_id).strip():
            raise ValueError("image_id must not be empty")
        bounds = _validate_bbox(self.bounds2056)
        object.__setattr__(self, "bounds2056", bounds)
        if len(self.size_px) != 2 or any(not isinstance(value, int) or value <= 0 for value in self.size_px):
            raise ValueError("size_px must contain two positive integers")
        if self.crs_epsg != 2056:
            raise ValueError("Geneva imagery must use CH1903+ / LV95 EPSG:2056")
        acquired = _normalise_date(self.acquired_at)
        if not acquired:
            raise ValueError("acquired_at must be explicit")
        object.__setattr__(self, "acquired_at", acquired)


@dataclass(frozen=True)
class GenevaLabelArtifact:
    """A single aligned annotation; empty artifacts are never negatives."""

    class_name: str
    lines: tuple[str, ...]
    source_feature_ids: tuple[Any, ...]
    imagery: dict[str, Any]
    alignment_status: str
    alignment_note: str


def _normalise_date(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value).strip()


def _validate_bbox(bbox2056: Sequence[float]) -> tuple[float, float, float, float]:
    if len(bbox2056) != 4:
        raise ValueError("bbox2056 must be (xmin, ymin, xmax, ymax)")
    try:
        bbox_values = tuple(float(value) for value in bbox2056)
    except (TypeError, ValueError) as error:
        raise ValueError("bbox2056 must contain finite numbers") from error
    if not all(math.isfinite(value) for value in bbox_values):
        raise ValueError("bbox2056 must contain finite numbers")
    xmin, ymin, xmax, ymax = bbox_values
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("bbox2056 must have positive width and height")
    if xmax - xmin > MAX_BBOX_WIDTH_M:
        raise ValueError(f"bbox2056 exceeds maximum width of {MAX_BBOX_WIDTH_M:g} m")
    if ymax - ymin > MAX_BBOX_HEIGHT_M:
        raise ValueError(f"bbox2056 exceeds maximum height of {MAX_BBOX_HEIGHT_M:g} m")
    gxmin, gymin, gxmax, gymax = GENEVA_EXTENT_2056
    if xmin < gxmin or ymin < gymin or xmax > gxmax or ymax > gymax:
        raise ValueError("bbox2056 lies outside the Geneva service extent")
    return bbox_values


def _format_query_coordinate(value: float) -> str:
    """Serialize an LV95 coordinate without significant-digit rounding."""

    serialized = repr(float(value))
    return serialized[:-2] if serialized.endswith(".0") else serialized


def _cache_key(bbox2056: tuple[float, float, float, float], page_size: int) -> str:
    value = json.dumps({"bbox2056": bbox2056, "page_size": page_size}, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _geojson_crs() -> dict[str, Any]:
    return {"type": "name", "properties": {"name": SPATIAL_REFERENCE}}


def _validate_feature_collection(document: Mapping[str, Any]) -> dict[str, Any]:
    if document.get("type") != "FeatureCollection" or not isinstance(document.get("features"), list):
        raise GenevaSourceError("SITG response is not a GeoJSON FeatureCollection")
    geojson = dict(document)
    geojson.setdefault("crs", _geojson_crs())
    crs = geojson.get("crs", {})
    crs_name = str(crs.get("properties", {}).get("name", "")) if isinstance(crs, Mapping) else ""
    if crs_name and crs_name.upper() not in {SPATIAL_REFERENCE, "URN:OGC:DEF:CRS:EPSG::2056"}:
        raise GenevaSourceError(f"SITG response has unexpected CRS: {crs_name}")
    for index, feature in enumerate(geojson["features"]):
        if not isinstance(feature, Mapping) or feature.get("type") != "Feature":
            raise GenevaSourceError(f"SITG feature {index} is malformed")
        geometry = feature.get("geometry")
        if not isinstance(geometry, Mapping):
            raise GenevaSourceError(f"SITG feature {index} has no geometry")
        try:
            candidate = shape(geometry)
        except Exception as error:
            raise GenevaSourceError(f"SITG feature {index} has invalid geometry") from error
        if candidate.is_empty or candidate.geom_type not in {"Polygon", "MultiPolygon"}:
            raise GenevaSourceError(f"SITG feature {index} has unsupported geometry")
    return geojson


def _result_from_document(
    document: Mapping[str, Any],
    cache_path: Path | None,
    *,
    from_cache: bool,
    expected_bbox2056: tuple[float, float, float, float] | None = None,
) -> GenevaSuperstructureResult:
    try:
        geojson = _validate_feature_collection(document["geojson"])
        provenance = dict(document["provenance"])
        pagination = dict(document["pagination"])
    except (KeyError, TypeError, GenevaSourceError) as error:
        raise GenevaSourceError("cached Geneva response is malformed") from error
    if expected_bbox2056 is not None:
        expected_query_geometry = ",".join(_format_query_coordinate(value) for value in expected_bbox2056)
        if provenance.get("queried_bbox2056") != list(expected_bbox2056) or provenance.get("query_geometry") != expected_query_geometry:
            raise GenevaSourceError("cached Geneva response lacks exact query-bound provenance")
    pagination["from_cache"] = from_cache
    return GenevaSuperstructureResult(geojson, provenance, pagination, cache_path)


def fetch_superstructures(
    bbox2056: Sequence[float],
    cache_dir: Path | None = None,
    *,
    session: requests.Session | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    timeout: float = 30.0,
    force_refresh: bool = False,
) -> GenevaSuperstructureResult:
    """Fetch a bounded LV95 bbox from SITG with deterministic pagination.

    The default 5 km side limit is deliberately conservative.  Responses are
    cached as an envelope containing the raw GeoJSON, query provenance, and
    page offsets; no imagery or unbounded canton-wide extraction is performed.
    """

    bbox = _validate_bbox(bbox2056)
    if not isinstance(page_size, int) or not 1 <= page_size <= 4_000:
        raise ValueError("page_size must be an integer in [1, 4000]")
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"geneva-superstructures-{_cache_key(bbox, page_size)}.json"
        if cache_path.is_file() and not force_refresh:
            try:
                return _result_from_document(
                    json.loads(cache_path.read_text(encoding="utf-8")),
                    cache_path,
                    from_cache=True,
                    expected_bbox2056=bbox,
                )
            except (OSError, json.JSONDecodeError, GenevaSourceError):
                cache_path.unlink(missing_ok=True)

    http = session or requests.Session()
    features: list[dict[str, Any]] = []
    offsets: list[int] = []
    offset = 0
    while True:
        query_geometry = ",".join(_format_query_coordinate(value) for value in bbox)
        params = {
            "where": "1=1",
            "geometry": query_geometry,
            "geometryType": "esriGeometryEnvelope",
            "inSR": 2056,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": ",".join(QUERY_FIELDS),
            "returnGeometry": "true",
            "returnZ": "false",
            "returnM": "false",
            "outSR": 2056,
            "f": "geojson",
            "resultType": "standard",
            "resultOffset": offset,
            "resultRecordCount": page_size,
            "orderByFields": "OBJECTID ASC",
        }
        try:
            response = http.get(QUERY_URL, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
        except Exception as error:
            if isinstance(error, GenevaSourceError):
                raise
            raise GenevaSourceError(f"SITG superstructure query failed at offset {offset}") from error
        if not isinstance(payload, Mapping):
            raise GenevaSourceError("SITG response was not a JSON object")
        page = _validate_feature_collection(payload)
        page_features = page["features"]
        offsets.append(offset)
        features.extend(page_features)
        if len(features) > MAX_FEATURES:
            raise GenevaSourceError(f"bbox response exceeds safety cap of {MAX_FEATURES} features")
        exceeded = bool(payload.get("exceededTransferLimit", False))
        if not page_features or (not exceeded and len(page_features) < page_size):
            break
        offset += len(page_features)

    geojson: dict[str, Any] = {
        "type": "FeatureCollection",
        "crs": _geojson_crs(),
        "features": features,
    }
    provenance = {
        "provider": "SITG / Département du territoire, Genève",
        "catalog_url": CATALOG_URL,
        "service_url": SERVICE_URL,
        "layer_url": LAYER_URL,
        "layer_id": LAYER_ID,
        "service_item_id": "c40c5bf33bc64dde9ea857cfdc2ff617",
        "spatial_reference": SPATIAL_REFERENCE,
        "bbox2056": list(bbox),
        "queried_bbox2056": list(bbox),
        "query_geometry": query_geometry,
        "retrieved_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "fields": list(QUERY_FIELDS),
        "update_cadence": "weekly (catalog metadata)",
        "license_note": "SITG catalog lists open access; use current conditions before redistribution.",
        "geometry_note": "2-D roof-superstructure footprints projected from 3-D building digitisation.",
        "identifier_note": "OBJECTID is not a permanent unique identifier; treat it as query output only.",
        "date_alignment_note": "DATE_LEVE is a source attribute; no alignment with imagery is inferred.",
    }
    pagination = {
        "page_size": page_size,
        "pages": len(offsets),
        "offsets": offsets,
        "records": len(features),
        "max_features": MAX_FEATURES,
        "from_cache": False,
    }
    result = GenevaSuperstructureResult(geojson, provenance, pagination, cache_path)
    if cache_path is not None:
        envelope = {"geojson": geojson, "provenance": provenance, "pagination": pagination}
        cache_path.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")
    return result


def superstructure_exclusions(source: GenevaSuperstructureResult | Mapping[str, Any]) -> tuple[BaseGeometry, ...]:
    """Return generic metric geometries suitable for ``calculate_usable_area``."""

    geojson = source.geojson if isinstance(source, GenevaSuperstructureResult) else source
    document = _validate_feature_collection(geojson)
    geometries: list[BaseGeometry] = []
    for feature in document["features"]:
        geometry = make_valid(shape(feature["geometry"]))
        if not geometry.is_empty:
            geometries.append(geometry)
    return tuple(geometries)


def _feature_id(feature: Mapping[str, Any], index: int) -> Any:
    properties = feature.get("properties")
    if isinstance(properties, Mapping) and properties.get("OBJECTID") is not None:
        return properties["OBJECTID"]
    if feature.get("id") is not None:
        return feature["id"]
    return index


def _polygon_parts(geometry: BaseGeometry) -> tuple[Polygon, ...]:
    geometry = make_valid(geometry)
    if geometry.is_empty:
        return ()
    if isinstance(geometry, Polygon):
        return (geometry,)
    parts: list[Polygon] = []
    if hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            parts.extend(_polygon_parts(part))
    return tuple(parts)


def geojson_to_yolo(
    source: GenevaSuperstructureResult | Mapping[str, Any],
    imagery: GenevaImagery,
    *,
    alignment_status: str = "unverified",
    alignment_note: str = "",
) -> GenevaLabelArtifact:
    """Convert one explicitly aligned image's intersecting footprints to YOLO.

    ``alignment_status='verified'`` is intentionally mandatory.  SITG's
    weekly layer is not an exhaustive image annotation, and Geneva's STDL
    ground truth specifically used synchronized 2019 true orthophotos and
    vectorized labels; a generic SwissImage tile must not be silently mixed in.
    """

    if alignment_status != "verified" or not alignment_note.strip():
        raise GenevaAlignmentError(
            "Geneva labels require alignment_status='verified' and a non-empty alignment_note"
        )
    geojson = source.geojson if isinstance(source, GenevaSuperstructureResult) else source
    document = _validate_feature_collection(geojson)
    xmin, ymin, xmax, ymax = imagery.bounds2056
    image_width, image_height = imagery.size_px
    image_box = box(xmin, ymin, xmax, ymax)
    lines: list[str] = []
    ids: list[Any] = []
    for index, feature in enumerate(document["features"]):
        geometry = make_valid(shape(feature["geometry"]))
        clipped = geometry.intersection(image_box)
        feature_lines: list[str] = []
        for polygon in _polygon_parts(clipped):
            points = list(polygon.exterior.coords)[:-1]
            if len(points) < 3:
                continue
            coordinates: list[float] = []
            for x, y in points:
                px = min(1.0, max(0.0, (float(x) - xmin) / (xmax - xmin)))
                py = min(1.0, max(0.0, (ymax - float(y)) / (ymax - ymin)))
                coordinates.extend((px, py))
            feature_lines.append("0 " + " ".join(f"{value:.6f}" for value in coordinates))
        if feature_lines:
            lines.extend(feature_lines)
            ids.append(_feature_id(feature, index))

    metadata = {
        "class_name": GENEVA_CLASS,
        "alignment_status": alignment_status,
        "alignment_note": alignment_note,
        "source_feature_ids": ids,
        "imagery": {
            "image_id": imagery.image_id,
            "bounds2056": list(imagery.bounds2056),
            "size_px": list(imagery.size_px),
            "acquired_at": imagery.acquired_at,
            "crs": f"EPSG:{imagery.crs_epsg}",
        },
        "negative_policy": "empty result is not written as a negative; source layer is not exhaustive",
    }
    return GenevaLabelArtifact(GENEVA_CLASS, tuple(lines), tuple(ids), metadata["imagery"], alignment_status, alignment_note)


def write_yolo_annotation(artifact: GenevaLabelArtifact, label_path: Path) -> Path | None:
    """Write one non-empty annotation and a sidecar; never materialize negatives."""

    label_path = Path(label_path)
    if not artifact.lines:
        return None
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(artifact.lines) + "\n", encoding="utf-8")
    metadata_path = label_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "class_name": artifact.class_name,
                "alignment_status": artifact.alignment_status,
                "alignment_note": artifact.alignment_note,
                "source_feature_ids": list(artifact.source_feature_ids),
                "imagery": artifact.imagery,
                "negative_policy": "empty result is not written as a negative; source layer is not exhaustive",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return metadata_path


__all__ = [
    "CATALOG_URL",
    "GENEVA_CLASS",
    "GENEVA_EXTENT_2056",
    "GenevaAlignmentError",
    "GenevaImagery",
    "GenevaLabelArtifact",
    "GenevaSourceError",
    "GenevaSuperstructureResult",
    "LAYER_URL",
    "QUERY_URL",
    "SERVICE_URL",
    "STDL_GROUND_TRUTH_URL",
    "fetch_superstructures",
    "geojson_to_yolo",
    "superstructure_exclusions",
    "write_yolo_annotation",
]
