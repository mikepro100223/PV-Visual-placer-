"""Bounded live connectors for Swiss rooftop-PV inputs.

The module deliberately keeps source data separate from modelling.  It talks
to the official GeoAdmin/Sonnendach, swisstopo SWISSIMAGE and EU-JRC PVGIS
services, validates the small response contracts we depend on, and raises an
explicit :class:`SourceError` instead of manufacturing a fallback value.

All requests use a finite connect/read timeout and at most two retries.  The
orthophoto connector additionally caps both the requested pixel dimensions and
ground extent.  PVGIS responses are cached as an envelope containing the raw
response, request parameters, and retrieval time under ``data/cache`` by
default so that provenance survives a later offline run.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Sequence

# Public endpoints, kept as constants both for provenance and to make them
# easy to replace in tests without changing connector behaviour.
GEOADMIN_SEARCH_URL = "https://api3.geo.admin.ch/rest/services/ech/SearchServer"
GEOADMIN_IDENTIFY_URL = "https://api3.geo.admin.ch/rest/services/ech/MapServer/identify"
GEOADMIN_FEATURE_URL = "https://api3.geo.admin.ch/rest/services/ech/MapServer"
GEOADMIN_FIND_URL = "https://api3.geo.admin.ch/rest/services/ech/MapServer/find"
ORTHOPHOTO_WMS_URL = "https://wms.geo.admin.ch/"
ORTHOPHOTO_CACHE_URL = (
    "https://api3.geo.admin.ch/rest/services/ech/MapServer/"
    "ch.swisstopo.swissimage-product/cacheUpdate"
)
PVGIS_BASE_URL = "https://re.jrc.ec.europa.eu/api/v5_3"
PVGIS_TMY_URL = f"{PVGIS_BASE_URL}/tmy"
PVGIS_HORIZON_URL = f"{PVGIS_BASE_URL}/printhorizon"

ROOF_LAYER = "ch.bfe.solarenergie-eignung-daecher"
ORTHOPHOTO_LAYER = "ch.swisstopo.swissimage-product"
HTTP_TIMEOUT = (5.0, 30.0)
MAX_RETRIES = 2
MAX_IMAGE_WIDTH = 2048
MAX_IMAGE_HEIGHT = 2048
MAX_IMAGE_PIXELS = 4_000_000
MAX_BBOX_SPAN_M = 5_000.0
# GeoAdmin identify documents a hard per-request ceiling of 200 features.
# Bbox roof retrieval has a smaller spatial and response budget so a caller
# cannot accidentally turn one image request into an unbounded national query.
ROOF_IDENTIFY_PAGE_LIMIT = 200
MAX_ROOF_BBOX_SPAN_M = 500.0
MAX_ROOF_BBOX_FEATURES = 20_000
MAX_ROOF_BBOX_PAGES = 100


class SourceError(RuntimeError):
    """Raised when an upstream source is unavailable or violates its contract."""


class HTTPError(SourceError):
    """Small requests-compatible HTTP error used in tests and error messages."""

    def __init__(self, status_code: int, url: str):
        self.status_code = status_code
        self.url = url
        super().__init__(f"HTTP {status_code} from {url}")


@dataclass(frozen=True)
class GeocodeResult:
    """One GeoAdmin location search result in WGS84 and LV95."""

    query: str
    label: str
    latitude: float
    longitude: float
    easting: float
    northing: float
    feature_id: str | None
    source_url: str
    source_fetched_at: str


@dataclass(frozen=True)
class RoofFeature:
    """A complete Sonnendach roof polygon and its official attributes.

    ``AUSRICHTUNG`` follows the Sonnendach convention, not the common compass
    convention used by many PV libraries: north=-180/180, east=-90, south=0,
    west=+90.  ``FLAECHE`` is the physical, sloped roof area in square metres.
    """

    feature_id: int | str
    geometry: Mapping[str, Any]
    properties: Mapping[str, Any]
    source_url: str
    source_data_date: str | None
    source_fetched_at: str

    @property
    def area_m2(self) -> float:
        """Official physical roof-plane area (``FLAECHE``) in m²."""

        return _as_float(self.properties.get("flaeche"), "roof area")

    @property
    def tilt_deg(self) -> float:
        """Official ``NEIGUNG`` angle to the horizontal in degrees."""

        value = _as_float(self.properties.get("neigung"), "roof tilt")
        if not 0 <= value <= 90:
            raise SourceError(f"Invalid Sonnendach NEIGUNG: {value}")
        return value

    @property
    def azimuth_deg(self) -> float:
        """PV-library azimuth (north=0, east=90, south=180, west=270).

        Sonnendach stores degrees from south with north at -180/180.  The
        conversion is ``(AUSRICHTUNG + 180) % 360``.
        """

        value = _as_float(self.properties.get("ausrichtung"), "roof orientation")
        if not -180 <= value <= 180:
            raise SourceError(f"Invalid Sonnendach AUSRICHTUNG: {value}")
        return (value + 180.0) % 360.0


@dataclass(frozen=True)
class OrthophotoResult:
    """North-up image bytes returned by the SWISSIMAGE WMS."""

    image_bytes: bytes
    content_type: str
    bbox2056: tuple[float, float, float, float]
    width: int
    height: int
    gsd_m: tuple[float, float]
    north_up: bool
    source_url: str
    source_data_date: str | None
    source_fetched_at: str

    def open_pil(self):
        """Decode the image lazily, keeping Pillow optional for metadata users."""

        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - dependency is in app env
            raise SourceError("Pillow is required to decode orthophoto bytes") from exc
        return Image.open(BytesIO(self.image_bytes))


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _requests():
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - app installs requests
        raise SourceError("requests is required for live source connectors") from exc
    return requests


def _default_session():
    return _requests().Session()


def _response_url(response: Any, fallback: str) -> str:
    return str(getattr(response, "url", "") or fallback)


def _http_get(session: Any, url: str, params: Mapping[str, Any], *, as_json: bool = True) -> tuple[Any, Any]:
    """Perform one bounded GET with two bounded retries.

    Returning the response and decoded payload separately keeps byte/image
    handling from accidentally passing an HTML error page to a decoder.
    """

    requests = _requests()
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = session.get(url, params=dict(params), timeout=HTTP_TIMEOUT)
            status = int(getattr(response, "status_code", 200))
            if status >= 400:
                raise HTTPError(status, _response_url(response, url))
            if as_json:
                try:
                    return response, response.json()
                except (ValueError, TypeError) as exc:
                    raise SourceError(
                        f"Invalid JSON from {_response_url(response, url)}"
                    ) from exc
            return response, None
        except HTTPError as exc:
            last_error = exc
            retryable = exc.status_code >= 500
        except requests.exceptions.RequestException as exc:
            last_error = exc
            retryable = True
        except SourceError:
            raise
        except Exception as exc:
            # Fake sessions and alternate requests-compatible transports may
            # not expose requests' exception hierarchy.  Preserve a clear
            # source error while still bounding transient attempts.
            last_error = exc
            retryable = True
        if not retryable or attempt >= MAX_RETRIES:
            break
        time.sleep(0.25 * (attempt + 1))
    assert last_error is not None
    if isinstance(last_error, SourceError):
        raise last_error
    raise SourceError(f"Request failed for {url}: {last_error}") from last_error


def _as_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SourceError(f"Invalid {name}: {value!r}") from exc
    if not math.isfinite(number):
        raise SourceError(f"Invalid non-finite {name}: {value!r}")
    return number


def _validate_wgs84(latitude: float, longitude: float) -> tuple[float, float]:
    lat = _as_float(latitude, "latitude")
    lon = _as_float(longitude, "longitude")
    if not -90 <= lat <= 90:
        raise ValueError("latitude must be between -90 and 90")
    if not -180 <= lon <= 180:
        raise ValueError("longitude must be between -180 and 180")
    return lat, lon


def _wgs84_to_lv95(latitude: float, longitude: float) -> tuple[float, float]:
    try:
        from pyproj import Transformer
    except ImportError as exc:  # pragma: no cover - app installs pyproj
        raise SourceError("pyproj is required for WGS84/LV95 conversion") from exc
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    easting, northing = transformer.transform(longitude, latitude)
    return _as_float(easting, "LV95 easting"), _as_float(northing, "LV95 northing")


def _source_date(properties: Mapping[str, Any]) -> str | None:
    for key in ("datum_aenderung", "datum_erstellung", "gs_serie_start", "sb_datum_aenderung"):
        value = properties.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def geocode(query: str, *, session: Any | None = None) -> GeocodeResult:
    """Resolve a Swiss address/location with GeoAdmin SearchServer.

    The request explicitly asks for ``sr=4326`` because SearchServer's
    GeoJSON axis order becomes surprising in LV95.  We retain the authoritative
    WGS84 result and derive LV95 using a standards-aware transformer.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if len(query) > 300:
        raise ValueError("query is too long (maximum 300 characters)")
    session = session or _default_session()
    params = {
        "searchText": query.strip(),
        "type": "locations",
        "geometryFormat": "geojson",
        "sr": 4326,
        "lang": "en",
    }
    response, payload = _http_get(session, GEOADMIN_SEARCH_URL, params)
    features = payload.get("features") if isinstance(payload, Mapping) else None
    if not isinstance(features, list) or not features:
        raise SourceError(f"No geocoding result for {query!r}")
    feature = features[0]
    if not isinstance(feature, Mapping):
        raise SourceError("GeoAdmin returned a malformed location feature")
    properties = feature.get("properties") if isinstance(feature.get("properties"), Mapping) else {}
    geometry = feature.get("geometry") if isinstance(feature.get("geometry"), Mapping) else {}
    coordinates = geometry.get("coordinates")
    lon = properties.get("lon")
    lat = properties.get("lat")
    if lat is None and isinstance(coordinates, Sequence) and len(coordinates) >= 2:
        lon, lat = coordinates[0], coordinates[1]
    lat, lon = _validate_wgs84(lat, lon)
    easting, northing = _wgs84_to_lv95(lat, lon)
    label = str(properties.get("label") or properties.get("detail") or query.strip())
    return GeocodeResult(
        query=query.strip(),
        label=label,
        latitude=lat,
        longitude=lon,
        easting=easting,
        northing=northing,
        feature_id=(str(properties["featureId"]) if properties.get("featureId") is not None else None),
        source_url=_response_url(response, GEOADMIN_SEARCH_URL),
        source_fetched_at=_utc_now(),
    )


def fetch_roofs(
    latitude: float,
    longitude: float,
    *,
    tolerance_m: float = 10.0,
    limit: int = 10,
    session: Any | None = None,
) -> list[RoofFeature]:
    """Fetch nearby official Sonnendach roof polygons and attributes.

    GeoAdmin ``identify`` finds roof feature IDs at a point.  Each ID is then
    fetched through the feature endpoint with ``returnGeometry=true`` so the
    returned geometry is the complete polygon rather than a screen-pixel
    approximation.  The result is allowed to be empty when no roof is present.
    """

    lat, lon = _validate_wgs84(latitude, longitude)
    tolerance = _as_float(tolerance_m, "tolerance_m")
    if tolerance < 0 or tolerance > 100:
        raise ValueError("tolerance_m must be between 0 and 100")
    if not isinstance(limit, int) or not 1 <= limit <= 20:
        raise ValueError("limit must be an integer between 1 and 20")
    easting, northing = _wgs84_to_lv95(lat, lon)
    pad = max(20.0, tolerance * 3.0)
    map_extent = f"{easting - pad:.3f},{northing - pad:.3f},{easting + pad:.3f},{northing + pad:.3f}"
    session = session or _default_session()
    identify_params = {
        "geometry": f"{easting:.3f},{northing:.3f}",
        "geometryFormat": "geojson",
        "geometryType": "esriGeometryPoint",
        "imageDisplay": "256,256,96",
        "lang": "en",
        "layers": f"all:{ROOF_LAYER}",
        "limit": limit,
        "mapExtent": map_extent,
        "returnGeometry": False,
        "sr": 2056,
        # GeoAdmin identify's tolerance is in screen pixels, not metres. The
        # map extent is rendered as 256 px, so convert the requested LV95
        # radius against the actual metres-per-pixel of this identify view.
        "tolerance": int(round(tolerance * 256.0 / (2.0 * pad))),
    }
    identify_response, identify_payload = _http_get(session, GEOADMIN_IDENTIFY_URL, identify_params)
    results = identify_payload.get("results") if isinstance(identify_payload, Mapping) else None
    if not isinstance(results, list):
        raise SourceError("GeoAdmin identify response has no results list")

    roofs: list[RoofFeature] = []
    seen_ids: set[str] = set()
    for result in results[:limit]:
        if not isinstance(result, Mapping):
            continue
        feature_id = result.get("featureId", result.get("id"))
        if feature_id is None or str(feature_id) in seen_ids:
            continue
        seen_ids.add(str(feature_id))
        feature_url = f"{GEOADMIN_FEATURE_URL}/{ROOF_LAYER}/{feature_id}"
        feature_params = {
            "returnGeometry": True,
            "geometryFormat": "geojson",
            "sr": 2056,
            "lang": "en",
        }
        feature_response, feature_payload = _http_get(session, feature_url, feature_params)
        feature = feature_payload.get("feature") if isinstance(feature_payload, Mapping) else None
        if not isinstance(feature, Mapping):
            raise SourceError(f"Sonnendach feature {feature_id} has no feature object")
        geometry = feature.get("geometry")
        if not isinstance(geometry, Mapping) or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            raise SourceError(f"Sonnendach feature {feature_id} has no polygon geometry")
        identify_properties = result.get("properties") if isinstance(result.get("properties"), Mapping) else {}
        feature_properties = feature.get("properties") if isinstance(feature.get("properties"), Mapping) else {}
        properties = dict(identify_properties)
        properties.update(feature_properties)
        roofs.append(
            RoofFeature(
                feature_id=feature.get("featureId", feature_id),
                geometry=geometry,
                properties=properties,
                source_url=_response_url(feature_response, feature_url),
                source_data_date=_source_date(properties),
                source_fetched_at=_utc_now(),
            )
        )
    return roofs


def _roof_feature_from_result(result: Mapping[str, Any], *, source_url: str) -> RoofFeature:
    """Validate one GeoJSON roof feature returned by GeoAdmin identify."""

    feature_id = result.get("featureId")
    if feature_id is None:
        feature_id = result.get("id")
    if feature_id is None:
        raise SourceError("Sonnendach bbox result has no feature ID")
    geometry = result.get("geometry")
    coordinates = geometry.get("coordinates") if isinstance(geometry, Mapping) else None
    if (
        not isinstance(geometry, Mapping)
        or geometry.get("type") not in {"Polygon", "MultiPolygon"}
        or not isinstance(coordinates, Sequence)
        or isinstance(coordinates, (str, bytes))
        or not coordinates
    ):
        raise SourceError(f"Sonnendach feature {feature_id} has no polygon geometry")
    properties = result.get("properties")
    if not isinstance(properties, Mapping):
        properties = {}
    roof = RoofFeature(
        feature_id=feature_id,
        geometry=geometry,
        properties=dict(properties),
        source_url=source_url,
        source_data_date=_source_date(properties),
        source_fetched_at=_utc_now(),
    )
    # Validate the physical attributes at the source boundary.  Otherwise a
    # malformed result would survive until area/physics calculation and could
    # be mistaken for a valid roof with missing metadata.
    roof.area_m2
    roof.tilt_deg
    roof.azimuth_deg
    return roof


def _roof_feature_sort_key(roof: RoofFeature) -> tuple[int, int | str]:
    """Sort numeric GeoAdmin IDs numerically and non-numeric IDs lexically."""

    text = str(roof.feature_id)
    try:
        return (0, int(text))
    except (TypeError, ValueError):
        return (1, text)


def fetch_roofs_bbox(
    bbox2056: Sequence[float],
    *,
    session: Any | None = None,
    max_features: int = MAX_ROOF_BBOX_FEATURES,
    max_pages: int = MAX_ROOF_BBOX_PAGES,
) -> list[RoofFeature]:
    """Fetch every Sonnendach roof intersecting a bounded LV95 bbox.

    GeoAdmin's identify endpoint accepts an ``esriGeometryEnvelope`` and
    paginates at most 200 results with ``offset``.  This connector follows
    those documented pages, de-duplicates feature IDs, and sorts the complete
    result deterministically.  It raises :class:`SourceError` if the upstream
    response or an explicit feature/page budget indicates that the inventory
    is incomplete; it never silently returns a truncated bbox result.

    The default 500 m maximum side and 20,000-feature/100-page budgets are
    deliberately finite.  Small image chips can therefore contain more than
    twenty roof planes while a broad accidental query fails explicitly.
    """

    bbox = _validate_bbox(bbox2056)
    if max(bbox[2] - bbox[0], bbox[3] - bbox[1]) > MAX_ROOF_BBOX_SPAN_M:
        raise ValueError(f"roof bbox span exceeds bounded limit of {MAX_ROOF_BBOX_SPAN_M:g} m")
    if isinstance(max_features, bool) or not isinstance(max_features, int) or not 1 <= max_features <= MAX_ROOF_BBOX_FEATURES:
        raise ValueError(f"max_features must be an integer between 1 and {MAX_ROOF_BBOX_FEATURES}")
    if isinstance(max_pages, bool) or not isinstance(max_pages, int) or not 1 <= max_pages <= MAX_ROOF_BBOX_PAGES:
        raise ValueError(f"max_pages must be an integer between 1 and {MAX_ROOF_BBOX_PAGES}")

    session = session or _default_session()
    map_extent = ",".join(f"{value:.3f}" for value in bbox)
    page_limit = min(ROOF_IDENTIFY_PAGE_LIMIT, max_features)
    roofs_by_id: dict[str, RoofFeature] = {}
    page_signatures: set[tuple[str, ...]] = set()
    offset = 0

    for page_number in range(max_pages):
        params = {
            "geometry": map_extent,
            "geometryType": "esriGeometryEnvelope",
            "layers": f"all:{ROOF_LAYER}",
            "mapExtent": map_extent,
            "imageDisplay": "256,256,96",
            "tolerance": 0,
            "returnGeometry": True,
            "geometryFormat": "geojson",
            "sr": 2056,
            "lang": "en",
            "limit": page_limit,
            "offset": offset,
        }
        response, payload = _http_get(session, GEOADMIN_IDENTIFY_URL, params)
        results = payload.get("results") if isinstance(payload, Mapping) else None
        if not isinstance(results, list):
            raise SourceError("GeoAdmin bbox identify response has no results list")
        if len(results) > page_limit:
            raise SourceError(
                f"GeoAdmin bbox identify returned {len(results)} results despite limit={page_limit}"
            )
        if not results:
            break

        source_url = _response_url(response, GEOADMIN_IDENTIFY_URL)
        page_ids: list[str] = []
        for result in results:
            if not isinstance(result, Mapping):
                raise SourceError("GeoAdmin bbox identify returned a malformed feature")
            roof = _roof_feature_from_result(result, source_url=source_url)
            feature_key = str(roof.feature_id)
            page_ids.append(feature_key)
            if feature_key not in roofs_by_id:
                if len(roofs_by_id) >= max_features:
                    raise SourceError(
                        f"GeoAdmin bbox result exceeds max_features={max_features}; "
                        "subdivide the bbox or raise the explicit cap"
                    )
                roofs_by_id[feature_key] = roof

        page_signature = tuple(page_ids)
        if page_signature in page_signatures:
            raise SourceError("GeoAdmin bbox pagination did not advance; refusing incomplete results")
        page_signatures.add(page_signature)
        if len(results) < page_limit:
            break
        offset += len(results)
    else:
        raise SourceError(
            f"GeoAdmin bbox pagination reached max_pages={max_pages}; "
            "subdivide the bbox or raise the explicit cap"
        )

    return sorted(roofs_by_id.values(), key=_roof_feature_sort_key)


def _validate_bbox(bbox2056: Sequence[float]) -> tuple[float, float, float, float]:
    if not isinstance(bbox2056, Sequence) or isinstance(bbox2056, (str, bytes)) or len(bbox2056) != 4:
        raise ValueError("bbox2056 must be a four-value (minx, miny, maxx, maxy) sequence")
    values = tuple(_as_float(value, "bbox coordinate") for value in bbox2056)
    minx, miny, maxx, maxy = values
    if not minx < maxx or not miny < maxy:
        raise ValueError("bbox2056 must have min values smaller than max values")
    if maxx - minx > MAX_BBOX_SPAN_M or maxy - miny > MAX_BBOX_SPAN_M:
        raise ValueError(f"bbox span exceeds bounded limit of {MAX_BBOX_SPAN_M:g} m")
    return values


def fetch_orthophoto(
    bbox2056: Sequence[float],
    width: int = 1024,
    height: int = 1024,
    *,
    session: Any | None = None,
) -> OrthophotoResult:
    """Download a finite, north-up LV95 SWISSIMAGE WMS image.

    WMS 1.3.0's EPSG:2056 BBOX is ``minx,miny,maxx,maxy``.  The returned image
    is north-up (row 0 is the max-y/north edge), and ``gsd_m`` reports the exact
    requested ground spacing in x and y metres per pixel.
    """

    bbox = _validate_bbox(bbox2056)
    if not isinstance(width, int) or not 1 <= width <= MAX_IMAGE_WIDTH:
        raise ValueError(f"width must be an integer between 1 and {MAX_IMAGE_WIDTH}")
    if not isinstance(height, int) or not 1 <= height <= MAX_IMAGE_HEIGHT:
        raise ValueError(f"height must be an integer between 1 and {MAX_IMAGE_HEIGHT}")
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(f"image pixel count exceeds bounded limit of {MAX_IMAGE_PIXELS}")
    minx, miny, maxx, maxy = bbox
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": ORTHOPHOTO_LAYER,
        "STYLES": "default",
        "CRS": "EPSG:2056",
        "BBOX": ",".join(f"{value:.3f}" for value in bbox),
        "WIDTH": width,
        "HEIGHT": height,
        "FORMAT": "image/jpeg",
        "TRANSPARENT": "false",
    }
    session = session or _default_session()
    response, _ = _http_get(session, ORTHOPHOTO_WMS_URL, params, as_json=False)
    content = bytes(getattr(response, "content", b""))
    content_type = str(getattr(response, "headers", {}).get("Content-Type", "image/jpeg")).split(";", 1)[0].lower()
    if not content:
        raise SourceError(f"SWISSIMAGE returned an empty image from {_response_url(response, ORTHOPHOTO_WMS_URL)}")
    if not content_type.startswith("image/") and not content.startswith((b"\xff\xd8", b"\x89PNG")):
        raise SourceError(f"SWISSIMAGE returned non-image content type {content_type!r}")

    source_data_date: str | None = None
    try:
        cache_response, cache_payload = _http_get(session, ORTHOPHOTO_CACHE_URL, {}, as_json=True)
        if isinstance(cache_payload, Mapping) and cache_payload.get("cache_update"):
            source_data_date = str(cache_payload["cache_update"])
    except SourceError:
        # The image itself is still authoritative; cache-update is a best
        # effort provenance hint and not a reason to discard usable imagery.
        source_data_date = None

    return OrthophotoResult(
        image_bytes=content,
        content_type=content_type,
        bbox2056=bbox,
        width=width,
        height=height,
        gsd_m=((maxx - minx) / width, (maxy - miny) / height),
        north_up=True,
        source_url=_response_url(response, ORTHOPHOTO_WMS_URL),
        source_data_date=source_data_date,
        source_fetched_at=_utc_now(),
    )


def _cache_key(params: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(params), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - app installs pandas
        raise SourceError("pandas is required for the PVGIS weather DataFrame") from exc
    return pd


def _weather_dataframe(payload: Mapping[str, Any]):
    pd = _pandas()
    outputs = payload.get("outputs")
    rows = outputs.get("tmy_hourly") if isinstance(outputs, Mapping) else None
    if not isinstance(rows, list) or not rows or not all(isinstance(row, Mapping) for row in rows):
        raise SourceError("PVGIS response has no usable outputs.tmy_hourly rows")
    frame = pd.DataFrame(rows)
    time_column = next((name for name in ("time(UTC)", "time") if name in frame.columns), None)
    if time_column is None:
        raise SourceError("PVGIS tmy_hourly response has no UTC time column")
    try:
        source_times = pd.to_datetime(
            frame[time_column].astype(str), format="%Y%m%d:%H%M", utc=True, errors="raise"
        )
    except (TypeError, ValueError) as exc:
        raise SourceError("PVGIS tmy_hourly contains an invalid UTC time") from exc

    # PVGIS selects a representative source year independently for each month.
    # Replacing those years with one non-leap reference year makes the frame a
    # continuous simulation year while retaining the original source timestamp
    # for auditability.  A February 29 would not have a valid non-leap mapping;
    # fail closed rather than silently dropping a weather observation.
    if ((source_times.dt.month == 2) & (source_times.dt.day == 29)).any():
        raise SourceError("PVGIS TMY contains February 29 and cannot map to a non-leap reference year")
    reference_times = pd.DatetimeIndex(pd.to_datetime(
        {
            "year": 2001,
            "month": source_times.dt.month.to_numpy(),
            "day": source_times.dt.day.to_numpy(),
            "hour": source_times.dt.hour.to_numpy(),
            "minute": source_times.dt.minute.to_numpy(),
            "second": source_times.dt.second.to_numpy(),
        },
        utc=True,
    ))
    inputs = payload.get("inputs") if isinstance(payload, Mapping) else {}
    location = inputs.get("location") if isinstance(inputs, Mapping) else {}
    try:
        irradiance_offset_hours = float(
            location.get("irradiance_time_offset", 0.0) if isinstance(location, Mapping) else 0.0
        )
    except (TypeError, ValueError) as exc:
        raise SourceError("PVGIS irradiance_time_offset is invalid") from exc
    if not math.isfinite(irradiance_offset_hours) or not -24 <= irradiance_offset_hours <= 24:
        raise SourceError("PVGIS irradiance_time_offset must be finite and within +/-24 hours")
    # PVGIS timestamps are labelled HH:00 while SARAH/ERA5 irradiance may
    # correspond to an offset within that hour. Shift the reference index so
    # pvlib's solar-position calculation uses the actual irradiance instant.
    if irradiance_offset_hours:
        reference_times = reference_times + pd.to_timedelta(irradiance_offset_hours, unit="h")
    if reference_times.has_duplicates or not reference_times.is_monotonic_increasing:
        raise SourceError("PVGIS TMY timestamps do not form one increasing reference year")
    frame["source_timestamp_utc"] = source_times
    normalized = {
        "G(h)": "ghi",
        "Gb(n)": "dni",
        "Gd(h)": "dhi",
        "T2m": "temp_air",
        "WS10m": "wind_speed",
    }
    # Preserve the exact PVGIS field names and add stable modelling aliases.
    for source, target in normalized.items():
        if source in frame.columns and target not in frame.columns:
            frame[target] = frame[source]
    required = {"ghi", "dni", "dhi", "temp_air", "wind_speed"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SourceError(f"PVGIS tmy_hourly is missing normalized fields: {', '.join(missing)}")
    clipped_counts: dict[str, int] = {}
    for column in required:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not values.map(math.isfinite).all():
            raise SourceError(f"PVGIS tmy_hourly field {column!r} contains non-finite values")
        # PVGIS may contain tiny negative wind-speed artefacts (for example
        # -0.08 m/s).  Clamp only this physically bounded noise in the
        # modelling alias; the untouched WS10m source field remains available.
        if column in {"ghi", "dni", "dhi", "wind_speed"}:
            clipped_counts[column] = int((values < 0).sum())
            values = values.clip(lower=0)
        frame[column] = values
    frame.index = reference_times
    frame.index.name = "timestamp_utc"
    frame.attrs["irradiance_time_offset_hours"] = irradiance_offset_hours
    frame.attrs["clipped_non_negative_counts"] = clipped_counts
    return frame


def fetch_weather(
    latitude: float,
    longitude: float,
    *,
    cache_dir: str | os.PathLike[str] = "data/cache",
    refresh: bool = False,
    session: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Return a PVGIS 5.3 Typical Meteorological Year DataFrame and metadata.

    The frame retains PVGIS field names (``G(h)``, ``T2m`` etc.) and adds a
    timezone-aware index named ``timestamp_utc`` in the reference year; the
    ``source_timestamp_utc`` column retains original source-year timestamps.
    Metadata includes PVGIS source
    dataset/period, the exact source URL and the raw-response cache path.
    """

    lat, lon = _validate_wgs84(latitude, longitude)
    params: dict[str, Any] = {
        "lat": f"{lat:.6f}",
        "lon": f"{lon:.6f}",
        "outputformat": "json",
        "browser": 1,
        "usehorizon": 1,
    }
    cache_path = Path(cache_dir) / f"pvgis_tmy_v5_3_{_cache_key(params)}.json"
    cache_hit = False
    envelope: Mapping[str, Any]
    if cache_path.exists() and not refresh:
        try:
            envelope = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SourceError(f"Cannot read PVGIS cache {cache_path}") from exc
        if not isinstance(envelope, Mapping) or not isinstance(envelope.get("payload"), Mapping):
            raise SourceError(f"PVGIS cache {cache_path} is not a connector cache envelope")
        payload = envelope["payload"]
        cache_hit = True
        source_url = str(envelope.get("source_url") or PVGIS_TMY_URL)
        retrieved_at = str(envelope.get("retrieved_at") or "")
    else:
        session = session or _default_session()
        response, payload = _http_get(session, PVGIS_TMY_URL, params)
        if not isinstance(payload, Mapping):
            raise SourceError("PVGIS response is not a JSON object")
        # Validate before writing so a transient upstream HTML/shape change
        # cannot poison the otherwise reusable local cache.
        _weather_dataframe(payload)
        source_url = _response_url(response, PVGIS_TMY_URL)
        retrieved_at = _utc_now()
        envelope_to_write = {
            "connector": "rooftop_pv.sources.fetch_weather",
            "source": "PVGIS 5.3 TMY",
            "source_url": source_url,
            "request_params": params,
            "retrieved_at": retrieved_at,
            "payload": payload,
        }
        try:
            _atomic_write_json(cache_path, envelope_to_write)
        except OSError as exc:
            raise SourceError(f"Cannot write PVGIS cache {cache_path}") from exc
        envelope = envelope_to_write

    frame = _weather_dataframe(payload)
    inputs = payload.get("inputs") if isinstance(payload, Mapping) else {}
    location = inputs.get("location") if isinstance(inputs, Mapping) else {}
    meteo = inputs.get("meteo_data") if isinstance(inputs, Mapping) else {}
    source_period = None
    if isinstance(meteo, Mapping) and meteo.get("year_min") is not None and meteo.get("year_max") is not None:
        source_period = f"{meteo['year_min']}-{meteo['year_max']}"
    metadata: dict[str, Any] = {
        "source": "PVGIS 5.3 TMY",
        "source_url": source_url,
        "retrieved_at": retrieved_at,
        "cache_path": str(cache_path),
        "cache_hit": cache_hit,
        "request_params": params,
        "source_data_period": source_period,
        "radiation_db": meteo.get("radiation_db") if isinstance(meteo, Mapping) else None,
        "meteo_db": meteo.get("meteo_db") if isinstance(meteo, Mapping) else None,
        "location": dict(location) if isinstance(location, Mapping) else {},
        "rows": len(frame),
        "time_column": "time(UTC)" if "time(UTC)" in frame.columns else "time",
        "time_reference": "UTC",
        "reference_year": 2001,
        "irradiance_time_offset_hours": float(frame.attrs.get("irradiance_time_offset_hours", 0.0)),
        "normalization": {
            "aliases": {"G(h)": "ghi", "Gb(n)": "dni", "Gd(h)": "dhi", "T2m": "temp_air", "WS10m": "wind_speed"},
            "negative_bounded_irradiance_or_wind_values_clipped": True,
            "clipped_non_negative_counts": dict(frame.attrs.get("clipped_non_negative_counts", {})),
            "raw_columns_preserved": True,
        },
    }
    return frame, metadata


def fetch_horizon(
    latitude: float,
    longitude: float,
    *,
    session: Any | None = None,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    """Fetch PVGIS DEM-calculated horizon profile (optional shading input)."""

    lat, lon = _validate_wgs84(latitude, longitude)
    params = {
        "lat": f"{lat:.6f}",
        "lon": f"{lon:.6f}",
        "outputformat": "json",
    }
    session = session or _default_session()
    response, payload = _http_get(session, PVGIS_HORIZON_URL, params)
    inputs = payload.get("inputs") if isinstance(payload, Mapping) else None
    outputs = payload.get("outputs") if isinstance(payload, Mapping) else None
    profile = outputs.get("horizon_profile") if isinstance(outputs, Mapping) else None
    if not isinstance(profile, list) or not profile or not all(isinstance(row, Mapping) for row in profile):
        raise SourceError("PVGIS horizon response has no usable horizon_profile")
    normalised: list[dict[str, float]] = []
    for row in profile:
        normalised.append({"A": _as_float(row.get("A"), "horizon azimuth"), "H_hor": _as_float(row.get("H_hor"), "horizon height")})
    metadata = {
        "source": "PVGIS 5.3 DEM horizon",
        "source_url": _response_url(response, PVGIS_HORIZON_URL),
        "retrieved_at": _utc_now(),
        "horizon_db": inputs.get("horizon_db") if isinstance(inputs, Mapping) else None,
        "rows": len(normalised),
    }
    return normalised, metadata
