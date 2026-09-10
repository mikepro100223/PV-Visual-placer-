"""Local Streamlit dashboard for Swiss rooftop PV planning.

The application deliberately keeps external work behind explicit form submits:
an address search calls the Swiss geodata services, the segmentation button
loads a trained checkpoint, and the calculation button requests PVGIS TMY
weather and runs pvlib.  An empty model index is a supported state: users can
still enter a manual roof area or GeoJSON footprint for a transparent scenario.
"""

from __future__ import annotations

import io
import json
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.validation import make_valid

from rooftop_pv.geometry import RoofGeometry, calculate_usable_area
from rooftop_pv.inference import (
    Detection,
    OBSTACLE_CLASSES,
    PV_CLASSES,
    overlay,
    pixel_to_map,
    predict,
    predict_tiled_obstacles,
)
from rooftop_pv.physics import LossFactors, PVConfig, simulate_pv
from rooftop_pv.runtime import ROOT


APP_TITLE = "Rooftop PV · Schweizer Dachpotenzial"
PVGIS_CACHE = ROOT / "data" / "cache"
GENEVA_CACHE = ROOT / "data" / "cache" / "geneva"
DEFAULT_IMAGE_SIZE = 1024
MIN_ORTHO_CONTEXT_M = 100.0
MAX_MODEL_CONTEXT_M = 500.0
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
MAX_GEOJSON_BYTES = 5 * 1024 * 1024
MAX_GEOJSON_FEATURES = 5_000
_LOSS_NAMES = ("soiling", "snow", "wiring", "mismatch", "degradation", "availability")
OBSTACLE_CANDIDATE_CLASSES = frozenset(
    OBSTACLE_CLASSES
    | {"tv_dish", "ladder", "balcony", "wall", "other"}
)


def _invalidate_analysis() -> None:
    """Drop every derived result when the image/geometry context changes."""

    st.session_state.pop("scenario", None)
    st.session_state.update(
        detections=[],
        obstacle_detections=[],
        selected_pv_mask_ids=[],
        selected_obstacle_mask_ids=[],
        _applied_selected_pv_mask_ids=None,
        _applied_selected_obstacle_mask_ids=None,
        pv_inference_run=False,
        obstacle_inference_run=False,
        pv_inference_metadata={},
        obstacle_inference_metadata={},
    )


def _invalidate_context(
    *, clear_assets: bool = False, clear_address: bool = False, clear_manual: bool = False
) -> None:
    """Central reset for context changes; prevents stale results/exports."""

    _invalidate_analysis()
    st.session_state.update(geneva_enabled=False, geneva_exclusions=[], geneva_metadata={})
    if clear_assets:
        st.session_state.update(orthophoto=None, orthophoto_bbox=None, manual_image_bytes=None)
        st.session_state["_upload_epoch"] = int(st.session_state.get("_upload_epoch", 0)) + 1
    if clear_address:
        st.session_state.update(address="", location=None, roofs=[], selected_roof=None, roof_index=0)
    if clear_manual:
        st.session_state.update(
            manual_roof_geometry=None,
            manual_geojson_exclusions=[],
            exclusion_upload_exclusions=[],
            manual_exclusions=[],
            manual_geojson_fingerprint=None,
            manual_geojson_role=None,
            exclusion_upload_fingerprint=None,
        )


def _quality_gate_failed(status: Mapping[str, Any]) -> bool:
    quality = str(status.get("quality_status", "")).strip().lower()
    return "failed" in quality or quality in {"rejected", "not_approved", "unverified", "explicit_path_unverified"}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _as_attr(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a connector dataclass or a mapping."""

    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def read_model_status(
    root: Path = ROOT,
    *,
    index_name: str = "current.json",
    expected_classes: set[str] | frozenset[str] = PV_CLASSES,
    model_label: str = "PV",
    explicit_weights: str | None = None,
) -> dict[str, Any]:
    """Return a truthful, UI-ready model state without loading Ultralytics.

    ``current.json`` and ``obstacles.json`` are intentionally separate model
    registries.  The optional ``explicit_weights`` path is only a convenience
    for a local secondary checkpoint; it never replaces the primary PV model.
    """

    explicit_path = Path(explicit_weights) if explicit_weights else None
    if explicit_path is not None and not explicit_path.is_absolute():
        explicit_path = Path(root) / explicit_path
    explicit_manifest = explicit_path is not None and explicit_path.suffix.lower() == ".json" and explicit_path.is_file()
    index = explicit_path if explicit_manifest else Path(root) / "artifacts" / "models" / index_name
    if not index.is_file() or (explicit_path is not None and not explicit_manifest):
        if explicit_path is not None and not explicit_manifest:
            raw_weights: str | None = str(explicit_path)
            manifest: dict[str, Any] = {"source": "explicit_path"}
        else:
            return {
                "available": False,
                "quality_status": "not_available" if index_name == "current.json" else "not_configured",
                "message": f"Kein {index_name} vorhanden. Es wurde noch kein trainiertes {model_label}-Modell registriert.",
                "index": str(index),
                "manifest": {},
                "weights": None,
                "model_label": model_label,
            }
    else:
        try:
            manifest = json.loads(index.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            return {
                "available": False,
                "quality_status": "invalid_manifest",
                "message": f"{model_label}-Modelldatei konnte nicht gelesen werden: {error}",
                "index": str(index),
                "manifest": {},
                "weights": None,
                "model_label": model_label,
            }
        if not isinstance(manifest, dict):
            manifest = {}
        raw_weights = (
            None if explicit_manifest else explicit_weights
        ) or manifest.get("best_weights") or manifest.get("weights")
    if not isinstance(manifest, dict):
        manifest = {}
    quality = str(manifest.get("quality_status", manifest.get("status", "unknown")))
    weights = Path(str(raw_weights)) if raw_weights else None
    if weights is not None and not weights.is_absolute():
        weights = Path(root) / weights
    classes = manifest.get("classes", {})
    if isinstance(classes, Mapping):
        names = {str(name).lower().replace(" ", "_") for name in classes.values()}
    elif isinstance(classes, (list, tuple)):
        names = {str(name).lower().replace(" ", "_") for name in classes}
    else:
        names = set()
    if weights is None or not weights.is_file():
        return {
            "available": False,
            "quality_status": quality,
            "message": f"{model_label}-Checkpoint fehlt; Inferenz bleibt deaktiviert. COCO-Startgewichte werden nicht als trainiertes Dachmodell verwendet.",
            "index": str(index),
            "manifest": manifest,
            "weights": str(weights) if weights else None,
            "model_label": model_label,
        }
    registered_hash = manifest.get("best_weights_sha256")
    if registered_hash and _sha256_file(weights) != registered_hash:
        return {
            "available": False,
            "quality_status": "checkpoint_hash_mismatch",
            "message": f"{model_label}-Checkpoint wurde seit der Registrierung verändert. Frühere Testergebnisse sind für diese Datei nicht bestätigt.",
            "index": str(index),
            "manifest": manifest,
            "weights": str(weights),
            "model_label": model_label,
        }
    if not names or not names.intersection(expected_classes):
        if explicit_path is not None and not explicit_manifest:
            return {
                "available": True,
                "quality_status": "explicit_path_unverified",
                "message": f"Expliziter {model_label}-Checkpoint wird beim Laden auf Klassen geprüft.",
                "index": str(index),
                "manifest": manifest,
                "weights": str(weights),
                "model_label": model_label,
            }
        return {
            "available": False,
            "quality_status": "unsupported_classes",
            "message": f"Checkpoint enthält keine dokumentierte {model_label}-Klasse; er wird nicht als solche ausgegeben.",
            "index": str(index),
            "manifest": manifest,
            "weights": str(weights),
            "model_label": model_label,
        }
    return {
        "available": True,
        "quality_status": quality,
        "message": f"Trainiertes {model_label}-Segmentierungsmodell verfügbar.",
        "index": str(index),
        "manifest": manifest,
        "weights": str(weights),
        "model_label": model_label,
    }


def read_obstacle_model_status(root: Path = ROOT, explicit_weights: str | None = None) -> dict[str, Any]:
    return read_model_status(
        root,
        index_name="obstacles.json",
        expected_classes=OBSTACLE_CANDIDATE_CLASSES,
        model_label="Hindernis",
        explicit_weights=explicit_weights,
    )


def _read_report_snapshot(reference: Any) -> dict[str, Any] | None:
    """Read a small evaluation report referenced by a model manifest."""

    if isinstance(reference, Mapping):
        return {"report": dict(reference)}
    if not reference:
        return None
    path = Path(str(reference))
    snapshot: dict[str, Any] = {"path": str(path)}
    try:
        snapshot["report"] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    except (OSError, json.JSONDecodeError):
        snapshot["report"] = None
    return snapshot


def _image_reference(
    image_record: Mapping[str, Any] | None,
    *,
    image: Image.Image | None = None,
    bbox: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    """Return exportable image provenance without embedding image bytes."""

    record = image_record if isinstance(image_record, Mapping) else {}
    raw = record.get("image_bytes")
    if isinstance(raw, (bytes, bytearray)):
        image_hash = _sha256_bytes(bytes(raw))
    else:
        image_hash = None
    actual_bbox = record.get("bbox2056") or bbox
    width = record.get("width") or (image.width if image is not None else None)
    height = record.get("height") or (image.height if image is not None else None)
    return {
        "mode": "georeferenced_orthophoto" if actual_bbox is not None else "manual_image",
        "sha256": image_hash,
        "width": width,
        "height": height,
        "bbox2056": actual_bbox,
        "requested_bbox2056": bbox,
        "gsd_m": record.get("gsd_m"),
        "north_up": record.get("north_up"),
        "source_url": record.get("source_url"),
        "source_data_date": record.get("source_data_date"),
        "source_fetched_at": record.get("source_fetched_at"),
    }


def _active_image_record() -> Mapping[str, Any]:
    orthophoto = st.session_state.get("orthophoto")
    if isinstance(orthophoto, Mapping):
        return orthophoto
    return {"image_bytes": st.session_state.get("manual_image_bytes")}


def _roof_provenance(roof: Any) -> dict[str, Any]:
    if roof is None:
        return {}
    return {
        "feature_id": _as_attr(roof, "feature_id"),
        "source_url": _as_attr(roof, "source_url"),
        "source_data_date": _as_attr(roof, "source_data_date"),
        "source_fetched_at": _as_attr(roof, "source_fetched_at"),
    }


def _model_snapshot(
    status: Mapping[str, Any],
    *,
    inference_run: bool = False,
    confidence: float | None = None,
    imgsz: int | None = None,
    image_reference: Mapping[str, Any] | None = None,
    detection_count: int | None = None,
    selected_count: int | None = None,
    loaded_weights_sha256: str | None = None,
) -> dict[str, Any]:
    """Freeze registry, quality and inference provenance for an export."""

    manifest = status.get("manifest", {})
    manifest = dict(manifest) if isinstance(manifest, Mapping) else {}
    weights = Path(str(status["weights"])) if status.get("weights") else None
    weights_hash = loaded_weights_sha256 or _sha256_file(weights)
    if weights_hash is None:
        weights_hash = manifest.get("best_weights_sha256")
    gates = manifest.get("quality_gates", manifest.get("gates", manifest.get("gate_results")))
    result: dict[str, Any] = {
        "model_label": status.get("model_label"),
        "available": bool(status.get("available", False)),
        "quality_status": status.get("quality_status"),
        "quality_gate_failed": _quality_gate_failed(status),
        "weights": {
            "path": str(weights) if weights is not None else None,
            "sha256": weights_hash,
        },
        "manifest": manifest,
        "quality_gates": gates,
        "validation_metrics": manifest.get("validation_metrics"),
        "validation_report": _read_report_snapshot(manifest.get("val_evaluation")),
        "test_report": _read_report_snapshot(manifest.get("test_evaluation")),
        "inference_run": bool(inference_run),
        "confidence_threshold": confidence,
        "imgsz": imgsz,
        "image": dict(image_reference or {}),
    }
    if detection_count is not None:
        result["detection_count"] = int(detection_count)
    if selected_count is not None:
        result["selected_detection_count"] = int(selected_count)
    return result


def _repair_geometry(geometry: BaseGeometry) -> BaseGeometry:
    if geometry.is_empty:
        raise ValueError("Geometrie darf nicht leer sein.")
    repaired = make_valid(geometry)
    if repaired.is_empty:
        raise ValueError("Geometrie ist nach Validierung leer.")
    return repaired


def _geojson_geometries(document: Mapping[str, Any]) -> list[BaseGeometry]:
    kind = document.get("type")
    if kind == "FeatureCollection":
        features = document.get("features", [])
        if not isinstance(features, list):
            raise ValueError("GeoJSON FeatureCollection hat keine gültige Feature-Liste.")
        if len(features) > MAX_GEOJSON_FEATURES:
            raise ValueError(f"GeoJSON überschreitet das Sicherheitslimit von {MAX_GEOJSON_FEATURES} Features.")
        result: list[BaseGeometry] = []
        for feature in features:
            if not isinstance(feature, Mapping) or feature.get("type") != "Feature":
                continue
            geometry = feature.get("geometry")
            if isinstance(geometry, Mapping):
                result.extend(_geojson_geometries(geometry))
        return result
    if kind == "Feature":
        geometry = document.get("geometry")
        return _geojson_geometries(geometry) if isinstance(geometry, Mapping) else []
    if kind in {"Polygon", "MultiPolygon"}:
        geometry = _repair_geometry(shape(document))
        if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
            raise ValueError("Nur Polygon- oder MultiPolygon-Geometrien werden unterstützt.")
        if max(abs(float(value)) for value in geometry.bounds) < 100_000:
            raise ValueError("GeoJSON sieht nach Gradkoordinaten aus; bitte in EPSG:2056/LV95 hochladen.")
        return [geometry]
    raise ValueError("GeoJSON muss Feature, FeatureCollection, Polygon oder MultiPolygon sein.")


def parse_geojson(value: str | bytes | Mapping[str, Any]) -> list[BaseGeometry]:
    """Parse metric LV95 polygons from a GeoJSON document.

    The dashboard labels uploads as EPSG:2056 by default.  That is intentional:
    area subtraction must happen in metres.  If a WGS84 upload is provided, the
    user can convert it first or use a source connector that returns LV95.
    """

    if isinstance(value, Mapping):
        document = value
    else:
        if isinstance(value, bytes):
            if len(value) > MAX_GEOJSON_BYTES:
                raise ValueError(f"GeoJSON-Datei ist größer als {MAX_GEOJSON_BYTES // (1024 * 1024)} MB.")
            value = value.decode("utf-8")
        elif isinstance(value, str) and len(value.encode("utf-8")) > MAX_GEOJSON_BYTES:
            raise ValueError(f"GeoJSON-Datei ist größer als {MAX_GEOJSON_BYTES // (1024 * 1024)} MB.")
        try:
            document = json.loads(value)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Ungültiges GeoJSON: {error}") from error
    if not isinstance(document, Mapping):
        raise ValueError("GeoJSON muss ein JSON-Objekt sein.")
    geometries = _geojson_geometries(document)
    if not geometries:
        raise ValueError("GeoJSON enthält keine Polygon-Geometrie.")
    return geometries


def normalise_weather(frame: pd.DataFrame) -> pd.DataFrame:
    """Map the PVGIS TMY response columns to the physics weather contract."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("PVGIS lieferte keine stündlichen Wetterdaten.")
    result = frame.copy()
    timestamp_column = next(
        (name for name in ("timestamp_utc", "time(UTC)", "timestamp", "time") if name in result), None
    )
    # The source connector already returns a timezone-aware 2001 reference
    # index.  Prefer it over the raw PVGIS source-year column (``time(UTC)``),
    # otherwise a TMY silently becomes a discontinuous 2006..2021 series.
    if not isinstance(result.index, pd.DatetimeIndex) and timestamp_column is not None:
        values = result.pop(timestamp_column)
        timestamps = pd.to_datetime(values, utc=True, errors="coerce")
        if timestamps.isna().any() and timestamp_column == "time(UTC)":
            timestamps = pd.to_datetime(values, format="%Y%m%d:%H%M", utc=True, errors="coerce")
        result.index = pd.DatetimeIndex(timestamps)
    elif not isinstance(result.index, pd.DatetimeIndex):
        raise ValueError("PVGIS-Daten enthalten keinen Zeitstempel.")
    if result.index.tz is None:
        result.index = result.index.tz_localize("UTC")
    else:
        result.index = result.index.tz_convert("UTC")

    aliases = {
        "ghi": ("ghi", "G(h)", "GHI"),
        "dni": ("dni", "Gb(n)", "DNI"),
        "dhi": ("dhi", "Gd(h)", "DHI"),
        "temp_air": ("temp_air", "T2m", "temperature"),
        "wind_speed": ("wind_speed", "WS10m", "wind"),
    }
    for target, candidates in aliases.items():
        if target not in result:
            candidate = next((name for name in candidates if name in result), None)
            if candidate is not None:
                result[target] = result[candidate]
    required = {"ghi", "dni", "dhi", "temp_air", "wind_speed"}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"PVGIS-Daten fehlen: {', '.join(sorted(missing))}")
    for column in required:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result[list(required)] = result[list(required)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    result["ghi"] = result["ghi"].clip(lower=0)
    result["dni"] = result["dni"].clip(lower=0)
    result["dhi"] = result["dhi"].clip(lower=0)
    result["wind_speed"] = result["wind_speed"].clip(lower=0)
    return result.sort_index()


def build_export_payload(
    *, location: Mapping[str, Any], assumptions: Mapping[str, Any], metrics: Mapping[str, Any],
    weather_metadata: Mapping[str, Any], model_metadata: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe export envelope without inventing missing metrics."""

    def clean(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    return {
        "schema": "rooftop-pv-scenario/v1",
        "location": clean(dict(location)),
        "assumptions": clean(dict(assumptions)),
        "metrics": clean(dict(metrics)),
        "weather": clean(dict(weather_metadata)),
        "model": clean(dict(model_metadata or {})),
        "provenance": clean(dict(provenance or {})),
    }


def _source_module():
    try:
        from rooftop_pv import sources

        return sources
    except ImportError:
        return None


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_weather_cached(latitude: float, longitude: float, refresh: bool = False):
    sources = _source_module()
    if sources is None:
        raise RuntimeError("Der Swiss/PVGIS-Quellenconnector ist in dieser lokalen Umgebung noch nicht installiert.")
    return sources.fetch_weather(
        float(latitude), float(longitude), cache_dir=PVGIS_CACHE, refresh=bool(refresh)
    )


@st.cache_resource(show_spinner=False)
def load_model_cached(weights: str, weights_sha256: str):
    from rooftop_pv.inference import load_segmenter

    model = load_segmenter(Path(weights))
    if _sha256_file(Path(weights)) != weights_sha256:
        raise ValueError("Checkpoint wurde während des Ladens geändert; bitte erneut versuchen.")
    return model


def _image_from_bytes(value: bytes | bytearray | None) -> Image.Image | None:
    if not value or len(value) > MAX_UPLOAD_BYTES:
        return None
    try:
        with Image.open(io.BytesIO(value)) as decoded:
            width, height = decoded.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                return None
            decoded.load()
            return decoded.convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError):
        return None


def _roof_property(roof: Any, name: str, default: Any = None) -> Any:
    properties = _as_attr(roof, "properties", {}) or {}
    if isinstance(properties, Mapping) and name in properties:
        return properties[name]
    return _as_attr(roof, name, default)


def _roof_azimuth(roof: Any, default: float = 180.0) -> float:
    """Return the Sonnendach orientation in the app's north-clockwise convention."""

    converted = _as_attr(roof, "azimuth_deg")
    if isinstance(converted, (int, float)):
        return float(converted) % 360.0
    raw = _roof_property(roof, "ausrichtung")
    if isinstance(raw, (int, float)):
        return (float(raw) + 180.0) % 360.0
    return float(default) % 360.0


def _roof_geometry(roof: Any) -> BaseGeometry | None:
    value = _as_attr(roof, "geometry")
    if value is None:
        return None
    if isinstance(value, BaseGeometry):
        return _repair_geometry(value)
    if isinstance(value, Mapping):
        return _repair_geometry(shape(value))
    return None


def _location_dict(location: Any) -> dict[str, Any]:
    if location is None:
        return {}
    return {
        "label": _as_attr(location, "label", _as_attr(location, "address", None)),
        "latitude": _as_attr(location, "latitude"),
        "longitude": _as_attr(location, "longitude"),
        "elevation": _as_attr(location, "elevation"),
        "easting": _as_attr(location, "easting"),
        "northing": _as_attr(location, "northing"),
        "feature_id": _as_attr(location, "feature_id"),
        "query": _as_attr(location, "query"),
        "source_url": _as_attr(location, "source_url"),
        "source_fetched_at": _as_attr(location, "source_fetched_at"),
    }


def _bbox_for_geometry(geometry: BaseGeometry, padding: float = 0.0) -> tuple[float, float, float, float]:
    min_x, min_y, max_x, max_y = geometry.bounds
    span_x = max_x - min_x
    span_y = max_y - min_y
    side = max(span_x, span_y, MIN_ORTHO_CONTEXT_M) + 2.0 * max(float(padding), 0.0)
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    half = side / 2.0
    return (center_x - half, center_y - half, center_x + half, center_y + half)


def _bbox_within(
    first: tuple[float, float, float, float] | None,
    second: tuple[float, float, float, float],
) -> bool:
    if first is None:
        return False
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    return bx1 <= ax1 and ay1 >= by1 and ax2 <= bx2 and ay2 <= by2


def _union_clipped(geometries: Iterable[BaseGeometry], roof: BaseGeometry | None) -> BaseGeometry | None:
    """Union polygons once, clipping them to the selected roof when available."""

    clipped: list[BaseGeometry] = []
    for geometry in geometries:
        candidate = _repair_geometry(geometry)
        if roof is not None:
            candidate = candidate.intersection(roof)
            if candidate.is_empty:
                continue
            candidate = _repair_geometry(candidate)
        if not candidate.is_empty and candidate.area > 0:
            clipped.append(candidate)
    if not clipped:
        return None
    return _repair_geometry(unary_union(clipped))


def _download_image_record(record: Any) -> dict[str, Any]:
    raw = _as_attr(record, "image_bytes")
    if raw is None:
        raise ValueError("Orthofoto-Antwort enthält keine Bilddaten.")
    width = int(_as_attr(record, "width", 0) or 0)
    height = int(_as_attr(record, "height", 0) or 0)
    if width <= 0 or height <= 0:
        image = _image_from_bytes(raw)
        if image is None:
            raise ValueError("Orthofoto konnte nicht gelesen werden.")
        width, height = image.size
    return {
        "image_bytes": bytes(raw),
        "width": width,
        "height": height,
        "gsd_m": _as_attr(record, "gsd_m"),
        "north_up": bool(_as_attr(record, "north_up", True)),
        "source_data_date": _as_attr(record, "source_data_date"),
        "source_url": _as_attr(record, "source_url"),
        "source_fetched_at": _as_attr(record, "source_fetched_at"),
        "bbox2056": _as_attr(record, "bbox2056"),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, BaseGeometry):
        return value.__geo_interface__
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"sha256": _sha256_bytes(value), "bytes": len(value)}
    return value


def _detections_from_state(values: Iterable[Mapping[str, Any]]) -> list[Detection]:
    result = []
    for value in values:
        polygon = tuple(tuple(float(coord) for coord in point) for point in value.get("polygon_pixels", ()))
        result.append(Detection(str(value.get("class", "unknown")), float(value.get("confidence", 0)), polygon))
    return result


def _map_ring_to_pixels(
    coordinates: Iterable[tuple[float, float]],
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    xmin, ymin, xmax, ymax = bbox
    return [
        (
            round((float(x) - xmin) / (xmax - xmin) * width),
            round((ymax - float(y)) / (ymax - ymin) * height),
        )
        for x, y in coordinates
    ]


def _draw_geometry_outline(
    image: Image.Image,
    geometry: BaseGeometry,
    bbox: tuple[float, float, float, float],
    color: tuple[int, int, int, int],
    width: int = 3,
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(getattr(geometry, "geoms", ()))
    for polygon in polygons:
        if polygon.geom_type != "Polygon":
            continue
        points = _map_ring_to_pixels(polygon.exterior.coords, bbox, image.width, image.height)
        if len(points) >= 2:
            draw.line(points, fill=color, width=width, joint="curve")


def _scene_overlay(
    image: Image.Image,
    pv_detections: list[Detection],
    obstacle_detections: list[Detection],
    roof_geometry: BaseGeometry | None,
    exclusions: Iterable[BaseGeometry],
    bbox: tuple[float, float, float, float] | None,
) -> Image.Image:
    result = overlay(image, pv_detections + obstacle_detections)
    if bbox is None:
        return result
    result = result.convert("RGBA")
    if roof_geometry is not None:
        _draw_geometry_outline(result, roof_geometry, bbox, (62, 153, 255, 230), 3)
    for exclusion in exclusions:
        _draw_geometry_outline(result, exclusion, bbox, (255, 168, 0, 235), 3)
    draw = ImageDraw.Draw(result, "RGBA")
    for detection in obstacle_detections:
        if len(detection.polygon) >= 3:
            points = [(round(x), round(y)) for x, y in detection.polygon]
            draw.line(points + [points[0]], fill=(255, 113, 66, 245), width=3)
    return result.convert("RGB")


def _projected_detection_geometry(
    detections: Iterable[Detection], bbox: tuple[float, float, float, float] | None,
    width: int, height: int, allowed_classes: set[str] | frozenset[str],
) -> tuple[list[BaseGeometry], list[str]]:
    selected = [
        detection for detection in detections
        if detection.class_name.lower().replace(" ", "_") in allowed_classes
    ]
    if bbox is None:
        if selected:
            label = "PV-" if allowed_classes is PV_CLASSES else "Hindernis-"
            return [], [f"Ohne georeferenziertes Orthofoto können {label}Pixelmasken nicht in m² umgerechnet werden."]
        return [], []
    geometries: list[BaseGeometry] = []
    errors: list[str] = []
    for detection in selected:
        class_name = detection.class_name.lower().replace(" ", "_")
        try:
            geometries.append(pixel_to_map(detection.polygon, bbox, width, height))
        except ValueError as error:
            errors.append(f"Maske {class_name}: {error}")
    return geometries, errors


def _metric_summary(
    *, roof_geometry: BaseGeometry | None, manual_area_m2: float | None,
    exclusions: list[BaseGeometry], detections: list[Detection] | None = None,
    pv_detections: list[Detection] | None = None,
    obstacle_detections: list[Detection] | None = None,
    geneva_exclusions: list[BaseGeometry] | None = None,
    bbox: tuple[float, float, float, float] | None, image_width: int, image_height: int,
    tilt_deg: float, setback_m: float, fill_ratio: float, module_area_m2: float,
    module_efficiency: float,
) -> tuple[dict[str, Any], list[str]]:
    """Calculate geometry metrics; all deductions use LV95 polygons."""

    warnings: list[str] = []
    primary_detections = pv_detections if pv_detections is not None else (detections or [])
    secondary_detections = obstacle_detections or []
    pv_geometries, projection_errors = _projected_detection_geometry(
        primary_detections, bbox, image_width, image_height, PV_CLASSES
    )
    warnings.extend(projection_errors)
    obstacle_geometries, obstacle_projection_errors = _projected_detection_geometry(
        secondary_detections, bbox, image_width, image_height, OBSTACLE_CANDIDATE_CLASSES
    )
    warnings.extend(obstacle_projection_errors)
    selected_geneva = list(geneva_exclusions or [])
    occupied = _union_clipped(pv_geometries, roof_geometry)
    ai_obstacles = _union_clipped(obstacle_geometries, roof_geometry)
    official_obstacles = _union_clipped(list(exclusions) + selected_geneva, roof_geometry)
    occupied_for_difference = occupied
    if ai_obstacles is not None and occupied_for_difference is not None:
        ai_difference = ai_obstacles.difference(occupied_for_difference)
        ai_obstacles = _repair_geometry(ai_difference) if not ai_difference.is_empty else None
        if ai_obstacles is not None and ai_obstacles.is_empty:
            ai_obstacles = None
    all_obstacles = ai_obstacles
    if official_obstacles is not None:
        occupied_and_ai = occupied
        if occupied_and_ai is not None and ai_obstacles is not None:
            occupied_and_ai = _repair_geometry(occupied_and_ai.union(ai_obstacles))
        elif occupied_and_ai is None:
            occupied_and_ai = ai_obstacles
        if occupied_and_ai is not None:
            official_difference = official_obstacles.difference(occupied_and_ai)
            official_obstacles = _repair_geometry(official_difference) if not official_difference.is_empty else None
        if official_obstacles is not None and official_obstacles.is_empty:
            official_obstacles = None
        all_obstacles = official_obstacles if all_obstacles is None else _repair_geometry(all_obstacles.union(official_obstacles))
    all_exclusions = [geometry for geometry in (occupied, ai_obstacles, official_obstacles) if geometry is not None]
    # ``raw_deductions`` is the union before the conservative obstacle buffer
    # is applied.  Keeping it separate makes the area ledger auditable: raw
    # masks are not additive when they overlap, and the buffer is its own loss.
    raw_deductions = _union_clipped(all_exclusions, roof_geometry)

    def tilted_area(geometry: BaseGeometry | None) -> float:
        if geometry is None or geometry.is_empty:
            return 0.0
        return float(geometry.area) / max(math.cos(math.radians(float(tilt_deg))), 1e-9)

    if roof_geometry is not None:
        # Keep the gross roof area independent of the user-selected setback.
        # The latter is reported separately, so users can see what changed.
        gross = calculate_usable_area(roof_geometry, tilt_deg=tilt_deg, setback_m=0.0)
        setback_only = calculate_usable_area(roof_geometry, tilt_deg=tilt_deg, setback_m=setback_m)
        usable = calculate_usable_area(
            RoofGeometry(polygon=roof_geometry, exclusions=tuple(all_exclusions)),
            tilt_deg=tilt_deg,
            setback_m=setback_m,
        )
    elif manual_area_m2 is not None:
        gross = calculate_usable_area(manual_area_m2, tilt_deg=tilt_deg)
        setback_only = gross
        if all_exclusions:
            warnings.append(
                "Manuelle Fläche hat keine Polygon-Geometrie: PV- und Ausschlussflächen können nicht geometrisch abgezogen werden."
            )
        usable = gross
    else:
        raise ValueError("Bitte eine Dachfläche oder manuelle Dachfläche angeben.")

    if roof_geometry is None and all_obstacles:
        warnings.append("Ausschlussflächen ohne Dachpolygon wurden nicht als m² abgezogen.")
    baseline_plane = max(float(setback_only.usable_area_m2) * float(fill_ratio), 0.0)
    usable_plane = max(float(usable.usable_area_m2) * float(fill_ratio), 0.0)
    baseline_module_count = max(int(math.floor(baseline_plane / module_area_m2)), 0)
    module_count = max(int(math.floor(usable_plane / module_area_m2)), 0)
    baseline_kwp = baseline_module_count * module_area_m2 * module_efficiency
    additional_kwp = module_count * module_area_m2 * module_efficiency
    obstacle_removed_area = max(float(setback_only.usable_area_m2 - usable.usable_area_m2), 0.0)
    raw_deduction_area = tilted_area(raw_deductions)
    obstacle_buffer_loss = max(obstacle_removed_area - raw_deduction_area, 0.0)
    metrics = {
        "gross_planimetric_area_m2": float(gross.planimetric_area_m2),
        "gross_roof_plane_area_m2": float(gross.tilted_area_m2),
        "setback_loss_m2": max(float(gross.tilted_area_m2 - setback_only.tilted_area_m2), 0.0),
        "occupied_area_m2": tilted_area(occupied),
        "ai_obstacle_area_m2": tilted_area(ai_obstacles),
        "manual_exclusion_area_m2": tilted_area(official_obstacles),
        "obstacle_exclusion_area_m2": tilted_area(all_obstacles),
        "raw_deduction_area_m2": raw_deduction_area,
        "obstacle_buffer_loss_m2": obstacle_buffer_loss,
        "usable_roof_plane_area_m2": float(usable.usable_area_m2),
        "fill_ratio": float(fill_ratio),
        "baseline_panel_area_budget_m2": baseline_plane,
        "baseline_module_count": baseline_module_count,
        "baseline_kwp": float(baseline_kwp),
        "panel_area_budget_m2": usable_plane,
        "module_count": module_count,
        "corrected_kwp": float(additional_kwp),
        "additional_kwp": float(additional_kwp),
        "capacity_reduction_kwp": max(float(baseline_kwp - additional_kwp), 0.0),
        "setback_m": float(setback_m),
        "tilt_deg": float(tilt_deg),
    }
    return metrics, warnings


def _monthly_frame(result: Any) -> pd.DataFrame:
    hourly = result.hourly if hasattr(result, "hourly") else pd.DataFrame()
    if hourly.empty or "energy_kwh" not in hourly:
        return pd.DataFrame(columns=["Monat", "Energie (kWh)"])
    index = hourly.index
    # PVGIS TMY is modelled on a UTC reference-year index.  Converting the
    # final 23:00 UTC sample to Europe/Zurich would create a spurious
    # thirteenth month (January of the next year), so group on the same UTC
    # labels used by ``simulate_pv.summary['monthly_energy_kwh']``.
    labels = index.tz_convert("UTC").strftime("%Y-%m") if index.tz is not None else index.strftime("%Y-%m")
    monthly = hourly.groupby(labels, sort=True)["energy_kwh"].sum().rename("Energie (kWh)").reset_index()
    return monthly.rename(columns={"index": "Monat"}) if "index" in monthly else monthly.rename(columns={monthly.columns[0]: "Monat"})


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { max-width: 1180px; padding-top: 2.3rem; }
        [data-testid="stMetricValue"] { color: #0c766e; }
        .pv-kicker { color: #0c766e; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; font-size: .76rem; }
        .pv-note { color: #56636d; font-size: .9rem; line-height: 1.5; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_model_status(status: Mapping[str, Any], label: str | None = None) -> None:
    title = label or str(status.get("model_label", "Modell"))
    if status.get("available"):
        quality = status.get("quality_status", "unbekannt")
        if quality != "dataset_gates_passed":
            st.warning(
                f"{title}-Modell geladen, aber nicht freigegeben · Qualitätsstatus: `{quality}`. "
                "Die Ausgabe bleibt experimentell und ist kein geprüfter Ertrag."
            )
        else:
            st.success(f"{title}-Modell bereit · Qualitätsstatus: `{quality}`")
        manifest = status.get("manifest", {})
        classes = manifest.get("classes") if isinstance(manifest, Mapping) else None
        if classes:
            st.caption(f"Dokumentierte Modellklassen: {classes}")
            names = {str(value).lower().replace(" ", "_") for value in classes.values()} if isinstance(classes, Mapping) else set()
            if "other" in names:
                st.caption("`other` bleibt eine unspezifische, unsichere Ausschlussklasse und wird separat als Annahme ausgewiesen.")
    else:
        st.warning(
            f"Kein nutzbares trainiertes {title}-Modell verfügbar. Der COCO-Startpunkt wird nicht als trainiertes Dachmodell verwendet."
        )
        st.caption(str(status.get("message", "Modellinferenz deaktiviert.")))


def _render_address_mode() -> tuple[BaseGeometry | None, float | None, dict[str, Any], tuple[float, float, float, float] | None]:
    sources = _source_module()
    location = st.session_state.get("location")
    roofs = st.session_state.get("roofs", [])
    orthophoto = st.session_state.get("orthophoto")
    selected_roof = st.session_state.get("selected_roof")
    if sources is None:
        st.info("Der Swiss geodata/PVGIS-Connector ist noch nicht vorhanden. Wechseln Sie für einen lokalen Test auf `Manuell`.")

    with st.form("address_form", clear_on_submit=False):
        address = st.text_input("Schweizer Adresse", value=st.session_state.get("address", ""), placeholder="z. B. Bahnhofstrasse 1, 8001 Zürich")
        submitted = st.form_submit_button("Adresse suchen", type="primary", disabled=sources is None)
    if submitted and sources is not None:
        # A new geocode request is a new scenario context even when the
        # upstream request fails; retaining the previous result would be an
        # especially dangerous silent export mismatch.
        _invalidate_context(clear_assets=True, clear_address=True, clear_manual=True)
        st.session_state.address = address
        try:
            location = sources.geocode(address)
            roofs = list(sources.fetch_roofs(float(_as_attr(location, "latitude")), float(_as_attr(location, "longitude"))))
            st.session_state.update(
                address=address, location=location, roofs=roofs,
                selected_roof=None, detections=[], obstacle_detections=[], orthophoto_bbox=None,
                geneva_enabled=False, geneva_exclusions=[], geneva_metadata={},
            )
            selected_roof = None
            orthophoto = None
            if not roofs:
                st.info("Für diesen Punkt wurde keine Dachfläche im Sonnendach gefunden.")
            else:
                st.success(f"{len(roofs)} Dachfläche(n) gefunden.")
        except Exception as error:
            st.error(f"Adresssuche/Sonnendach fehlgeschlagen: {error}")

    if roofs:
        labels = []
        for index, roof in enumerate(roofs):
            area = _roof_property(roof, "flaeche")
            tilt = _roof_property(roof, "neigung")
            azimuth = _roof_azimuth(roof)
            labels.append(f"Dachfläche {index + 1} · {area:.1f} m²" if isinstance(area, (int, float)) else f"Dachfläche {index + 1}")
            if tilt is not None or azimuth is not None:
                labels[-1] += f" · Neigung {tilt if tilt is not None else '—'}° · Azimut {azimuth if azimuth is not None else '—'}°"
        current = st.session_state.get("roof_index", 0)
        with st.form("roof_form", clear_on_submit=False):
            index = st.selectbox("Dachfläche auswählen", range(len(roofs)), index=min(current, len(roofs) - 1), format_func=lambda i: labels[i])
            load = st.form_submit_button("Dachfläche und Orthofoto laden")
        if load:
            _invalidate_context(clear_assets=True, clear_manual=True)
            st.session_state.update(
                selected_roof=None, orthophoto=None, orthophoto_bbox=None,
                geneva_enabled=False, geneva_exclusions=[], geneva_metadata={},
            )
            selected_roof = roofs[index]
            geometry = _roof_geometry(selected_roof)
            if geometry is None:
                st.error("Die Dachfläche enthält keine gültige LV95-Geometrie.")
            else:
                try:
                    bbox = _bbox_for_geometry(geometry)
                    min_x, min_y, max_x, max_y = bbox
                    span_x = max_x - min_x
                    span_y = max_y - min_y
                    if max(span_x, span_y) > MAX_MODEL_CONTEXT_M:
                        raise ValueError(
                            f"Dachfläche plus Kontext ist {max(span_x, span_y):.0f} m breit. "
                            f"Für YOLO bitte einen Ausschnitt unter {MAX_MODEL_CONTEXT_M:.0f} m oder Kacheln verwenden."
                        )
                    width = DEFAULT_IMAGE_SIZE
                    height = max(128, min(DEFAULT_IMAGE_SIZE, round(DEFAULT_IMAGE_SIZE * span_y / max(span_x, 1e-9))))
                    record = sources.fetch_orthophoto(bbox, width, height)
                    orthophoto = _download_image_record(record)
                    st.session_state.update(
                        selected_roof=selected_roof, roof_index=index, orthophoto=orthophoto,
                        orthophoto_bbox=bbox, detections=[], obstacle_detections=[],
                        geneva_enabled=False, geneva_exclusions=[], geneva_metadata={},
                    )
                    st.success("Georeferenziertes, nordausgerichtetes Orthofoto geladen.")
                    st.caption(
                        f"SWISSIMAGE-Datenstand: {orthophoto.get('source_data_date') or 'nicht angegeben'} · "
                        f"GSD: {orthophoto.get('gsd_m') or 'nicht angegeben'} m/Pixel · "
                        f"Sonnendach-Datenstand: {_as_attr(selected_roof, 'source_data_date', 'nicht angegeben')}"
                    )
                except Exception as error:
                    st.error(f"Orthofoto konnte nicht geladen werden: {error}")
    if location is not None:
        location_data = _location_dict(location)
    else:
        location_data = {}
    geometry = _roof_geometry(selected_roof) if selected_roof is not None else None
    area = None if geometry is not None else (_roof_property(selected_roof, "flaeche") if selected_roof is not None else None)
    if selected_roof is not None and geometry is not None:
        official_area = _roof_property(selected_roof, "flaeche")
        official_tilt = _roof_property(selected_roof, "neigung")
        if isinstance(official_area, (int, float)) and isinstance(official_tilt, (int, float)):
            projected_sloped = geometry.area / max(math.cos(math.radians(float(official_tilt))), 1e-9)
            st.caption(
                f"Sonnendach-Metadatum: {official_area:.1f} m² physische Dachfläche · "
                f"LV95-Polygon abgeleitet: {projected_sloped:.1f} m². Diese Quellenlücke bleibt sichtbar; "
                "die PV-Physik verwendet das ausgewählte Polygon und ersetzt es nicht stillschweigend durch das Metadatum."
            )
        properties = _as_attr(selected_roof, "properties", {}) or {}
        if isinstance(properties, Mapping):
            energy_fields = {
                str(key): value for key, value in properties.items()
                if any(token in str(key).lower() for token in ("energie", "ertrag", "kwh"))
            }
            if energy_fields:
                source_date = _as_attr(selected_roof, "source_data_date", "nicht angegeben")
                st.caption(
                    f"Offizielle Sonnendach-Energie-/Ertragsfelder (Datenstand {source_date}; "
                    f"nur separat datierte Referenz, nicht direkt mit dieser PVlib-Schätzung vergleichbar): {energy_fields}"
                )
    bbox = st.session_state.get("orthophoto_bbox")
    return geometry, float(area) if isinstance(area, (int, float)) else None, location_data, bbox


def _render_manual_mode() -> tuple[BaseGeometry | None, float | None, dict[str, Any], tuple[float, float, float, float] | None]:
    st.caption("Manueller Fallback: keine Geodaten werden vorgetäuscht. Für PVGIS werden echte Koordinaten benötigt.")
    upload_epoch = int(st.session_state.get("_upload_epoch", 0))
    image_file = st.file_uploader(
        "Optionales Dachbild", type=["png", "jpg", "jpeg", "tif", "tiff"],
        key=f"manual_image_{upload_epoch}",
    )
    geojson_file = st.file_uploader(
        "Optionales GeoJSON (EPSG:2056 / LV95, Meter)", type=["geojson", "json"],
        key=f"manual_geojson_{upload_epoch}",
    )
    role = st.radio("GeoJSON verwenden als", ["Dachfläche", "Ausschlussflächen"], horizontal=True, key="geojson_role")
    roof_geometry = None
    exclusions: list[BaseGeometry] = []
    geojson_bytes = geojson_file.getvalue() if geojson_file is not None else None
    geojson_fingerprint = _sha256_bytes(geojson_bytes) if geojson_bytes is not None else None
    geojson_context_changed = (
        geojson_fingerprint != st.session_state.get("manual_geojson_fingerprint")
        or role != st.session_state.get("manual_geojson_role")
    )
    if geojson_context_changed:
        _invalidate_analysis()
        st.session_state.update(
            manual_geojson_fingerprint=geojson_fingerprint,
            manual_geojson_role=role,
            manual_roof_geometry=None,
            manual_geojson_exclusions=[],
        )
        if geojson_bytes is not None:
            try:
                parsed = parse_geojson(geojson_bytes)
                if role == "Dachfläche":
                    roof_geometry = _repair_geometry(unary_union(parsed))
                    st.session_state.manual_roof_geometry = roof_geometry
                else:
                    exclusions = parsed
                    st.session_state.manual_geojson_exclusions = exclusions
                st.caption(f"{len(parsed)} Polygon(e) aus GeoJSON übernommen.")
            except ValueError as error:
                st.error(str(error))
    elif role == "Dachfläche" and st.session_state.get("manual_roof_geometry") is not None:
        roof_geometry = st.session_state.manual_roof_geometry
    elif role == "Ausschlussflächen":
        exclusions = list(st.session_state.get("manual_geojson_exclusions", []))
    st.session_state.manual_exclusions = exclusions
    with st.form("manual_location_form", clear_on_submit=False):
        manual_area = st.number_input(
            "Dachfläche als horizontaler Grundriss, wenn kein Polygon vorliegt (m²)",
            min_value=0.1, value=None, step=1.0, format="%.1f",
            help="Die Zahl wird als horizontale LV95-Grundrissfläche verstanden; die Dachneigung wird anschließend in die geneigte Modulfläche umgerechnet.",
        )
        latitude = st.number_input("Breitengrad für PVGIS (WGS84)", min_value=-90.0, max_value=90.0, value=None, format="%.6f", placeholder="z. B. 47.3769")
        longitude = st.number_input("Längengrad für PVGIS (WGS84)", min_value=-180.0, max_value=180.0, value=None, format="%.6f", placeholder="z. B. 8.5417")
        st.form_submit_button("Manuelle Eingaben übernehmen")
    manual_values = (manual_area, latitude, longitude)
    if manual_values != st.session_state.get("_manual_location_values"):
        st.session_state.pop("scenario", None)
        st.session_state["_manual_location_values"] = manual_values
    image_bytes = image_file.getvalue() if image_file is not None else None
    if image_bytes != st.session_state.get("manual_image_bytes"):
        _invalidate_analysis()
        # A manual exclusion drawn against another image is not silently
        # carried across an image replacement/removal.
        st.session_state.update(
            manual_geojson_exclusions=[],
            exclusion_upload_exclusions=[],
            manual_exclusions=[],
        )
    # A missing uploader value is meaningful: it means the previous image was
    # removed, so never retain its bytes or masks in the next run.
    st.session_state.manual_image_bytes = image_bytes
    if image_bytes is not None:
        if _image_from_bytes(image_bytes) is None:
            st.error(
                f"Bild konnte nicht sicher geladen werden (max. {MAX_UPLOAD_BYTES // (1024 * 1024)} MB / "
                f"{MAX_IMAGE_PIXELS:,} Pixel)."
            )
    location = {
        "label": "Manuelle Eingabe",
        "latitude": latitude,
        "longitude": longitude,
    }
    return roof_geometry, float(manual_area) if manual_area is not None else None, location, None


def _render_segmentation(
    image: Image.Image | None,
    bbox: tuple[float, float, float, float] | None,
    status: Mapping[str, Any],
    obstacle_status: Mapping[str, Any],
    roof_geometry: BaseGeometry | None = None,
    exclusions: Iterable[BaseGeometry] = (),
) -> tuple[list[Detection], list[Detection]]:
    detections = _detections_from_state(st.session_state.get("detections", []))
    obstacle_detections = _detections_from_state(st.session_state.get("obstacle_detections", []))
    if image is None:
        st.info("Laden Sie zuerst ein Orthofoto oder ein manuelles Bild.")
        return detections, obstacle_detections

    def select_masks(
        values: list[Detection], key: str, label: str, prefix: str,
    ) -> list[Detection]:
        options = list(range(len(values)))
        if not options:
            st.session_state[key] = []
            return []
        previous = st.session_state.get(key)
        if not isinstance(previous, list) or any(item not in options for item in previous):
            st.session_state[key] = options
        applied_key = f"_applied_{key}"
        applied = st.session_state.get(applied_key)
        selected = st.multiselect(
            label,
            options=options,
            format_func=lambda index: (
                f"{prefix} {index + 1}: {values[index].class_name} · "
                f"{values[index].confidence:.0%}"
            ),
            key=key,
        )
        if isinstance(applied, list) and list(selected) != applied:
            # The previous result no longer represents the selected masks.
            st.session_state.pop("scenario", None)
        st.session_state[applied_key] = list(selected)
        return [values[index] for index in selected]

    with st.expander("Bild und YOLO11-Segmentierung", expanded=True):
        scene = _scene_overlay(image, detections, obstacle_detections, roof_geometry, exclusions, bbox)
        st.image(scene, caption="Orthofoto · Dachumriss blau · manuelle/verifizierte Ausschlüsse orange · Hindernismasken rot", width="stretch")
        if bbox is not None:
            st.caption(
                "Georeferenz: EPSG:2056-Bounding-Box, north-up; der Kontext umfasst mindestens 100 m "
                "und Pixelmasken werden mit inference.pixel_to_map projiziert. "
                "Das RID2-Modell erhält überlappende 40.96-m-Ausschnitte; deren Masken werden für Flächen unioniert. "
                "Maskenzahl ist deshalb keine Objektzählung."
            )
        else:
            st.caption("Manuelles Bild ohne Georeferenz: Masken werden angezeigt, aber nicht als m² abgezogen.")
        with st.form("segmentation_form", clear_on_submit=False):
            confidence = st.slider("Konfidenzschwelle", 0.05, 0.95, 0.25, 0.05)
            run_pv = st.form_submit_button(
                "PV-Segmentierung ausführen",
                disabled=not status.get("available", False),
            )
            run_obstacles = st.form_submit_button(
                "Hindernis-Segmentierung ausführen",
                disabled=not obstacle_status.get("available", False),
            )
        if run_pv or run_obstacles:
            # A submit starts a new inference context.  Clear the shared
            # scenario and only the submitted role before loading the model;
            # if loading/prediction fails, no prior mask can leak into the
            # overlay, area calculation or export.  The other role remains
            # available for a deliberate one-model rerun.
            st.session_state.pop("scenario", None)
            if run_pv:
                detections = []
                st.session_state.update(
                    detections=[],
                    selected_pv_mask_ids=[],
                    _applied_selected_pv_mask_ids=None,
                    pv_inference_run=False,
                    pv_inference_metadata={},
                )
            if run_obstacles:
                obstacle_detections = []
                st.session_state.update(
                    obstacle_detections=[],
                    selected_obstacle_mask_ids=[],
                    _applied_selected_obstacle_mask_ids=None,
                    obstacle_inference_run=False,
                    obstacle_inference_metadata={},
                )
            try:
                if run_pv:
                    model_weights_hash = _sha256_file(Path(status["weights"]))
                    model = load_model_cached(str(status["weights"]), model_weights_hash)
                    predictions = predict(image, model=model, confidence=float(confidence), imgsz=512)
                    detections = [
                        detection for detection in predictions
                        if detection.class_name.lower().replace(" ", "_") in PV_CLASSES
                    ]
                    st.session_state.detections = [detection.as_dict() for detection in detections]
                    st.session_state.selected_pv_mask_ids = list(range(len(detections)))
                    st.session_state.pv_inference_run = True
                    st.session_state.pv_inference_metadata = _model_snapshot(
                        status,
                        loaded_weights_sha256=model_weights_hash,
                        inference_run=True,
                        confidence=float(confidence),
                        imgsz=512,
                        image_reference=_image_reference(
                            _active_image_record(), image=image, bbox=bbox
                        ),
                        detection_count=len(detections),
                        selected_count=len(detections),
                    )
                    st.success(f"{len(detections)} Maske(n) aus dem trainierten PV-Modell · Status {status.get('quality_status')}.")
                if run_obstacles:
                    model_weights_hash = _sha256_file(Path(obstacle_status["weights"]))
                    model = load_model_cached(str(obstacle_status["weights"]), model_weights_hash)
                    if bbox is not None:
                        predictions = predict_tiled_obstacles(
                            image, model=model, bbox=bbox, confidence=float(confidence), imgsz=640)
                    else:
                        predictions = predict(image, model=model, confidence=float(confidence), imgsz=640)
                    # A secondary checkpoint may contain solar_panel as a shared
                    # class; only verified obstacle classes become exclusions.
                    obstacle_detections = [
                        detection for detection in predictions
                        if detection.class_name.lower().replace(" ", "_") in OBSTACLE_CANDIDATE_CLASSES
                    ]
                    st.session_state.obstacle_detections = [
                        detection.as_dict() for detection in obstacle_detections
                    ]
                    st.session_state.selected_obstacle_mask_ids = list(range(len(obstacle_detections)))
                    st.session_state.obstacle_inference_run = True
                    st.session_state.obstacle_inference_metadata = _model_snapshot(
                        obstacle_status,
                        loaded_weights_sha256=model_weights_hash,
                        inference_run=True,
                        confidence=float(confidence),
                        imgsz=640,
                        image_reference=_image_reference(
                            _active_image_record(), image=image, bbox=bbox
                        ),
                        detection_count=len(obstacle_detections),
                        selected_count=len(obstacle_detections),
                    )
                    st.success(
                        f"{len(obstacle_detections)} Hindernismaske(n) · Status {obstacle_status.get('quality_status')}."
                    )
                st.rerun()
            except Exception as error:
                st.error(f"Segmentierung fehlgeschlagen: {error}")
        selected_pv = select_masks(
            detections, "selected_pv_mask_ids", "PV-Masken für Flächenabzug", "PV-Maske"
        )
        selected_obstacles = select_masks(
            obstacle_detections, "selected_obstacle_mask_ids", "Hindernismasken für Flächenabzug", "Hindernis"
        )
        if not st.session_state.get("pv_inference_run", False):
            st.warning(
                "Noch keine PV-Segmentierung ausgeführt: Eine leere Maskenliste ist kein geprüfter Nachweis "
                "für eine freie Dachfläche."
            )
        elif not detections:
            st.warning(
                "Die PV-Segmentierung hat keine Maske geliefert; das ist kein geprüfter Nachweis "
                "für eine freie Dachfläche."
            )
        elif detections and not selected_pv:
            st.info("Alle PV-Masken sind vom Flächenabzug ausgeschlossen; die Sichtprüfung bleibt im Overlay sichtbar.")
        if obstacle_status.get("available") and not st.session_state.get("obstacle_inference_run", False):
            st.info("Die separate Hindernis-Segmentierung wurde noch nicht ausgeführt; fehlende Masken gelten nicht als freie Fläche.")
        elif obstacle_status.get("available") and not obstacle_detections:
            st.info("Die Hindernis-Segmentierung hat keine Maske geliefert; fehlende Masken gelten nicht als freie Fläche.")
    return selected_pv, selected_obstacles


def _render_exclusions(
    roof_geometry: BaseGeometry | None,
    bbox: tuple[float, float, float, float] | None,
) -> tuple[list[BaseGeometry], list[BaseGeometry]]:
    exclusions = list(st.session_state.get("manual_geojson_exclusions", []))
    geneva_exclusions = list(st.session_state.get("geneva_exclusions", []))
    with st.expander("Manuelle Ausschlüsse / Hindernisse", expanded=False):
        st.warning(
            "Hinderniserkennung (Kamin, Dachfenster, Gaube, HVAC) ist nicht automatisch vollständig verifiziert. "
            "Eine RGB-Maske beweist keine freie Dachfläche. Zeichnen oder laden Sie verifizierte Ausschlussflächen manuell."
        )
        upload_epoch = int(st.session_state.get("_upload_epoch", 0))
        geojson_file = st.file_uploader(
            "Ausschluss-GeoJSON (EPSG:2056 / LV95)", type=["geojson", "json"],
            key=f"exclusion_geojson_{upload_epoch}",
        )
        exclusion_bytes = geojson_file.getvalue() if geojson_file is not None else None
        exclusion_fingerprint = _sha256_bytes(exclusion_bytes) if exclusion_bytes is not None else None
        exclusion_context_changed = exclusion_fingerprint != st.session_state.get("exclusion_upload_fingerprint")
        if exclusion_context_changed:
            _invalidate_analysis()
            st.session_state.exclusion_upload_fingerprint = exclusion_fingerprint
            st.session_state.exclusion_upload_exclusions = []
            if exclusion_bytes is not None:
                try:
                    uploaded_exclusions = parse_geojson(exclusion_bytes)
                    st.session_state.exclusion_upload_exclusions = uploaded_exclusions
                    st.success(f"{len(uploaded_exclusions)} manuelle Ausschlussfläche(n) geladen.")
                except ValueError as error:
                    st.error(str(error))
        exclusions.extend(st.session_state.get("exclusion_upload_exclusions", []))
        st.session_state.manual_exclusions = exclusions
        area_text = st.text_input("Zusätzliche Ausschlussfläche (m², optional)", placeholder="Nur als Hinweis; Polygon wird für Abzug benötigt")
        if area_text:
            st.caption("Eine reine m²-Angabe wird nicht stillschweigend abgezogen; laden Sie für den Geometrieabzug ein Polygon hoch.")
        try:
            from rooftop_pv.geneva import GENEVA_EXTENT_2056
        except ImportError:
            GENEVA_EXTENT_2056 = None
        if GENEVA_EXTENT_2056 is not None and _bbox_within(bbox, GENEVA_EXTENT_2056):
            st.markdown("**Offizielle Genfer Dachaufbauten (SITG)**")
            st.caption(
                "Optionaler, separater LV95-Vektorlayer. Er wird als generische geometrische Ausschlussfläche "
                "verwendet, nicht als YOLO-Label und nicht als vollständige Hindernisgrundwahrheit."
            )
            with st.form("geneva_form", clear_on_submit=False):
                enable_geneva = st.checkbox(
                    "SITG-Superstructures für diese Geneva-Dachfläche verwenden",
                    value=bool(st.session_state.get("geneva_enabled", False)),
                )
                fetch_geneva = st.form_submit_button("Offizielle Genfer Footprints laden")
            if fetch_geneva:
                if not enable_geneva:
                    st.session_state.update(geneva_enabled=False, geneva_exclusions=[], geneva_metadata={})
                    geneva_exclusions = []
                elif bbox is not None:
                    try:
                        from rooftop_pv.geneva import fetch_superstructures, superstructure_exclusions

                        source = fetch_superstructures(bbox, cache_dir=GENEVA_CACHE)
                        geneva_exclusions = list(superstructure_exclusions(source))
                        st.session_state.update(
                            geneva_enabled=True,
                            geneva_exclusions=geneva_exclusions,
                            geneva_metadata={
                                **dict(source.provenance),
                                **dict(source.pagination),
                            },
                        )
                        st.success(f"{len(geneva_exclusions)} offizielle Genfer Footprint(s) geladen.")
                    except Exception as error:
                        st.error(f"SITG-Quelle konnte nicht geladen werden: {error}")
            if st.session_state.get("geneva_enabled") and st.session_state.get("geneva_metadata"):
                metadata = st.session_state["geneva_metadata"]
                st.caption(
                    f"SITG abgerufen: {metadata.get('retrieved_at_utc', 'unbekannt')} · "
                    f"Datensatzpflege: {metadata.get('update_cadence', 'unbekannt')} · "
                    "Zeit-/Bildausrichtung bleibt eine Unsicherheit; Footprints sind nicht als vollständige Negativlabels zu lesen."
                )
        elif bbox is not None:
            st.caption("SITG-Genf-Option nur innerhalb der offiziellen Geneva-LV95-Ausdehnung verfügbar.")
    return exclusions, geneva_exclusions


def _run_scenario(
    *, geometry: BaseGeometry | None, manual_area: float | None, location: Mapping[str, Any],
    bbox: tuple[float, float, float, float] | None, image: Image.Image | None,
    detections: list[Detection], obstacle_detections: list[Detection],
    exclusions: list[BaseGeometry], geneva_exclusions: list[BaseGeometry],
    model_status: Mapping[str, Any], obstacle_status: Mapping[str, Any],
) -> None:
    roof_tilt = _as_attr(st.session_state.get("selected_roof"), "properties", {}) or {}
    default_tilt = float(roof_tilt.get("neigung", 30.0)) if isinstance(roof_tilt, Mapping) and roof_tilt.get("neigung") is not None else 30.0
    default_azimuth = _roof_azimuth(st.session_state.get("selected_roof"), 180.0)
    with st.form("scenario_form", clear_on_submit=False):
        st.subheader("Annahmen und PV-System")
        left, right = st.columns(2)
        with left:
            st.caption(
                "Montageannahme: dachparallel. Dach- und Modulneigung sind daher identisch; "
                "Aufständerung wird nicht automatisch angenommen."
            )
            if default_tilt < 5.0:
                st.info(
                    "Flachdach-Hinweis: Eine aufgeständerte Modulneigung und deren Reihenverschattung "
                    "sind in diesem Szenario nicht separat modelliert."
                )
            tilt = st.slider(
                "Dach-/Modulneigung bei dachparalleler Montage (°)",
                0.0, 89.0, min(max(default_tilt, 0.0), 89.0), 0.5,
            )
            azimuth = st.slider("Azimut, Uhrzeigersinn ab Nord (°)", 0.0, 359.0, default_azimuth % 360.0, 1.0)
            setback = st.slider("Randabstand (m)", 0.0, 3.0, 0.3, 0.1)
            fill_ratio = st.slider("Belegungsfaktor innerhalb der nutzbaren Fläche", 0.10, 1.00, 0.85, 0.05)
        with right:
            module_width = st.number_input("Modulbreite (m)", min_value=0.1, max_value=3.0, value=1.134, step=0.01)
            module_height = st.number_input("Modullänge (m)", min_value=0.1, max_value=3.0, value=1.722, step=0.01)
            efficiency_pct = st.slider("Modulwirkungsgrad (%)", 10.0, 30.0, 21.0, 0.5)
            refresh_weather = st.checkbox("PVGIS-Cache aktualisieren", value=False)
        st.markdown("**Verluste**")
        loss_cols = st.columns(3)
        with loss_cols[0]:
            soiling = st.slider("Verschmutzung (%)", 0.0, 20.0, 2.0, 0.5) / 100
            snow = st.slider("Schnee (%)", 0.0, 30.0, 0.0, 0.5) / 100
        with loss_cols[1]:
            wiring = st.slider("Verkabelung (%)", 0.0, 15.0, 2.0, 0.5) / 100
            mismatch = st.slider("Mismatch (%)", 0.0, 15.0, 2.0, 0.5) / 100
        with loss_cols[2]:
            degradation = st.slider("Degradation (%)", 0.0, 20.0, 0.0, 0.5) / 100
            availability = st.slider("Verfügbarkeit (%)", 50.0, 100.0, 99.0, 0.5) / 100
        calculate = st.form_submit_button("PVGIS TMY laden und Szenario berechnen", type="primary")
    if not calculate:
        return
    # A failed replacement must not leave an older result attached to the
    # newly submitted assumptions or available for download.
    st.session_state.pop("scenario", None)
    try:
        if image is not None and not st.session_state.get("pv_inference_run", False):
            st.warning(
                "Für dieses Bild wurde noch keine PV-Segmentierung ausgeführt. Die Berechnung setzt deshalb "
                "keine belegten PV-Flächen voraus; das ist kein geprüfter Freigabenachweis."
            )
        metric_values, geometry_warnings = _metric_summary(
            roof_geometry=geometry,
            manual_area_m2=manual_area,
            exclusions=exclusions,
            geneva_exclusions=geneva_exclusions,
            pv_detections=detections,
            obstacle_detections=obstacle_detections,
            bbox=bbox,
            image_width=image.width if image else 0,
            image_height=image.height if image else 0,
            tilt_deg=tilt,
            setback_m=setback,
            fill_ratio=fill_ratio,
            module_area_m2=module_width * module_height,
            module_efficiency=efficiency_pct / 100,
        )
        for warning in geometry_warnings:
            st.warning(warning)
        latitude = location.get("latitude")
        longitude = location.get("longitude")
        if latitude is None or longitude is None:
            raise ValueError("Für PVGIS sind echte WGS84-Koordinaten erforderlich; im manuellen Modus bitte beide Felder ausfüllen.")
        if metric_values["module_count"] <= 0:
            st.warning("Die Fläche reicht mit den aktuellen Annahmen für kein vollständiges Modul.")
        with st.spinner("PVGIS TMY wird aus Cache oder Quelle geladen und mit pvlib simuliert …"):
            weather_frame, weather_metadata = fetch_weather_cached(float(latitude), float(longitude), refresh_weather)
            weather = normalise_weather(weather_frame)
            location_metadata = weather_metadata.get("location", {})
            elevation = location_metadata.get("elevation", 0.0) if isinstance(location_metadata, Mapping) else 0.0
            try:
                elevation = float(elevation)
            except (TypeError, ValueError):
                elevation = 0.0
            losses = LossFactors(
                soiling=soiling, snow=snow, wiring=wiring, mismatch=mismatch,
                degradation=degradation, availability=availability,
            )
            config = PVConfig(
                latitude=float(latitude), longitude=float(longitude), tilt_deg=float(tilt), azimuth_deg=float(azimuth),
                available_area_m2=float(metric_values["panel_area_budget_m2"]),
                module_area_m2=float(module_width * module_height), module_efficiency=float(efficiency_pct / 100),
                module_count=int(metric_values["module_count"]), losses=losses,
                weather_source=str(weather_metadata.get("source", "PVGIS TMY")),
                elevation_m=elevation,
            )
            result = simulate_pv(weather, config)
        summary = dict(result.summary)
        summary.update(metric_values)
        image_reference = _image_reference(
            _active_image_record(), image=image, bbox=bbox
        )
        pv_snapshot = st.session_state.get("pv_inference_metadata")
        if not isinstance(pv_snapshot, Mapping) or not st.session_state.get("pv_inference_run", False):
            pv_snapshot = _model_snapshot(model_status, image_reference=image_reference)
        else:
            pv_snapshot = dict(pv_snapshot)
            pv_snapshot["selected_detection_count"] = len(detections)
            pv_snapshot["selected_mask_indices"] = list(st.session_state.get("selected_pv_mask_ids", []))
        obstacle_snapshot = st.session_state.get("obstacle_inference_metadata")
        if not isinstance(obstacle_snapshot, Mapping) or not st.session_state.get("obstacle_inference_run", False):
            obstacle_snapshot = _model_snapshot(obstacle_status, image_reference=image_reference)
        else:
            obstacle_snapshot = dict(obstacle_snapshot)
            obstacle_snapshot["selected_detection_count"] = len(obstacle_detections)
            obstacle_snapshot["selected_mask_indices"] = list(
                st.session_state.get("selected_obstacle_mask_ids", [])
            )
        st.session_state.scenario = {
            "summary": summary,
            "weather_metadata": dict(weather_metadata),
            "monthly": _monthly_frame(result),
            "hourly": result.hourly,
            "location": dict(location),
            "assumptions": {
                "tilt_deg": tilt, "azimuth_deg": azimuth, "mounting": "dachparallel",
                "setback_m": setback,
                "module_width_m": module_width, "module_height_m": module_height,
                "module_efficiency": efficiency_pct / 100, "fill_ratio": fill_ratio,
                "losses": losses.__dict__,
            },
            "model": {
                "pv": dict(pv_snapshot),
                "obstacles": dict(obstacle_snapshot),
                "geneva_footprints": bool(geneva_exclusions),
            },
            "provenance": {
                "geocode": dict(location),
                "roof": _roof_provenance(st.session_state.get("selected_roof")),
                "orthophoto": image_reference,
                "inference_image_binding": image_reference,
            },
        }
        st.success("Szenario berechnet. Die Ergebnisse sind eine PVGIS-TMY-Jahresschätzung, keine Prognose für morgen.")
    except Exception as error:
        st.error(f"Berechnung nicht möglich: {error}")


def _render_results() -> None:
    scenario = st.session_state.get("scenario")
    if not scenario:
        st.info("Noch keine Berechnung. Die Ergebnisse erscheinen nach dem Absenden des Szenarioformulars.")
        return
    summary = scenario["summary"]
    st.subheader("Ergebnis")
    metrics = st.columns(5)
    metrics[0].metric("Jahresenergie", f"{summary.get('energy_kwh', 0):,.0f} kWh".replace(",", "'"))
    metrics[1].metric("Baseline ohne Belegungen", f"{summary.get('baseline_kwp', 0):,.2f} kWp".replace(",", "'"))
    metrics[2].metric("Korrigiert / zusätzlich", f"{summary.get('corrected_kwp', summary.get('additional_kwp', 0)):,.2f} kWp".replace(",", "'"))
    metrics[3].metric("Kapazitätsreduktion", f"{summary.get('capacity_reduction_kwp', 0):,.2f} kWp".replace(",", "'"))
    metrics[4].metric("PV-Module", f"{int(summary.get('module_count', 0))}")
    st.caption(
        "Belegte PV-Fläche und Hindernisse sind unioniert, auf das ausgewählte Dach geclippt und in geneigter Dachfläche angegeben. "
        "Überlappungen werden nicht doppelt gezählt; statische und saisonale Verschattung bleibt eine manuelle Unsicherheit. "
        "Die Baseline nutzt denselben Randabstand, Belegungsfaktor und dieselben Module, setzt aber keine belegten oder ausgeschlossenen Flächen voraus."
    )
    chart_data = scenario.get("monthly", pd.DataFrame())
    if not chart_data.empty:
        st.bar_chart(chart_data.set_index("Monat"), y="Energie (kWh)", color="#0c766e")
    area_data = pd.DataFrame(
        {
            "Kategorie": [
                "Brutto Dachfläche", "Randabstand-Verlust", "Bereits belegt",
                "KI-Hindernisse", "Manuelle/SITG-Ausschlüsse", "Rohunion aller Abzüge",
                "Zusätzlicher Ausschluss-/Hindernis-Pufferverlust", "Nutzbar nach allen Abzügen",
            ],
            "Fläche (m²)": [
                summary.get("gross_roof_plane_area_m2", 0), summary.get("setback_loss_m2", 0),
                summary.get("occupied_area_m2", 0), summary.get("ai_obstacle_area_m2", 0),
                summary.get("manual_exclusion_area_m2", 0), summary.get("raw_deduction_area_m2", 0),
                summary.get("obstacle_buffer_loss_m2", 0), summary.get("usable_roof_plane_area_m2", 0),
            ],
        }
    )
    st.dataframe(area_data, hide_index=True, width="stretch")
    st.caption(
        "Die Flächenzeilen sind ein Prüfledger und nicht als additive Summe zu lesen: Masken können überlappen, "
        "und der zusätzliche Hindernis-Pufferverlust wird separat ausgewiesen."
    )
    weather_metadata = scenario.get("weather_metadata", {})
    st.caption(
        f"Wetterquelle: {weather_metadata.get('source', 'unbekannt')} · "
        f"Cache: {'Treffer' if weather_metadata.get('cache_hit') else 'neu geladen'} · "
        f"Quellzeitraum: {weather_metadata.get('source_data_period', 'nicht angegeben')} · "
        f"TMY-Referenzjahr: {weather_metadata.get('reference_year', 'nicht angegeben')} · "
        f"Höhe: {weather_metadata.get('location', {}).get('elevation', 'nicht angegeben') if isinstance(weather_metadata.get('location'), Mapping) else 'nicht angegeben'} m"
    )
    st.info(
        "PVGIS TMY ist eine typische Jahres-Wettersimulation, keine Wettervorhersage. "
        "Wolken-/Globalstrahlung, Lufttemperatur und Wind gehen über pvlib ein; Schatten durch Bäume, Nachbargebäude "
        "und saisonale Hindernisse ist ohne manuelles Horizonprofil nicht vollständig bekannt."
    )
    payload = build_export_payload(
        location=scenario.get("location", {}), assumptions=scenario.get("assumptions", {}),
        metrics=summary, weather_metadata=weather_metadata, model_metadata=scenario.get("model", {}),
        provenance=scenario.get("provenance", {}),
    )
    st.download_button("JSON exportieren", data=json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), file_name="rooftop-pv-szenario.json", mime="application/json")
    if not chart_data.empty:
        st.download_button("Monatliche CSV exportieren", data=chart_data.to_csv(index=False), file_name="rooftop-pv-monatlich.csv", mime="text/csv")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="☀️", layout="wide", initial_sidebar_state="expanded")
    _inject_css()
    st.markdown('<div class="pv-kicker">Swiss rooftop intelligence</div>', unsafe_allow_html=True)
    st.title("Rooftop PV")
    st.markdown("**Schweizer Dachpotenzial sichtbar machen – mit echten Geodaten, transparenten Annahmen und pvlib.**")
    st.markdown('<p class="pv-note">Lokale Streamlit-App · keine Anmeldung · keine Analyse- oder Trackingdienste</p>', unsafe_allow_html=True)

    model_status = read_model_status()
    with st.sidebar:
        st.header("Projektstatus")
        _render_model_status(model_status, "PV")
        obstacle_path = st.text_input(
            "Optionaler Hindernis-Checkpoint",
            value="",
            placeholder="artifacts/models/obstacles.json oder best.pt",
            help="Separate RID2-Inferenz; ersetzt niemals das primäre Swiss-PV-Modell.",
        )
        obstacle_status = read_obstacle_model_status(
            explicit_weights=obstacle_path.strip() or None
        )
        _render_model_status(obstacle_status, "Hindernis")
        st.divider()
        mode = st.radio("Arbeitsmodus", ["Adresse & Sonnendach", "Manuell"], key="mode")
        st.caption("Netzwerkzugriffe passieren ausschließlich nach einem Formular-Submit.")

    if st.session_state.get("_last_mode") != mode:
        _invalidate_context(clear_assets=True, clear_address=True, clear_manual=True)
        st.session_state["_last_mode"] = mode

    if mode == "Adresse & Sonnendach":
        geometry, manual_area, location, bbox = _render_address_mode()
    else:
        geometry, manual_area, location, bbox = _render_manual_mode()
    if mode == "Manuell":
        image = _image_from_bytes(st.session_state.get("manual_image_bytes"))
    else:
        image_record = st.session_state.get("orthophoto")
        image = _image_from_bytes(image_record.get("image_bytes")) if image_record else None
    exclusions, geneva_exclusions = _render_exclusions(geometry, bbox)
    detections, obstacle_detections = _render_segmentation(
        image,
        bbox,
        model_status,
        obstacle_status,
        geometry,
        exclusions + geneva_exclusions,
    )

    if mode == "Adresse & Sonnendach" and geometry is None and manual_area is None:
        st.info("Suchen Sie eine Adresse und laden Sie eine Dachfläche, oder wechseln Sie auf `Manuell`.")
    else:
        _run_scenario(
            geometry=geometry, manual_area=manual_area, location=location, bbox=bbox, image=image,
            detections=detections, obstacle_detections=obstacle_detections,
            exclusions=exclusions, geneva_exclusions=geneva_exclusions,
            model_status=model_status, obstacle_status=obstacle_status,
        )
    _render_results()


if __name__ == "__main__":
    main()
