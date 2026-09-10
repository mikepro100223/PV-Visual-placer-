"""Acquire and convert the public RID2 rooftop-obstacle dataset.

RID2 publishes EPSG:28992 GeoJSON geometries and 512px roof-centred PNGs,
not YOLO labels.  This module keeps the original ZIP untouched, clips the
published polygons to each roof-centred image footprint, and emits a
deterministic YOLO segmentation tree.  Tree and shadow are intentionally not
invented as negatives: RID2 explicitly excludes them from its label contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import requests
import yaml
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box, shape
from shapely.strtree import STRtree

RID2_RECORD_ID = 14062580
RID2_API_URL = f"https://zenodo.org/api/records/{RID2_RECORD_ID}"
RID2_ARCHIVE_URL = (
    f"{RID2_API_URL}/files/roof_information_dataset_2.zip/content"
)
RID2_ARCHIVE_NAME = "roof_information_dataset_2.zip"
RID2_LICENSE = "CC BY 4.0"
RID2_CRS = "EPSG:28992"
RID2_ARCHIVE_BYTES = 6_957_075_811
RID2_ARCHIVE_MD5 = "9683e12c911358528416849c0074e17d"
RID2_UNSUPPORTED_LABELS = ("Shadow", "Tree")

# Canonical names are deliberately explicit.  AC Outlet and AC System are
# both HVAC evidence; Window is kept distinct from the explicitly-labelled
# Skylight class rather than silently merging the two.
CANONICAL_CLASSES = (
    "solar_panel",
    "chimney",
    "skylight",
    "dormer",
    "roof_window",
    "hvac",
    "tv_dish",
    "ladder",
    "balcony",
    "wall",
    "other",
)
RAW_LABEL_TO_CLASS = {
    "PVModule": "solar_panel",
    "Chimney": "chimney",
    "Skylight": "skylight",
    "Dormer": "dormer",
    "Window": "roof_window",
    "AC Outlet": "hvac",
    "AC System": "hvac",
    "TV-Dish": "tv_dish",
    "Ladder": "ladder",
    "Balcony": "balcony",
    "Wall": "wall",
    "Other": "other",
}
CLASS_TO_ID = {name: index for index, name in enumerate(CANONICAL_CLASSES)}

_IMAGE_PREFIX = "case_study_roof_centered/images_roof_centered/"
_IMAGE_GEOMETRY_MEMBER = "geometries/gdf_images_roof_centered_512_case_study.json"
_SUPERSTRUCTURE_MEMBER = "geometries/gdf_all_superstructures.json"


@dataclass(frozen=True)
class PreparedRid2:
    """Paths and counts emitted by :func:`prepare_rid2_dataset`."""

    root: Path
    dataset_yaml: Path
    splits_json: Path
    images: int
    polygons: int
    split_counts: dict[str, int]
    class_counts: dict[str, int]
    skipped_polylines: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    """Return the MD5 used by Zenodo's published file metadata."""

    return _digest(Path(path), "md5")


def fetch_rid2_metadata(raw_dir: Path, *, session: requests.Session | None = None) -> Path:
    """Persist the official Zenodo record as a provenance sidecar."""

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = session or requests.Session()
    response = session.get(RID2_API_URL, timeout=(10.0, 60.0))
    response.raise_for_status()
    record = response.json()
    record_metadata = record.get("metadata", {})
    record_license = record_metadata.get("license", {})
    license_id = record_license.get("id") if isinstance(record_license, Mapping) else None
    if license_id not in {"cc-by-4.0", "CC BY 4.0"}:
        raise ValueError(f"RID2 Zenodo record is not CC BY 4.0: {license_id!r}")
    archive_records = [
        item for item in record.get("files", [])
        if isinstance(item, Mapping) and item.get("key") == RID2_ARCHIVE_NAME
    ]
    if not archive_records:
        raise ValueError("RID2 Zenodo record is missing the expected archive")
    archive_record = archive_records[0]
    if int(archive_record.get("size", -1)) != RID2_ARCHIVE_BYTES or str(
        archive_record.get("checksum", "")
    ) != f"md5:{RID2_ARCHIVE_MD5}":
        raise ValueError("RID2 Zenodo archive metadata changed; refusing provenance drift")
    metadata = {
        "dataset": "RID2 / Roof Information Dataset 2",
        "record_id": RID2_RECORD_ID,
        "record_url": RID2_API_URL,
        "archive_url": RID2_ARCHIVE_URL,
        "license": RID2_LICENSE,
        "access_right": record_metadata.get("access_right"),
        "doi": record_metadata.get("doi"),
        "retrieved_at": _utc_now(),
        "api_record": record,
    }
    destination = raw_dir / "source-metadata.json"
    destination.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def ensure_rid2_archive(
    raw_dir: Path,
    *,
    session: requests.Session | None = None,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Download and checksum the official RID2 archive if not already present."""

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / RID2_ARCHIVE_NAME
    if archive.is_file() and archive.stat().st_size == RID2_ARCHIVE_BYTES:
        if md5_file(archive) != RID2_ARCHIVE_MD5:
            raise ValueError(f"RID2 archive checksum mismatch: {archive}")
        return archive

    partial = archive.with_suffix(archive.suffix + ".part")
    session = session or requests.Session()
    response = session.get(RID2_ARCHIVE_URL, stream=True, timeout=(10.0, 120.0))
    response.raise_for_status()
    with partial.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                handle.write(chunk)
    if partial.stat().st_size != RID2_ARCHIVE_BYTES:
        raise ValueError(
            f"incomplete RID2 archive: {partial.stat().st_size} bytes, "
            f"expected {RID2_ARCHIVE_BYTES}"
        )
    if md5_file(partial) != RID2_ARCHIVE_MD5:
        raise ValueError(f"RID2 archive checksum mismatch: {partial}")
    partial.replace(archive)
    return archive


def parse_rid2_site_group(image_name: str | Path) -> str:
    """Return a deterministic 10km EPSG:28992 region key from an image ID."""

    stem = Path(image_name).stem
    try:
        easting, northing = (float(part) for part in stem.split("_", 1))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"not an RID2 roof-centred image ID: {image_name}") from exc
    if not (math.isfinite(easting) and math.isfinite(northing)):
        raise ValueError(f"non-finite RID2 image coordinates: {image_name}")
    return f"{math.floor(easting / 10000):02d}-{math.floor(northing / 10000):02d}"


def build_group_split(
    image_names: Iterable[str],
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.5, 0.25, 0.25),
) -> dict[str, dict[str, list[str]]]:
    """Assign complete spatial groups to train/val/test deterministically."""

    names = tuple(sorted(str(name) for name in image_names))
    if len(names) != len(set(names)):
        raise ValueError("image names must be unique")
    if len(ratios) != 3 or any(value <= 0 for value in ratios) or not math.isclose(sum(ratios), 1):
        raise ValueError("ratios must be three positive values summing to one")
    grouped: dict[str, list[str]] = {}
    for name in names:
        grouped.setdefault(parse_rid2_site_group(name), []).append(name)
    if len(grouped) < 3:
        raise ValueError("at least three spatial groups are required")
    ordered = sorted(
        grouped,
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(),
    )
    raw = [ratio * len(ordered) for ratio in ratios]
    counts = [math.floor(value) for value in raw]
    for index in sorted(range(3), key=lambda i: (-(raw[i] - counts[i]), i))[
        : len(ordered) - sum(counts)
    ]:
        counts[index] += 1
    for index in range(3):
        if counts[index] == 0:
            donor = max(range(3), key=counts.__getitem__)
            if counts[donor] <= 1:
                raise ValueError("not enough spatial groups for three splits")
            counts[donor] -= 1
            counts[index] = 1
    result: dict[str, dict[str, list[str]]] = {}
    cursor = 0
    for split, count in zip(("train", "val", "test"), counts):
        groups = sorted(ordered[cursor : cursor + count])
        cursor += count
        result[split] = {
            "groups": groups,
            "images": sorted(name for group in groups for name in grouped[group]),
        }
    return result


def _safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"unsafe archive member: {name!r}")
    return path


def _load_json_member(archive: zipfile.ZipFile, member: str) -> Mapping[str, Any]:
    _safe_member(member)
    try:
        document = json.loads(archive.read(member))
    except KeyError as exc:
        raise ValueError(f"RID2 archive is missing {member}") from exc
    if not isinstance(document, Mapping) or document.get("type") != "FeatureCollection":
        raise ValueError(f"RID2 member is not a GeoJSON FeatureCollection: {member}")
    return document


def _iter_polygon_parts(geometry: Any) -> Iterable[Polygon]:
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms
    elif isinstance(geometry, GeometryCollection):
        for part in geometry.geoms:
            yield from _iter_polygon_parts(part)


def geometry_to_yolo_lines(
    geometry: Mapping[str, Any],
    bounds: tuple[float, float, float, float],
    *,
    class_id: int,
    width: int,
    height: int,
    min_area_px: float = 0.25,
) -> tuple[str, ...]:
    """Clip GeoJSON geometry to an image and return normalized YOLO polygons."""

    minx, miny, maxx, maxy = bounds
    if maxx <= minx or maxy <= miny or width < 1 or height < 1:
        raise ValueError("invalid image bounds or dimensions")
    image_box = box(minx, miny, maxx, maxy)
    clipped = shape(geometry).intersection(image_box)
    pixel_area = ((maxx - minx) / width) * ((maxy - miny) / height)
    lines: list[str] = []
    for polygon in _iter_polygon_parts(clipped):
        if polygon.is_empty or polygon.area < min_area_px * pixel_area:
            continue
        coordinates: list[float] = []
        for x, y in polygon.exterior.coords:
            coordinates.extend(
                (
                    min(1.0, max(0.0, (x - minx) / (maxx - minx))),
                    min(1.0, max(0.0, (maxy - y) / (maxy - miny))),
                )
            )
        if len(coordinates) >= 6:
            lines.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in coordinates))
    return tuple(lines)


def _feature_label(feature: Mapping[str, Any]) -> str:
    properties = feature.get("properties")
    if not isinstance(properties, Mapping) or not isinstance(properties.get("label"), str):
        raise ValueError("RID2 superstructure feature has no string label")
    return str(properties["label"])


def _prepare_records(
    archive: zipfile.ZipFile,
    image_names: tuple[str, ...],
) -> tuple[dict[str, tuple[float, float, float, float, int, int]], dict[str, Any]]:
    image_document = _load_json_member(archive, _IMAGE_GEOMETRY_MEMBER)
    bounds_by_id: dict[str, tuple[float, float, float, float, int, int]] = {}
    for feature in image_document["features"]:
        properties = feature.get("properties", {})
        image_id = properties.get("id")
        if not isinstance(image_id, str):
            continue
        geometry = shape(feature["geometry"])
        width = int(properties.get("image_width_px", 512))
        height = int(properties.get("image_height_px", 512))
        bounds_by_id[image_id] = (*geometry.bounds, width, height)
    missing_ids = [Path(name).stem for name in image_names if Path(name).stem not in bounds_by_id]
    if missing_ids:
        raise ValueError(f"RID2 image geometry missing IDs, e.g. {missing_ids[:3]}")
    return bounds_by_id, _load_json_member(archive, _SUPERSTRUCTURE_MEMBER)


def prepare_rid2_dataset(
    archive_path: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    max_images: int | None = None,
) -> PreparedRid2:
    """Create a YOLO11-seg dataset from RID2 roof-centred imagery."""

    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if output_dir.exists():
        raise FileExistsError(
            f"output path already exists: {output_dir}; "
            "choose a fresh output path rather than deleting existing data"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="rid2-", dir=output_dir.parent))
    try:
        archive_bytes = archive_path.stat().st_size
        archive_md5 = md5_file(archive_path)
        archive_verified = archive_bytes == RID2_ARCHIVE_BYTES and archive_md5 == RID2_ARCHIVE_MD5
        with zipfile.ZipFile(archive_path) as archive:
            members = tuple(
                sorted(
                    name
                    for name in archive.namelist()
                    if name.startswith(_IMAGE_PREFIX) and name.lower().endswith(".png")
                )
            )
            if not members:
                raise ValueError("RID2 archive has no roof-centred PNG images")
            if max_images is not None:
                if max_images < 3:
                    raise ValueError("max_images must be at least three")
                members = tuple(
                    sorted(
                        members,
                        key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).hexdigest(),
                    )[:max_images]
                )
            bounds_by_id, superstructure_document = _prepare_records(archive, members)
            split = build_group_split([Path(name).name for name in members], seed=seed)
            member_by_id = {Path(name).name: name for name in members}
            geometries = []
            properties = []
            for feature in superstructure_document["features"]:
                try:
                    geometries.append(shape(feature["geometry"]))
                    properties.append(feature)
                except (KeyError, TypeError, ValueError):
                    continue
            tree = STRtree(geometries)
            class_counts: Counter[str] = Counter()
            split_counts: dict[str, int] = {}
            polygon_count = 0
            skipped_polylines = 0
            skipped_unsupported = Counter()
            for split_name, split_data in split.items():
                image_dir = staging / "images" / split_name
                label_dir = staging / "labels" / split_name
                image_dir.mkdir(parents=True, exist_ok=True)
                label_dir.mkdir(parents=True, exist_ok=True)
                split_counts[split_name] = len(split_data["images"])
                for image_basename in split_data["images"]:
                    member = member_by_id[image_basename]
                    image_id = Path(image_basename).stem
                    minx, miny, maxx, maxy, width, height = bounds_by_id[image_id]
                    image_box = box(minx, miny, maxx, maxy)
                    label_lines: list[str] = []
                    for feature_index in tree.query(image_box, predicate="intersects"):
                        feature = properties[int(feature_index)]
                        raw_label = _feature_label(feature)
                        geometry_type = str(feature.get("properties", {}).get("type", ""))
                        if geometry_type.lower() in {"polyline", "linestring", "line"}:
                            skipped_polylines += 1
                            continue
                        if raw_label in RID2_UNSUPPORTED_LABELS:
                            skipped_unsupported[raw_label] += 1
                            continue
                        try:
                            canonical = RAW_LABEL_TO_CLASS[raw_label]
                        except KeyError as exc:
                            raise ValueError(f"unmapped RID2 label: {raw_label!r}") from exc
                        lines = geometry_to_yolo_lines(
                            feature["geometry"],
                            (minx, miny, maxx, maxy),
                            class_id=CLASS_TO_ID[canonical],
                            width=width,
                            height=height,
                        )
                        label_lines.extend(lines)
                        class_counts[canonical] += len(lines)
                    label_lines.sort()
                    (label_dir / f"{image_id}.txt").write_text(
                        "\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8"
                    )
                    with archive.open(member) as source, (image_dir / image_basename).open("wb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                    polygon_count += len(label_lines)

        yaml_path = staging / "dataset.yaml"
        yaml_path.write_text(
            yaml.safe_dump(
                {
                    # Ultralytics resolves ``path`` from the process working
                    # directory rather than from the YAML's directory.  An
                    # absolute generated path keeps both Ultralytics and the
                    # repository's validation helper pointed at this exact
                    # prepared snapshot.
                    "path": str(output_dir.resolve()),
                    "train": "images/train",
                    "val": "images/val",
                    "test": "images/test",
                    "names": {index: name for index, name in enumerate(CANONICAL_CLASSES)},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        manifest = {
            "source": {
                "dataset": "RID2 / Roof Information Dataset 2",
                "record_id": RID2_RECORD_ID,
                "record_url": RID2_API_URL,
                "archive_url": RID2_ARCHIVE_URL,
                "archive_name": archive_path.name,
                "archive_bytes": archive_bytes,
                "archive_md5": archive_md5,
                "archive_verified": archive_verified,
                # A synthetic/test archive must not inherit the real dataset's
                # license claim merely because it follows the same structure.
                "license": RID2_LICENSE if archive_verified else None,
                "coordinate_reference_system": RID2_CRS,
                "retrieved_at": _utc_now(),
            },
            "label_contract": {
                "raw_to_canonical": RAW_LABEL_TO_CLASS,
                "unsupported_not_negative": list(RID2_UNSUPPORTED_LABELS),
                "polylines_skipped_as_area": skipped_polylines,
                "unsupported_skipped": dict(skipped_unsupported),
            },
            "seed": seed,
            "splits": split,
            "images": len(members),
            "polygons": polygon_count,
            "class_counts": dict(class_counts),
        }
        (staging / "splits.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        staging.rename(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return PreparedRid2(
        root=output_dir,
        dataset_yaml=output_dir / "dataset.yaml",
        splits_json=output_dir / "splits.json",
        images=sum(split_counts.values()),
        polygons=polygon_count,
        split_counts=split_counts,
        class_counts=dict(class_counts),
        skipped_polylines=skipped_polylines,
    )
