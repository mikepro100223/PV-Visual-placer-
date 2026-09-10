"""Acquire and prepare the public Swiss aerial PV segmentation dataset.

The Kaggle release is a binary semantic-segmentation dataset.  This module
keeps the original archive untouched and converts only its ``images`` and
``labels`` members into a deterministic YOLO polygon dataset with the single
annotated class ``solar_panel``.  Source masks in ``roofs/masks`` are roof
footprints, not obstacle annotations, and are intentionally not promoted to a
second training class.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
import requests
import yaml
from PIL import Image

DATASET_REF = "jeanprbt/swiss-solar-panels-segmentation"
DATASET_URL = f"https://www.kaggle.com/datasets/{DATASET_REF}"
DATASET_VERSION = 1
METADATA_URL = (
    "https://www.kaggle.com/api/v1/datasets/view/jeanprbt/swiss-solar-panels-segmentation"
    f"?datasetVersionNumber={DATASET_VERSION}"
)
DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/jeanprbt/swiss-solar-panels-segmentation"
    f"?datasetVersionNumber={DATASET_VERSION}"
)
DATASET_LICENSE = "CC0: Public Domain"
SUPPORTED_CLASSES = ("solar_panel",)
DEFAULT_ARCHIVE_NAME = "swiss-solar-panels-segmentation-v1.zip"
DEFAULT_OUTPUT = Path("data/processed/swiss-pv")

_TILE_RE = re.compile(
    r"^swissimage-dop10_(?P<year>\d{4})_(?P<east>\d+)\.(?P<east_sub>\d+)"
    r"-(?P<north>\d+)\.(?P<north_sub>\d+)$"
)
_IMAGE_SUFFIXES = {".jpg", ".jpeg"}


@dataclass(frozen=True)
class ArchiveInventory:
    """Validated source members needed for preparation."""

    images: tuple[str, ...]
    labels: tuple[str, ...]
    unmatched_images: tuple[str, ...]
    unmatched_labels: tuple[str, ...]
    member_count: int


@dataclass(frozen=True)
class SplitManifest:
    """Image and geographic-group membership for each split."""

    splits: dict[str, tuple[str, ...]]
    groups: dict[str, tuple[str, ...]]
    seed: int
    ratios: tuple[float, float, float]


@dataclass(frozen=True)
class PreparedDataset:
    """Paths and counts emitted by :func:`prepare_dataset`."""

    root: Path
    dataset_yaml: Path
    splits_json: Path
    images: int
    polygons: int
    split_counts: dict[str, int]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a file's SHA-256 digest without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_site_group(image_name: str | Path) -> str:
    """Return the integer SwissImage tile/site key for ``image_name``.

    Decimal suffixes identify the ten subtiles inside a 1 km source tile.  The
    imagery year is deliberately excluded, so even a future archive containing
    another year cannot place the same geographic tile in different splits.
    """

    stem = Path(image_name).stem
    match = _TILE_RE.fullmatch(stem)
    if match is None:
        raise ValueError(f"not a SwissImage tile name: {image_name}")
    return f"{match.group('east')}-{match.group('north')}"


def _validate_ratios(ratios: tuple[float, float, float]) -> tuple[float, float, float]:
    if len(ratios) != 3 or any(not math.isfinite(float(value)) or float(value) < 0 for value in ratios):
        raise ValueError("ratios must contain three finite non-negative values")
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ValueError("ratios must sum to 1")
    if ratios[0] <= 0 or ratios[1] <= 0 or ratios[2] <= 0:
        raise ValueError("train, val, and test ratios must all be positive")
    return tuple(float(value) for value in ratios)


def build_split_manifest(
    image_names: Iterable[str],
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> SplitManifest:
    """Create deterministic, geographic-group-disjoint train/val/test splits."""

    ratios = _validate_ratios(ratios)
    names = tuple(sorted(str(name) for name in image_names))
    if len(names) != len(set(names)):
        raise ValueError("image names must be unique")
    if len(names) < 3:
        raise ValueError("at least three images are required for train/val/test")

    grouped: dict[str, list[str]] = {}
    for name in names:
        grouped.setdefault(parse_site_group(name), []).append(name)
    if len(grouped) < 3:
        raise ValueError("at least three geographic groups are required for train/val/test")

    # Hash ordering is stable across Python versions and independent of input
    # order.  Group-level assignment prevents neighbouring source subtiles from
    # leaking into validation/test through the random split.
    ordered_groups = sorted(
        grouped,
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest(),
    )
    group_count = len(ordered_groups)
    raw_targets = [ratio * group_count for ratio in ratios]
    target_counts = [math.floor(target) for target in raw_targets]
    for index in sorted(range(3), key=lambda i: (-(raw_targets[i] - target_counts[i]), i))[
        : group_count - sum(target_counts)
    ]:
        target_counts[index] += 1
    # Each split must contain one whole geographic group, even for small test
    # fixtures.  For the real release (76 groups) this branch is unnecessary.
    for index in range(3):
        if target_counts[index] == 0:
            donor = max(range(3), key=lambda i: target_counts[i])
            if target_counts[donor] <= 1:
                raise ValueError("not enough geographic groups for three splits")
            target_counts[donor] -= 1
            target_counts[index] = 1

    groups_by_split: dict[str, tuple[str, ...]] = {}
    cursor = 0
    for split, count in zip(("train", "val", "test"), target_counts):
        groups_by_split[split] = tuple(sorted(ordered_groups[cursor : cursor + count]))
        cursor += count

    splits = {
        split: tuple(sorted(name for group in groups_by_split[split] for name in grouped[group]))
        for split in ("train", "val", "test")
    }
    return SplitManifest(splits=splits, groups=groups_by_split, seed=seed, ratios=ratios)


def mask_to_yolo_lines(mask: np.ndarray, *, class_id: int = 0) -> tuple[str, ...]:
    """Convert a binary mask to normalized YOLO segmentation polygon lines."""

    if class_id != 0:
        raise ValueError("the Swiss source contains only the solar_panel class (id 0)")
    array = np.asarray(mask)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("mask must be a non-empty 2-D array")
    height, width = array.shape
    binary = np.where(array > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[tuple[float, float, float]] = []
    for contour in contours:
        if cv2.contourArea(contour) <= 0:
            continue
        perimeter = cv2.arcLength(contour, closed=True)
        simplified = cv2.approxPolyDP(contour, max(0.5, 0.002 * perimeter), closed=True)
        points = simplified.reshape(-1, 2)
        if len(points) < 3:
            continue
        coordinates: list[float] = []
        for x, y in points:
            coordinates.extend(
                (
                    min(1.0, max(0.0, float(x) / width)),
                    min(1.0, max(0.0, float(y) / height)),
                )
            )
        polygons.append((float(cv2.contourArea(contour)), *coordinates))
    # Preserve deterministic ordering even though OpenCV contour order is not a
    # public API guarantee.  Larger panel components are emitted first.
    polygons.sort(key=lambda polygon: (-polygon[0], polygon[1:]))
    return tuple(
        "0 " + " ".join(f"{coordinate:.6f}" for coordinate in polygon[1:])
        for polygon in polygons
    )


def _rasterize_yolo_lines(lines: Iterable[str], shape: tuple[int, int]) -> np.ndarray:
    """Rasterize normalized YOLO polygons for conversion-fidelity checks."""

    height, width = shape
    reconstructed = np.zeros((height, width), dtype=np.uint8)
    for line in lines:
        fields = line.split()
        if len(fields) < 7 or (len(fields) - 1) % 2:
            raise ValueError("invalid YOLO polygon line")
        if int(fields[0]) != 0:
            raise ValueError("only class id 0 is supported")
        points = np.asarray(
            [
                (round(float(fields[index]) * width), round(float(fields[index + 1]) * height))
                for index in range(1, len(fields), 2)
            ],
            dtype=np.int32,
        )
        if len(points) >= 3:
            cv2.fillPoly(reconstructed, [points], 255)
    return reconstructed > 0


def polygon_mask_iou(mask: np.ndarray, lines: Iterable[str]) -> float:
    """Return IoU between a source raster mask and its YOLO reconstruction."""

    source = np.asarray(mask) > 0
    if source.ndim != 2 or not source.size:
        raise ValueError("mask must be a non-empty 2-D array")
    reconstructed = _rasterize_yolo_lines(lines, source.shape)
    union = np.logical_or(source, reconstructed).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(source, reconstructed).sum() / union)


def _safe_member_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"unsafe archive member: {name!r}")
    return path


def inspect_archive(archive_path: Path) -> ArchiveInventory:
    """Validate a zip and return image/label members with pairing diagnostics."""

    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        names = tuple(info.filename for info in archive.infolist() if not info.is_dir())
        for name in names:
            _safe_member_name(name)
        images = tuple(sorted(name for name in names if name.startswith("images/") and Path(name).suffix.lower() in _IMAGE_SUFFIXES))
        labels = tuple(sorted(name for name in names if name.startswith("labels/") and Path(name).suffix.lower() == ".png"))
    image_stems = {Path(name).stem for name in images}
    label_stems = {Path(name).stem for name in labels}
    return ArchiveInventory(
        images=images,
        labels=labels,
        unmatched_images=tuple(sorted(image_stems - label_stems)),
        unmatched_labels=tuple(sorted(label_stems - image_stems)),
        member_count=len(names),
    )


def _copy_member(archive: zipfile.ZipFile, member: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)


def _load_source_metadata(raw_dir: Path, archive_path: Path) -> dict[str, Any]:
    metadata_path = raw_dir / "source-metadata.json"
    if metadata_path.is_file():
        try:
            document = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(document, dict):
                _validate_source_metadata(document)
                if document.get("archive_sha256") != sha256_file(archive_path):
                    raise ValueError("source metadata hash does not match the supplied archive")
                return document
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "dataset_ref": DATASET_REF,
        "dataset_url": DATASET_URL,
        "metadata_url": METADATA_URL,
        "version": DATASET_VERSION,
        "license": None,
        "provenance_status": "unverified_local_archive_no_source_metadata",
        "archive_sha256": sha256_file(archive_path),
        "archive_bytes": archive_path.stat().st_size,
    }


def _validate_source_metadata(document: Mapping[str, Any]) -> None:
    """Fail closed if a cached provenance sidecar is for another release."""

    if not isinstance(document, Mapping):
        raise ValueError("source-metadata.json is not a JSON object")
    if document.get("dataset_ref") != DATASET_REF:
        raise ValueError("source-metadata.json belongs to a different dataset")
    try:
        version = int(document.get("version", -1))
    except (TypeError, ValueError):
        version = -1
    if version != DATASET_VERSION:
        raise ValueError(f"source-metadata.json is not pinned to dataset version {DATASET_VERSION}")
    if document.get("license") != DATASET_LICENSE:
        raise ValueError("source-metadata.json has an unexpected dataset license")


def _clear_managed_output(output_dir: Path) -> None:
    """Remove only a previously generated dataset, never an arbitrary tree."""

    marker = output_dir / "splits.json"
    if not marker.is_file():
        raise FileExistsError(
            f"refusing to overwrite an unrecognized output directory: {output_dir}"
        )
    try:
        document = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FileExistsError(f"refusing to overwrite an invalid output directory: {output_dir}") from error
    source = document.get("source", {}) if isinstance(document, dict) else {}
    if source.get("dataset_ref") != DATASET_REF:
        raise FileExistsError(f"refusing to overwrite another dataset at {output_dir}")
    for parent in ("images", "labels"):
        for split in ("train", "val", "test"):
            path = output_dir / parent / split
            if path.exists():
                if not path.is_dir():
                    raise FileExistsError(f"managed output path is not a directory: {path}")
                shutil.rmtree(path)
    for filename in ("dataset.yaml", "splits.json"):
        (output_dir / filename).unlink(missing_ok=True)


def prepare_dataset(
    archive_path: Path,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    overwrite: bool = False,
) -> PreparedDataset:
    """Convert the source archive into a local YOLO11 segmentation dataset."""

    archive_path = Path(archive_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"prepared dataset path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"prepared dataset already exists: {output_dir}; pass overwrite=True")
    inventory = inspect_archive(archive_path)
    if inventory.unmatched_images or inventory.unmatched_labels:
        raise ValueError(
            "images and labels must have identical stems; "
            f"unmatched images={inventory.unmatched_images}, labels={inventory.unmatched_labels}"
        )
    if not inventory.images:
        raise ValueError("archive contains no images/*.jpg members")
    split_manifest = build_split_manifest((Path(name).name for name in inventory.images), seed=seed, ratios=ratios)

    if output_dir.exists() and overwrite and any(output_dir.iterdir()):
        _clear_managed_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    member_by_stem = {Path(name).stem: name for name in inventory.labels}
    polygons_total = 0
    ious: list[float] = []
    exact_masks = 0
    valid_images = 0
    with zipfile.ZipFile(archive_path) as archive:
        for split, names in split_manifest.splits.items():
            for image_name in names:
                stem = Path(image_name).stem
                source_image = f"images/{image_name}"
                source_label = member_by_stem[stem]
                target_image = output_dir / "images" / split / image_name
                target_label = output_dir / "labels" / split / f"{stem}.txt"
                target_mask = output_dir / "masks" / split / f"{stem}.png"
                _copy_member(archive, source_image, target_image)
                try:
                    with Image.open(target_image) as image:
                        image.verify()
                    with Image.open(target_image) as image:
                        image_size = image.size
                except Exception as error:
                    raise ValueError(f"invalid source image member: {source_image}") from error
                if image_size[0] <= 0 or image_size[1] <= 0:
                    raise ValueError(f"source image has invalid dimensions: {source_image}")
                valid_images += 1
                with archive.open(source_label) as source:
                    mask = np.asarray(Image.open(source).convert("L"))
                # Keep the original raster alongside the YOLO conversion so
                # evaluation can measure polygon fidelity and future tooling
                # can recover holes/connected-component semantics exactly.
                _copy_member(archive, source_label, target_mask)
                if mask.shape != (image_size[1], image_size[0]):
                    raise ValueError(
                        f"image/mask dimensions differ for {stem}: image={image_size}, mask={mask.shape[::-1]}"
                    )
                lines = mask_to_yolo_lines(mask)
                target_label.parent.mkdir(parents=True, exist_ok=True)
                target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                polygons_total += len(lines)
                iou = polygon_mask_iou(mask, lines)
                ious.append(iou)
                exact_masks += int(iou >= 1.0 - 1e-12)

    dataset_yaml = output_dir / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                # Omitting ``path`` is intentional: Ultralytics resolves
                # split directories relative to this YAML file.  A literal
                # ``path: .`` is resolved against the process CWD by some
                # Ultralytics releases and can silently point training away
                # from this prepared dataset.
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": list(SUPPORTED_CLASSES),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    source_metadata = _load_source_metadata(archive_path.parent, archive_path)
    split_document = {
        "source": {
            "dataset_ref": DATASET_REF,
            "dataset_url": DATASET_URL,
            "version": DATASET_VERSION,
            "license": source_metadata.get("license"),
            "archive_sha256": sha256_file(archive_path),
            "archive_bytes": archive_path.stat().st_size,
            "metadata": source_metadata,
        },
        "prepared_at_utc": _utc_now(),
        "seed": split_manifest.seed,
        "ratios": split_manifest.ratios,
        "classes": list(SUPPORTED_CLASSES),
        "inventory": {
            "archive_members": inventory.member_count,
            "images": len(inventory.images),
            "labels": len(inventory.labels),
            "polygons": polygons_total,
            "valid_images": valid_images,
        },
        "conversion": {
            "mask_iou_mean": float(np.mean(ious)) if ious else 1.0,
            "mask_iou_min": float(np.min(ious)) if ious else 1.0,
            "masks_exact": exact_masks,
            "masks_total": len(ious),
            "method": "external connected-component polygons; no obstacle classes",
        },
        "groups": {split: list(groups) for split, groups in split_manifest.groups.items()},
        "splits": {split: list(names) for split, names in split_manifest.splits.items()},
    }
    splits_json = output_dir / "splits.json"
    splits_json.write_text(json.dumps(split_document, indent=2) + "\n", encoding="utf-8")
    return PreparedDataset(
        root=output_dir,
        dataset_yaml=dataset_yaml,
        splits_json=splits_json,
        images=len(inventory.images),
        polygons=polygons_total,
        split_counts={split: len(names) for split, names in split_manifest.splits.items()},
    )


def _metadata_document(raw: Mapping[str, Any], archive_path: Path) -> dict[str, Any]:
    """Persist public Kaggle metadata without cookies or signed URLs."""

    if raw.get("ref") != DATASET_REF:
        raise ValueError("Kaggle metadata identifies a different dataset")
    if raw.get("licenseName") != DATASET_LICENSE:
        raise ValueError("Kaggle dataset license differs from the documented source license")
    versions = raw.get("versions", [])
    if not any(item.get("versionNumber") == DATASET_VERSION for item in versions):
        raise ValueError("Pinned dataset version is missing from the source metadata")
    allowed = {
        key: raw[key]
        for key in (
            "id",
            "ref",
            "title",
            "description",
            "licenseName",
            "creatorName",
            "ownerRef",
            "currentVersionNumber",
            "lastUpdated",
            "totalBytes",
            "downloadCount",
            "isPrivate",
            "versions",
        )
        if key in raw
    }
    return {
        "dataset_ref": DATASET_REF,
        "dataset_url": DATASET_URL,
        "metadata_url": METADATA_URL,
        "version": DATASET_VERSION,
        "license": raw["licenseName"],
        "provenance_status": "source_metadata_recorded",
        "retrieved_at_utc": _utc_now(),
        "archive_path": archive_path.name,
        "archive_sha256": sha256_file(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "kaggle": allowed,
    }


def download_dataset(
    raw_dir: Path = Path("data/raw"),
    *,
    force: bool = False,
    timeout: float = 120.0,
    session: requests.Session | None = None,
) -> Path:
    """Download the public v1 archive and a credential-free provenance snapshot."""

    raw_dir = Path(raw_dir).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / DEFAULT_ARCHIVE_NAME
    metadata_path = raw_dir / "source-metadata.json"
    http = session or requests.Session()
    if force or not archive_path.is_file():
        response = http.get(DOWNLOAD_URL, stream=True, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(prefix=f"{archive_path.name}.", suffix=".part", dir=raw_dir, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            try:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        temporary.write(chunk)
                temporary.flush()
                os.fsync(temporary.fileno())
            except BaseException:
                temporary_path.unlink(missing_ok=True)
                raise
        try:
            with zipfile.ZipFile(temporary_path) as archive:
                bad_member = archive.testzip()
                if bad_member:
                    raise ValueError(f"downloaded archive is corrupt at member {bad_member}")
                for info in archive.infolist():
                    _safe_member_name(info.filename)
            temporary_path.replace(archive_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    else:
        with zipfile.ZipFile(archive_path) as archive:
            if archive.testzip():
                raise ValueError(f"existing archive failed zip integrity check: {archive_path}")
    if force or not metadata_path.is_file():
        response = http.get(METADATA_URL, timeout=timeout)
        response.raise_for_status()
        raw = response.json()
        if not isinstance(raw, Mapping):
            raise ValueError("Kaggle metadata response was not a JSON object")
        metadata_path.write_text(
            json.dumps(_metadata_document(raw, archive_path), indent=2) + "\n", encoding="utf-8"
        )
    else:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        _validate_source_metadata(metadata)
        if metadata.get("archive_sha256") != sha256_file(archive_path):
            raise ValueError("cached source metadata hash does not match the downloaded archive")
    return archive_path


def prepare_from_kaggle(
    raw_dir: Path = Path("data/raw"),
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    seed: int = 42,
    overwrite: bool = False,
) -> PreparedDataset:
    """Download (or reuse) the source and prepare its YOLO dataset."""

    archive = download_dataset(raw_dir)
    return prepare_dataset(archive, output_dir, seed=seed, overwrite=overwrite)


if __name__ == "__main__":  # pragma: no cover - convenience for local acquisition
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "prepare"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    archive = download_dataset(arguments.raw_dir)
    if arguments.command == "download":
        print(archive)
    else:
        result = prepare_dataset(archive, arguments.output_dir, seed=arguments.seed, overwrite=arguments.overwrite)
        print(json.dumps({"dataset_yaml": str(result.dataset_yaml), "split_counts": result.split_counts}, indent=2))
