"""Contract tests for the bounded official Geneva superstructure adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from shapely.geometry import Polygon

from rooftop_pv.geneva import (
    GENEVA_CLASS,
    GenevaAlignmentError,
    GenevaImagery,
    GenevaSourceError,
    fetch_superstructures,
    geojson_to_yolo,
    superstructure_exclusions,
    write_yolo_annotation,
)


def _feature(object_id: int, x0: float = 2487000.0) -> dict:
    return {
        "type": "Feature",
        "id": object_id,
        "properties": {
            "OBJECTID": object_id,
            "EGID": 295159833,
            "ALTITUDE_MIN": 415.01,
            "ALTITUDE_MAX": 415.41,
            "DATE_LEVE": 1559952000000,
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[x0, 1111000], [x0 + 10, 1111000], [x0 + 10, 1111010], [x0, 1111010], [x0, 1111000]]],
        },
    }


class _Response:
    def __init__(self, payload: dict):
        self.payload = payload
        self.status_code = 200
        self.url = ""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


class _Session:
    def __init__(self, responses: list[_Response]):
        self.responses = responses
        self.calls: list[tuple[str, dict, float]] = []

    def get(self, url: str, *, params: dict, timeout: float) -> _Response:
        self.calls.append((url, params, timeout))
        return self.responses.pop(0)


def test_fetch_paginates_and_preserves_epsg2056_provenance(tmp_path: Path) -> None:
    session = _Session(
        [
            _Response({"type": "FeatureCollection", "features": [_feature(1), _feature(2, 2487020)], "exceededTransferLimit": True}),
            _Response({"type": "FeatureCollection", "features": [_feature(3, 2487040)]}),
        ]
    )

    result = fetch_superstructures(
        (2487000, 1111000, 2488000, 1112000),
        tmp_path,
        session=session,
        page_size=2,
    )

    assert len(result.geojson["features"]) == 3
    assert result.geojson["crs"]["properties"]["name"] == "EPSG:2056"
    assert result.provenance["spatial_reference"] == "EPSG:2056"
    assert result.pagination["pages"] == 2
    assert result.pagination["offsets"] == [0, 2]
    assert session.calls[0][1]["inSR"] == 2056
    assert session.calls[0][1]["outSR"] == 2056
    assert session.calls[1][1]["resultOffset"] == 2
    assert result.cache_path is not None and result.cache_path.exists()


def test_fetch_preserves_submeter_lv95_query_bounds_in_request_and_provenance(tmp_path: Path) -> None:
    session = _Session([_Response({"type": "FeatureCollection", "features": []})])
    bbox = (
        2499914.331920191,
        1117259.4878581492,
        2500014.331920191,
        1117359.4878581492,
    )

    result = fetch_superstructures(bbox, tmp_path, session=session)

    query_geometry = "2499914.331920191,1117259.4878581492,2500014.331920191,1117359.4878581492"
    assert session.calls[0][1]["geometry"] == query_geometry
    assert result.provenance["query_geometry"] == query_geometry
    assert result.provenance["queried_bbox2056"] == list(bbox)


def test_fetch_uses_cache_without_second_network_call(tmp_path: Path) -> None:
    session = _Session([_Response({"type": "FeatureCollection", "features": []})])
    bbox = (2487000, 1111000, 2488000, 1112000)
    first = fetch_superstructures(bbox, tmp_path, session=session)

    class _FailingSession:
        def get(self, *args, **kwargs):
            raise AssertionError("cache should avoid a network call")

    second = fetch_superstructures(bbox, tmp_path, session=_FailingSession())

    assert first.geojson == second.geojson
    assert second.pagination["from_cache"] is True


def test_fetch_refetches_legacy_cache_without_exact_query_provenance(tmp_path: Path) -> None:
    bbox = (2487000, 1111000, 2488000, 1112000)
    first = fetch_superstructures(
        bbox,
        tmp_path,
        session=_Session([_Response({"type": "FeatureCollection", "features": []})]),
    )
    cached = json.loads(first.cache_path.read_text())
    cached["provenance"].pop("query_geometry")
    first.cache_path.write_text(json.dumps(cached))

    refresh_session = _Session([_Response({"type": "FeatureCollection", "features": []})])
    second = fetch_superstructures(bbox, tmp_path, session=refresh_session)

    assert len(refresh_session.calls) == 1
    assert second.pagination["from_cache"] is False
    assert "query_geometry" in second.provenance


def test_fetch_rejects_unbounded_or_non_geneva_bbox(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="maximum width"):
        fetch_superstructures((2487000, 1111000, 2500000, 1112000), tmp_path)
    with pytest.raises(ValueError, match="Geneva service extent"):
        fetch_superstructures((2400000, 1111000, 2401000, 1112000), tmp_path)


def test_invalid_payload_is_source_error(tmp_path: Path) -> None:
    session = _Session([_Response({"type": "FeatureCollection", "features": [{"type": "Feature"}]})])
    with pytest.raises(GenevaSourceError, match="geometry"):
        fetch_superstructures((2487000, 1111000, 2488000, 1112000), tmp_path, session=session)


def test_superstructure_geometry_is_a_generic_exclusion() -> None:
    geojson = {"type": "FeatureCollection", "features": [_feature(1)]}

    exclusions = superstructure_exclusions(geojson)

    assert len(exclusions) == 1
    assert exclusions[0].area == pytest.approx(100)
    assert isinstance(exclusions[0], Polygon)


def test_geojson_to_yolo_requires_verified_imagery_alignment() -> None:
    imagery = GenevaImagery("tile-1.jpg", (2487000, 1110900, 2487100, 1111100), (100, 100), "2019-05-01")
    geojson = {"type": "FeatureCollection", "features": [_feature(1)]}

    with pytest.raises(GenevaAlignmentError, match="verified"):
        geojson_to_yolo(geojson, imagery)

    artifact = geojson_to_yolo(
        geojson,
        imagery,
        alignment_status="verified",
        alignment_note="True orthophoto and vector survey are the same 2019 acquisition.",
    )

    assert artifact.class_name == GENEVA_CLASS == "superstructure"
    assert artifact.source_feature_ids == (1,)
    assert len(artifact.lines) == 1
    assert artifact.lines[0].startswith("0 ")
    assert artifact.imagery["acquired_at"] == "2019-05-01"


def test_empty_result_is_not_written_as_a_negative_label(tmp_path: Path) -> None:
    imagery = GenevaImagery("tile-1.jpg", (2487000, 1110900, 2487100, 1111100), (100, 100), "2019-05-01")
    artifact = geojson_to_yolo(
        {"type": "FeatureCollection", "features": []},
        imagery,
        alignment_status="verified",
        alignment_note="manual alignment review",
    )
    target = tmp_path / "tile-1.txt"

    assert artifact.lines == ()
    assert write_yolo_annotation(artifact, target) is None
    assert not target.exists()


def test_written_annotation_has_no_untracked_metadata(tmp_path: Path) -> None:
    imagery = GenevaImagery("tile-1.jpg", (2487000, 1110900, 2487100, 1111100), (100, 100), "2019-05-01")
    artifact = geojson_to_yolo(
        {"type": "FeatureCollection", "features": [_feature(1)]},
        imagery,
        alignment_status="verified",
        alignment_note="manual alignment review",
    )
    target = tmp_path / "tile-1.txt"

    metadata_path = write_yolo_annotation(artifact, target)

    assert target.read_text().startswith("0 ")
    assert metadata_path == target.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    assert metadata["alignment_status"] == "verified"
    assert metadata["source_feature_ids"] == [1]
