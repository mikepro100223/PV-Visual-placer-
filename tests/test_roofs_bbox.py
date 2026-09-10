"""Contract tests for bounded Sonnendach bbox retrieval.

The GeoAdmin identify service caps one request at 200 features.  These tests
exercise the documented offset pagination and ensure the connector never turns
an upstream cap into a silently incomplete roof inventory.
"""

from __future__ import annotations

import pytest

from rooftop_pv import sources


class _Response:
    def __init__(self, payload, *, url="", status_code=200):
        self._payload = payload
        self.url = url
        self.status_code = status_code

    def json(self):
        return self._payload


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, *, params=None, timeout=None):
        self.calls.append((url, params, timeout))
        if not self.responses:
            raise AssertionError(f"unexpected request: {url}")
        response = self.responses.pop(0)
        response.url = response.url or url
        return response


def _roof_row(feature_id: int, *, building_id: int | None = None):
    return {
        "type": "Feature",
        "featureId": feature_id,
        "id": feature_id,
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": [
                [[[2650000.0 + feature_id, 1110000.0],
                  [2650001.0 + feature_id, 1110000.0],
                  [2650001.0 + feature_id, 1110001.0],
                  [2650000.0 + feature_id, 1110000.0]]]
            ],
        },
        "properties": {
            "building_id": building_id if building_id is not None else feature_id,
            "flaeche": 42.5,
            "ausrichtung": -90,
            "neigung": 30,
            "datum_aenderung": "2023-01-31T00:00:00",
        },
    }


def test_fetch_roofs_bbox_paginates_deduplicates_and_returns_full_geometry():
    first_page = {"results": [_roof_row(feature_id) for feature_id in range(200, 0, -1)]}
    second_page = {"results": [_roof_row(200), _roof_row(201)]}
    session = _Session([_Response(first_page), _Response(second_page)])

    roofs = sources.fetch_roofs_bbox(
        (2650000.0, 1110000.0, 2650100.0, 1110100.0),
        session=session,
    )

    assert [roof.feature_id for roof in roofs] == list(range(1, 202))
    assert len({roof.feature_id for roof in roofs}) == 201
    assert roofs[0].geometry["type"] == "MultiPolygon"
    assert roofs[0].properties["flaeche"] == pytest.approx(42.5)
    assert roofs[0].source_data_date == "2023-01-31T00:00:00"

    assert len(session.calls) == 2
    url, first_params, timeout = session.calls[0]
    assert url == sources.GEOADMIN_IDENTIFY_URL
    assert timeout == sources.HTTP_TIMEOUT
    assert first_params["geometryType"] == "esriGeometryEnvelope"
    assert first_params["geometry"] == "2650000.000,1110000.000,2650100.000,1110100.000"
    assert first_params["mapExtent"] == first_params["geometry"]
    assert first_params["returnGeometry"] is True
    assert first_params["geometryFormat"] == "geojson"
    assert first_params["sr"] == 2056
    assert first_params["limit"] == 200
    assert first_params["offset"] == 0
    assert session.calls[1][1]["offset"] == 200


def test_fetch_roofs_bbox_empty_result_is_not_a_fabricated_roof():
    session = _Session([_Response({"results": []})])

    assert sources.fetch_roofs_bbox((1, 2, 3, 4), session=session) == []
    assert len(session.calls) == 1


def test_fetch_roofs_bbox_rejects_malformed_feature_geometry():
    malformed = _roof_row(1)
    malformed["geometry"] = {"type": "Point", "coordinates": [1, 2]}
    session = _Session([_Response({"results": [malformed]})])

    with pytest.raises(sources.SourceError, match="no polygon geometry"):
        sources.fetch_roofs_bbox((1, 2, 3, 4), session=session)


def test_fetch_roofs_bbox_fails_explicitly_when_feature_cap_is_exceeded():
    session = _Session(
        [
            _Response({"results": [_roof_row(1), _roof_row(2)]}),
            _Response({"results": [_roof_row(3)]}),
        ]
    )

    with pytest.raises(sources.SourceError, match="max_features"):
        sources.fetch_roofs_bbox((1, 2, 3, 4), session=session, max_features=2)


def test_fetch_roofs_bbox_fails_explicitly_when_page_cap_is_reached():
    page = {"results": [_roof_row(feature_id) for feature_id in range(200)]}
    session = _Session([_Response(page)])

    with pytest.raises(sources.SourceError, match="pagination"):
        sources.fetch_roofs_bbox((1, 2, 3, 4), session=session, max_pages=1)


def test_fetch_roofs_bbox_enforces_small_bounded_lv95_extent():
    with pytest.raises(ValueError, match="roof bbox span"):
        sources.fetch_roofs_bbox((0, 0, 500.1, 100), session=_Session([]))

    with pytest.raises(ValueError, match="max_features"):
        sources.fetch_roofs_bbox((0, 0, 10, 10), session=_Session([]), max_features=20_001)
