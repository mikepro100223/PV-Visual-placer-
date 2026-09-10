"""Contract tests for the bounded live data connectors.

The tests use a tiny fake HTTP session so that API failures and response shape
are deterministic.  The live smoke check is intentionally kept separate from
this unit suite; see ``docs/SOURCES.md`` for the commands and observed sample.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rooftop_pv import sources


class FakeResponse:
    def __init__(self, payload=None, *, content=b"", status_code=200, headers=None, url=""):
        self._payload = payload
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}
        self.url = url

    def raise_for_status(self):
        if self.status_code >= 400:
            raise sources.HTTPError(self.status_code, self.url)

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, *, params=None, timeout=None):
        self.calls.append((url, params, timeout))
        if not self.responses:
            raise AssertionError(f"unexpected request: {url}")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        response.url = response.url or url
        return response


def _identify_result(feature_id=123, building_id=77):
    return {
        "results": [
            {
                "type": "Feature",
                "featureId": feature_id,
                "layerBodId": sources.ROOF_LAYER,
                "properties": {
                    "building_id": building_id,
                    "flaeche": 42.5,
                    "ausrichtung": -90,
                    "neigung": 30,
                    "datum_aenderung": "2023-01-31T00:00:00",
                },
            }
        ]
    }


def _feature_payload(feature_id=123):
    return {
        "feature": {
            "type": "Feature",
            "featureId": feature_id,
            "layerBodId": sources.ROOF_LAYER,
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [[[[2657922.0, 1259430.4], [2657867.7, 1259427.7]]]],
            },
            "properties": {
                "flaeche": 42.5,
                "ausrichtung": -90,
                "neigung": 30,
                "datum_aenderung": "2023-01-31T00:00:00",
            },
        }
    }


def test_geocode_uses_wgs84_geojson_and_returns_explicit_coordinate_record(monkeypatch):
    session = FakeSession(
        [
            FakeResponse(
                {
                    "features": [
                        {
                            "geometry": {"type": "Point", "coordinates": [8.207111, 47.483116]},
                            "properties": {
                                "label": "Hauptstrasse 1 <b>5200 Brugg AG</b>",
                                "featureId": "570060_0",
                                "lat": 47.483116,
                                "lon": 8.207111,
                            },
                        }
                    ]
                }
            )
        ]
    )
    monkeypatch.setattr(sources, "_wgs84_to_lv95", lambda lat, lon: (2657921.25, 1259433.5))

    result = sources.geocode("Hauptstrasse 1 5200 Brugg", session=session)

    assert result.latitude == pytest.approx(47.483116)
    assert result.longitude == pytest.approx(8.207111)
    assert result.easting == pytest.approx(2657921.25)
    assert result.northing == pytest.approx(1259433.5)
    assert result.feature_id == "570060_0"
    url, params, timeout = session.calls[0]
    assert url == sources.GEOADMIN_SEARCH_URL
    assert params["type"] == "locations"
    assert params["sr"] == 4326
    assert params["geometryFormat"] == "geojson"
    assert timeout == sources.HTTP_TIMEOUT


def test_geocode_has_no_fabricated_fallback_for_empty_result():
    session = FakeSession([FakeResponse({"features": []})])

    with pytest.raises(sources.SourceError, match="No geocoding result"):
        sources.geocode("not a Swiss address", session=session)


def test_fetch_roofs_identifies_then_fetches_full_feature_geometry(monkeypatch):
    monkeypatch.setattr(sources, "_wgs84_to_lv95", lambda lat, lon: (2657921.25, 1259433.5))
    session = FakeSession(
        [
            FakeResponse(_identify_result()),
            FakeResponse(_feature_payload()),
        ]
    )

    roofs = sources.fetch_roofs(47.483116, 8.207111, session=session)

    assert len(roofs) == 1
    roof = roofs[0]
    assert roof.feature_id == 123
    assert roof.geometry["type"] == "MultiPolygon"
    assert roof.properties["flaeche"] == pytest.approx(42.5)
    assert roof.properties["ausrichtung"] == -90
    assert roof.properties["neigung"] == 30
    assert roof.area_m2 == pytest.approx(42.5)
    assert roof.tilt_deg == 30
    assert roof.azimuth_deg == 90
    assert roof.source_data_date == "2023-01-31T00:00:00"
    assert session.calls[0][1]["layers"] == f"all:{sources.ROOF_LAYER}"
    assert session.calls[0][1]["tolerance"] == 43  # 10 m over a 60 m, 256 px extent
    assert session.calls[0][1]["returnGeometry"] is False
    assert session.calls[1][0].endswith(f"/{sources.ROOF_LAYER}/123")
    assert session.calls[1][1]["returnGeometry"] is True


def test_fetch_roofs_returns_empty_list_when_point_has_no_roof(monkeypatch):
    monkeypatch.setattr(sources, "_wgs84_to_lv95", lambda lat, lon: (2657921.25, 1259433.5))
    session = FakeSession([FakeResponse({"results": []})])

    assert sources.fetch_roofs(47.483116, 8.207111, session=session) == []


def test_fetch_orthophoto_is_bounded_north_up_and_reports_gsd():
    image_bytes = b"\xff\xd8\xff\xd9"
    session = FakeSession(
        [
            FakeResponse(
                content=image_bytes,
                headers={"Content-Type": "image/jpeg"},
                url="https://wms.geo.admin.ch/...?REQUEST=GetMap",
            ),
            FakeResponse({"cache_update": "2026-08-25T19:27:03"}),
        ]
    )

    result = sources.fetch_orthophoto((2657800, 1259300, 2658000, 1259500), 512, 512, session=session)

    assert result.image_bytes == image_bytes
    assert result.width == 512 and result.height == 512
    assert result.gsd_m == pytest.approx((200 / 512, 200 / 512))
    assert result.north_up is True
    assert result.source_data_date == "2026-08-25T19:27:03"
    assert session.calls[0][1]["CRS"] == "EPSG:2056"
    assert session.calls[0][1]["BBOX"] == "2657800.000,1259300.000,2658000.000,1259500.000"
    assert session.calls[0][1]["FORMAT"] == "image/jpeg"


def test_fetch_orthophoto_rejects_unbounded_requests():
    with pytest.raises(ValueError, match="width"):
        sources.fetch_orthophoto((0, 0, 10, 10), 4096, 10, session=FakeSession([]))
    with pytest.raises(ValueError, match="bbox span"):
        sources.fetch_orthophoto((0, 0, 6000, 10), 512, 512, session=FakeSession([]))


def _pvgis_payload():
    return {
        "inputs": {
            "location": {"latitude": 47.483116, "longitude": 8.207111, "elevation": 355.0},
            "meteo_data": {
                "radiation_db": "PVGIS-SARAH3",
                "meteo_db": "ERA5",
                "year_min": 2005,
                "year_max": 2023,
                "use_horizon": True,
                "horizon_db": "DEM-calculated",
            },
        },
        "outputs": {
            "months_selected": [{"month": 1, "year": 2018}],
            "tmy_hourly": [
                {
                    "time(UTC)": "20180101:0000",
                    "T2m": 5.46,
                    "RH": 85.98,
                    "G(h)": 0.0,
                    "Gb(n)": 0.0,
                    "Gd(h)": 0.0,
                    "IR(h)": 319.21,
                    "WS10m": 3.35,
                    "WD10m": 233.0,
                    "SP": 96898.0,
                }
            ],
        },
        "meta": {},
    }


def test_fetch_weather_returns_dataframe_and_persists_provenance_cache(tmp_path):
    session = FakeSession([FakeResponse(_pvgis_payload())])

    frame, metadata = sources.fetch_weather(
        47.483116,
        8.207111,
        cache_dir=tmp_path,
        session=session,
        refresh=True,
    )

    assert len(frame) == 1
    assert frame.loc[frame.index[0], "temp_air"] == pytest.approx(5.46)
    assert frame.index.name == "timestamp_utc"
    assert str(frame.index[0]) == "2001-01-01 00:00:00+00:00"
    assert str(frame.loc[frame.index[0], "source_timestamp_utc"]) == "2018-01-01 00:00:00+00:00"
    assert metadata["source"] == "PVGIS 5.3 TMY"
    assert "/api/v5_3/tmy" in metadata["source_url"]
    assert metadata["source_data_period"] == "2005-2023"
    assert metadata["reference_year"] == 2001
    assert metadata["irradiance_time_offset_hours"] == 0.0
    assert metadata["normalization"]["clipped_non_negative_counts"]["wind_speed"] == 0
    assert metadata["cache_hit"] is False
    cache_path = Path(metadata["cache_path"])
    assert cache_path.exists()
    envelope = json.loads(cache_path.read_text())
    assert envelope["payload"]["outputs"]["tmy_hourly"]

    # A cached response is used without another network call.
    cached_frame, cached_metadata = sources.fetch_weather(
        47.483116,
        8.207111,
        cache_dir=tmp_path,
        session=FakeSession([]),
    )
    assert cached_metadata["cache_hit"] is True
    assert cached_frame.equals(frame)


def test_fetch_weather_rejects_malformed_upstream_payload(tmp_path):
    session = FakeSession([FakeResponse({"outputs": {"tmy_hourly": []}})])

    with pytest.raises(sources.SourceError, match="tmy_hourly"):
        sources.fetch_weather(47.483116, 8.207111, cache_dir=tmp_path, session=session, refresh=True)


def test_fetch_weather_applies_irradiance_offset_and_records_clipped_wind(tmp_path):
    payload = _pvgis_payload()
    payload["inputs"]["location"]["irradiance_time_offset"] = 0.5
    payload["outputs"]["tmy_hourly"][0]["WS10m"] = -0.08

    frame, metadata = sources.fetch_weather(
        47.483116,
        8.207111,
        cache_dir=tmp_path,
        session=FakeSession([FakeResponse(payload)]),
        refresh=True,
    )

    assert str(frame.index[0]) == "2001-01-01 00:30:00+00:00"
    assert frame.loc[frame.index[0], "wind_speed"] == 0
    assert frame.loc[frame.index[0], "WS10m"] == -0.08
    assert metadata["irradiance_time_offset_hours"] == 0.5
    assert metadata["normalization"]["clipped_non_negative_counts"]["wind_speed"] == 1


def test_fetch_horizon_parses_pvgis_profile():
    payload = {
        "inputs": {"location": {"latitude": 47.48, "longitude": 8.2}, "horizon_db": "DEM-calculated"},
        "outputs": {"horizon_profile": [{"A": -180.0, "H_hor": 6.1}]},
    }
    profile, metadata = sources.fetch_horizon(47.48, 8.2, session=FakeSession([FakeResponse(payload)]))
    assert profile == [{"A": -180.0, "H_hor": 6.1}]
    assert metadata["horizon_db"] == "DEM-calculated"
