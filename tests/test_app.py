from __future__ import annotations

import json
import io
from pathlib import Path
from types import SimpleNamespace
import hashlib

import pandas as pd
import pytest
from shapely.geometry import box
from PIL import Image


@pytest.fixture
def app_module():
    return __import__("app")


def test_model_status_is_explicit_when_index_is_missing(tmp_path: Path, app_module):
    status = app_module.read_model_status(tmp_path)

    assert status["available"] is False
    assert status["quality_status"] == "not_available"
    assert "current.json" in status["message"]


@pytest.mark.parametrize("quality", ["experimental_gates_failed", "experimental_not_evaluated", "unknown"])
def test_unapproved_model_is_not_rendered_as_ready(quality):
    testing = pytest.importorskip("streamlit.testing.v1")

    def gate_status_app(quality):
        import app as dashboard

        dashboard._render_model_status(
            {
                "available": True,
                "quality_status": quality,
                "model_label": "PV",
                "manifest": {},
            },
            "PV",
        )

    app = testing.AppTest.from_function(gate_status_app, args=(quality,), default_timeout=10).run()

    assert not app.exception
    assert not any("Modell bereit" in value.value for value in app.success)
    assert any("nicht freigegeben" in value.value for value in app.warning)


def test_model_status_rejects_a_missing_checkpoint(tmp_path: Path, app_module):
    index = tmp_path / "artifacts" / "models" / "current.json"
    index.parent.mkdir(parents=True)
    index.write_text(
        json.dumps(
            {
                "status": "completed",
                "quality_status": "experimental",
                "best_weights": str(tmp_path / "best.pt"),
                "classes": {"0": "solar_panel"},
            }
        )
    )

    status = app_module.read_model_status(tmp_path)

    assert status["available"] is False
    assert status["quality_status"] == "experimental"
    assert "Checkpoint" in status["message"]


def test_obstacle_manifest_is_separate_and_accepts_only_obstacle_classes(tmp_path: Path, app_module):
    weights = tmp_path / "obstacles.pt"
    weights.write_bytes(b"placeholder")
    manifest = tmp_path / "artifacts" / "models" / "obstacles.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "status": "completed",
                "quality_status": "experimental",
                "best_weights": str(weights),
                "classes": {"0": "solar_panel", "1": "chimney"},
            }
        )
    )

    status = app_module.read_obstacle_model_status(tmp_path)

    assert status["available"] is True
    assert status["quality_status"] == "experimental"
    assert status["model_label"] == "Hindernis"


def test_obstacle_explicit_checkpoint_path_is_supported_but_class_checked_late(tmp_path: Path, app_module):
    weights = tmp_path / "obstacles.pt"
    weights.write_bytes(b"placeholder")

    status = app_module.read_obstacle_model_status(tmp_path, explicit_weights=str(weights))

    assert status["available"] is True
    assert status["quality_status"] == "explicit_path_unverified"


def test_explicit_weights_never_inherit_a_different_model_evaluation(tmp_path, app_module):
    weights = tmp_path / "unverified.pt"
    weights.write_bytes(b"new model")
    index = tmp_path / "artifacts/models/obstacles.json"
    index.parent.mkdir(parents=True)
    index.write_text(json.dumps({"best_weights": "old.pt", "quality_status": "dataset_gates_passed",
                                "classes": {"0": "chimney"}}))
    status = app_module.read_obstacle_model_status(tmp_path, explicit_weights=str(weights))
    assert status["quality_status"] == "explicit_path_unverified"
    assert "test_evaluation" not in status["manifest"]


def test_changed_registered_checkpoint_is_not_accepted_under_previous_hash(tmp_path, app_module):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"changed")
    index = tmp_path / "artifacts/models/current.json"
    index.parent.mkdir(parents=True)
    index.write_text(json.dumps({"best_weights": str(weights), "best_weights_sha256": "old-hash",
                                "quality_status": "dataset_gates_passed", "classes": {"0": "solar_panel"}}))
    status = app_module.read_model_status(tmp_path)
    assert status["available"] is False
    assert status["quality_status"] == "checkpoint_hash_mismatch"


def test_degree_like_geojson_is_rejected(app_module):
    document = {
        "type": "Polygon",
        "coordinates": [[[8.2, 47.3], [8.2001, 47.3], [8.2001, 47.3001], [8.2, 47.3]]],
    }

    with pytest.raises(ValueError, match="EPSG:2056"):
        app_module.parse_geojson(document)


def test_upload_size_and_geojson_feature_limits_are_rejected(app_module):
    assert app_module._image_from_bytes(b"x" * (app_module.MAX_UPLOAD_BYTES + 1)) is None
    oversized_features = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "geometry": None}] * (app_module.MAX_GEOJSON_FEATURES + 1),
    }

    with pytest.raises(ValueError, match="Sicherheitslimit"):
        app_module.parse_geojson(json.dumps(oversized_features))


def test_image_pixel_limit_is_checked_before_conversion(app_module):
    image = Image.new("RGB", (5_001, 5_001), color=(0, 0, 0))
    stream = io.BytesIO()
    image.save(stream, format="PNG", optimize=True)

    assert app_module._image_from_bytes(stream.getvalue()) is None


def test_parse_geojson_returns_metric_exclusions_and_rejects_latlon_without_transformer(
    app_module,
):
    document = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Chimney"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[2657900, 1259400], [2657902, 1259400], [2657902, 1259402], [2657900, 1259400]]],
                },
            }
        ],
    }

    geometries = app_module.parse_geojson(json.dumps(document))

    assert len(geometries) == 1
    assert geometries[0].area == pytest.approx(2.0)


def test_normalise_weather_maps_pvgis_columns_and_timezone(app_module):
    frame = pd.DataFrame(
        {
            "timestamp_utc": ["2024-01-01 00:00:00+00:00", "2024-01-01 01:00:00+00:00"],
            "G(h)": [0.0, 100.0],
            "Gb(n)": [0.0, 40.0],
            "Gd(h)": [0.0, 60.0],
            "T2m": [2.0, 3.0],
            "WS10m": [1.0, 2.0],
        }
    )

    result = app_module.normalise_weather(frame)

    assert result.index.tz is not None
    assert {"ghi", "dni", "dhi", "temp_air", "wind_speed"}.issubset(result.columns)
    assert result.loc[result.index[1], "ghi"] == pytest.approx(100.0)


def test_normalise_weather_preserves_pvgis_reference_year_2001(app_module):
    index = pd.date_range("2001-01-01", periods=2, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "time(UTC)": ["20160101:0000", "20160101:0100"],
            "ghi": [0.0, 100.0], "dni": [0.0, 40.0], "dhi": [0.0, 60.0],
            "temp_air": [2.0, 3.0], "wind_speed": [1.0, 2.0],
        },
        index=index,
    )

    result = app_module.normalise_weather(frame)

    assert result.index.year.tolist() == [2001, 2001]


def test_metrics_clip_outside_masks_and_do_not_double_count_overlapping_obstacles(app_module):
    from rooftop_pv.inference import Detection

    roof = box(0, 0, 10, 10)
    bbox = (-5.0, -5.0, 15.0, 15.0)
    outside = Detection("solar_panel", 0.9, ((16, 12), (18, 12), (18, 14), (16, 14)))
    overlapping_pv = Detection("solar_panel", 0.9, ((7, 9), (11, 9), (11, 13), (7, 13)))
    overlapping_chimney = Detection("chimney", 0.9, ((7, 9), (11, 9), (11, 13), (7, 13)))

    outside_metrics, _ = app_module._metric_summary(
        roof_geometry=roof, manual_area_m2=None, exclusions=[], pv_detections=[outside],
        obstacle_detections=[], bbox=bbox, image_width=20, image_height=20,
        tilt_deg=0, setback_m=0, fill_ratio=1, module_area_m2=2, module_efficiency=0.2,
    )
    overlap_metrics, _ = app_module._metric_summary(
        roof_geometry=roof, manual_area_m2=None, exclusions=[], pv_detections=[overlapping_pv],
        obstacle_detections=[overlapping_chimney], bbox=bbox, image_width=20, image_height=20,
        tilt_deg=0, setback_m=0, fill_ratio=1, module_area_m2=2, module_efficiency=0.2,
    )

    assert outside_metrics["occupied_area_m2"] == pytest.approx(0.0)
    assert overlap_metrics["occupied_area_m2"] == pytest.approx(16.0)
    assert overlap_metrics["ai_obstacle_area_m2"] == pytest.approx(0.0)
    assert overlap_metrics["usable_roof_plane_area_m2"] == pytest.approx(84.0)
    assert overlap_metrics["baseline_kwp"] == pytest.approx(20.0)
    assert overlap_metrics["corrected_kwp"] == pytest.approx(16.8)
    assert overlap_metrics["capacity_reduction_kwp"] == pytest.approx(3.2)


def test_orthophoto_context_has_minimum_training_scale(app_module):
    bbox = app_module._bbox_for_geometry(box(2_600_000, 1_200_000, 2_600_010, 1_200_020))

    assert bbox[2] - bbox[0] >= 100.0
    assert bbox[3] - bbox[1] >= 100.0


def test_monthly_frame_keeps_a_utc_tmy_to_twelve_months(app_module):
    index = pd.date_range("2001-01-01", periods=8_760, freq="h", tz="UTC")
    result = SimpleNamespace(hourly=pd.DataFrame({"energy_kwh": 1.0}, index=index))

    monthly = app_module._monthly_frame(result)

    assert len(monthly) == 12
    assert monthly["Monat"].str.startswith("2001-").all()
    assert monthly["Energie (kWh)"].sum() == pytest.approx(8_760.0)


def test_export_payload_has_no_fake_metrics(app_module):
    payload = app_module.build_export_payload(
        location={"latitude": 47.48, "longitude": 8.2},
        assumptions={"tilt_deg": 30},
        metrics={"annual_kwh": 1234.5},
        weather_metadata={"source": "PVGIS 5.2 TMY"},
    )

    assert payload["metrics"]["annual_kwh"] == 1234.5
    assert payload["weather"]["source"] == "PVGIS 5.2 TMY"
    assert "occupied_area_m2" not in payload["metrics"]


def test_export_payload_keeps_imagery_geocode_and_model_provenance(app_module):
    payload = app_module.build_export_payload(
        location={"label": "Brugg", "source_url": "https://geo.example", "source_fetched_at": "2026-09-10T12:00:00Z"},
        assumptions={"tilt_deg": 0},
        metrics={"energy_kwh": 1},
        weather_metadata={"source": "PVGIS 5.3 TMY"},
        model_metadata={"pv": {"weights": {"sha256": "abc"}, "test_report": {"report": {"gates": "failed"}}}},
        provenance={"orthophoto": {"source_url": "https://wms.example", "gsd_m": 0.1, "bbox2056": [1, 2, 3, 4]}},
    )

    assert payload["location"]["source_url"] == "https://geo.example"
    assert payload["provenance"]["orthophoto"]["gsd_m"] == pytest.approx(0.1)
    assert payload["model"]["pv"]["weights"]["sha256"] == "abc"


def test_model_snapshot_freezes_hash_confidence_image_and_reports(tmp_path: Path, app_module):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"weights")
    report = tmp_path / "test-report.json"
    report.write_text(json.dumps({"gates": "failed", "mAP50": 0.7}), encoding="utf-8")
    status = {
        "available": True,
        "quality_status": "experimental_gates_failed",
        "model_label": "PV",
        "weights": str(weights),
        "manifest": {"test_evaluation": str(report), "quality_gates": {"approved": False}},
    }
    image = {"image_bytes": b"orthophoto", "bbox2056": (1, 2, 3, 4), "gsd_m": 0.1}

    snapshot = app_module._model_snapshot(
        status,
        inference_run=True,
        confidence=0.4,
        imgsz=512,
        image_reference=app_module._image_reference(image, bbox=(1, 2, 3, 4)),
        detection_count=2,
        selected_count=1,
    )

    assert snapshot["weights"]["sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert snapshot["confidence_threshold"] == pytest.approx(0.4)
    assert snapshot["image"]["sha256"] == hashlib.sha256(b"orthophoto").hexdigest()
    assert snapshot["test_report"]["report"]["gates"] == "failed"
    assert snapshot["quality_gates"]["approved"] is False
    assert snapshot["selected_detection_count"] == 1


def test_model_cache_changes_with_checkpoint_content(tmp_path, app_module, monkeypatch):
    from rooftop_pv import inference

    weights = tmp_path / "best.pt"
    weights.write_bytes(b"old")
    loads = []

    def load(path):
        loads.append(path.read_bytes())
        return object()

    monkeypatch.setattr(inference, "load_segmenter", load)
    app_module.load_model_cached.clear()
    first_hash = hashlib.sha256(b"old").hexdigest()
    first = app_module.load_model_cached(str(weights), first_hash)
    assert app_module.load_model_cached(str(weights), first_hash) is first
    weights.write_bytes(b"new")
    second = app_module.load_model_cached(str(weights), hashlib.sha256(b"new").hexdigest())
    assert first is not second
    assert loads == [b"old", b"new"]
    app_module.load_model_cached.clear()


def test_context_reset_removes_stale_scenario_assets():
    testing = pytest.importorskip("streamlit.testing.v1")

    def reset_app():
        import streamlit as st
        import app as dashboard
        from shapely.geometry import box

        st.session_state.update(
            scenario={"energy_kwh": 999},
            detections=[{"class": "solar_panel"}],
            obstacle_detections=[{"class": "chimney"}],
            manual_image_bytes=b"old-image",
            manual_exclusions=[box(0, 0, 1, 1)],
        )
        dashboard._invalidate_context(clear_assets=True, clear_address=True, clear_manual=True)
        st.write("reset")

    app = testing.AppTest.from_function(reset_app, default_timeout=10).run()
    state = app.session_state.filtered_state

    assert not app.exception
    assert "scenario" not in state
    assert state["detections"] == []
    assert state["obstacle_detections"] == []
    assert state["manual_image_bytes"] is None
    assert state["manual_exclusions"] == []


def test_manual_upload_replacement_and_removal_clear_old_analysis():
    testing = pytest.importorskip("streamlit.testing.v1")
    script = Path(__file__).resolve().parents[1] / "code" / "app.py"
    app = testing.AppTest.from_file(script, default_timeout=10).run()
    app.radio[0].set_value("Manuell").run()
    app.session_state["scenario"] = {"energy_kwh": 999}
    app.session_state["detections"] = [{"class": "solar_panel"}]
    app.session_state["manual_exclusions"] = [box(0, 0, 1, 1)]
    image = io.BytesIO()
    Image.new("RGB", (2, 2), color=(1, 2, 3)).save(image, format="PNG")

    app.file_uploader[0].upload("roof.png", image.getvalue(), "image/png").run()
    state = app.session_state.filtered_state
    assert state["manual_image_bytes"]
    assert "scenario" not in state
    assert state["detections"] == []
    assert state["manual_exclusions"] == []

    app.file_uploader[0].clear().run()
    state = app.session_state.filtered_state
    assert state["manual_image_bytes"] is None
    assert state["detections"] == []
    assert "scenario" not in state

    app.session_state["scenario"] = {"energy_kwh": 999}
    app.radio("mode").set_value("Adresse & Sonnendach").run()
    assert "scenario" not in app.session_state.filtered_state


def test_streamlit_initial_load_is_local_and_explains_missing_inputs():
    testing = pytest.importorskip("streamlit.testing.v1")
    script = Path(__file__).resolve().parents[1] / "code" / "app.py"
    app = testing.AppTest.from_file(script, default_timeout=10).run()

    assert not app.exception
    assert any("Rooftop PV" in title.value for title in app.title)
    assert any("Adresse" in value.value or "manuell" in value.value.lower() for value in app.info)


def test_streamlit_manual_mode_initial_load_is_local():
    testing = pytest.importorskip("streamlit.testing.v1")
    script = Path(__file__).resolve().parents[1] / "code" / "app.py"
    app = testing.AppTest.from_file(script, default_timeout=10).run()
    app.radio[0].set_value("Manuell").run()

    assert not app.exception
    assert any("Manuelle Eingaben" in button.label for button in app.button)


def _mocked_manual_app():
    """Small AppTest entrypoint with a deterministic PVGIS-like response."""

    import pandas as pd
    import app as dashboard

    index = pd.date_range("2001-01-01", periods=2, freq="h", tz="UTC")
    weather = pd.DataFrame(
        {
            "ghi": [0.0, 100.0], "dni": [0.0, 40.0], "dhi": [0.0, 60.0],
            "temp_air": [2.0, 3.0], "wind_speed": [1.0, 2.0],
        },
        index=index,
    )
    dashboard.fetch_weather_cached = lambda *args, **kwargs: (
        weather,
        {
            "source": "PVGIS 5.3 TMY (mocked)", "source_data_period": "2005-2023",
            "reference_year": 2001, "location": {"elevation": 355.0}, "cache_hit": True,
        },
    )
    dashboard.main()


def _mask_selection_app():
    import streamlit as st
    import app as dashboard
    from PIL import Image

    st.session_state.detections = [
        {"class": "solar_panel", "confidence": 0.9, "polygon_pixels": ((0, 0), (5, 0), (5, 5))}
    ]
    st.session_state.obstacle_detections = [
        {"class": "chimney", "confidence": 0.8, "polygon_pixels": ((1, 1), (3, 1), (3, 3))}
    ]
    st.session_state.pv_inference_run = True
    st.session_state.obstacle_inference_run = True
    dashboard._render_segmentation(
        Image.new("RGB", (10, 10)),
        None,
        {"available": True, "quality_status": "experimental", "model_label": "PV", "manifest": {}},
        {"available": True, "quality_status": "experimental", "model_label": "Hindernis", "manifest": {}},
    )


def test_mask_selection_widgets_default_to_all_and_render_without_error():
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_function(_mask_selection_app, default_timeout=10).run()

    assert not app.exception
    assert len(app.multiselect) == 2
    assert app.multiselect[0].value == [0]
    assert app.multiselect[1].value == [0]

    app.session_state["scenario"] = {"energy_kwh": 999}
    app.multiselect[0].set_value([]).run()
    assert "scenario" not in app.session_state.filtered_state


def _failed_pv_reinference_app():
    import streamlit as st
    import app as dashboard
    from PIL import Image

    if not st.session_state.get("seeded"):
        st.session_state.update(
            seeded=True,
            scenario={"energy_kwh": 999},
            detections=[{"class": "solar_panel", "confidence": 0.9, "polygon_pixels": ((0, 0), (4, 0), (4, 4))}],
            obstacle_detections=[{"class": "chimney", "confidence": 0.8, "polygon_pixels": ((6, 6), (8, 6), (8, 8))}],
            pv_inference_run=True,
            obstacle_inference_run=True,
            selected_pv_mask_ids=[0],
            selected_obstacle_mask_ids=[0],
        )
    dashboard.load_model_cached = lambda *args: object()

    def fail_predict(*args, **kwargs):
        raise RuntimeError("PV checkpoint failed")

    dashboard.predict = fail_predict
    dashboard._render_segmentation(
        Image.new("RGB", (10, 10)),
        None,
        {"available": True, "quality_status": "experimental", "weights": "missing.pt", "manifest": {}},
        {"available": False, "quality_status": "not_configured", "manifest": {}},
    )


def _successful_same_count_pv_reinference_app():
    import streamlit as st
    import app as dashboard
    from PIL import Image
    from rooftop_pv.inference import Detection

    if not st.session_state.get("seeded"):
        st.session_state.update(
            seeded=True,
            scenario={"energy_kwh": 999},
            detections=[{"class": "solar_panel", "confidence": 0.9, "polygon_pixels": ((0, 0), (4, 0), (4, 4))}],
            obstacle_detections=[{"class": "chimney", "confidence": 0.8, "polygon_pixels": ((6, 6), (8, 6), (8, 8))}],
            pv_inference_run=True,
            obstacle_inference_run=True,
            selected_pv_mask_ids=[0],
            selected_obstacle_mask_ids=[0],
        )
    dashboard.load_model_cached = lambda *args: object()
    dashboard.predict = lambda *args, **kwargs: [
        Detection("solar_panel", 0.95, ((1, 1), (8, 1), (8, 8), (1, 8)))
    ]
    dashboard._render_segmentation(
        Image.new("RGB", (10, 10)),
        None,
        {"available": True, "quality_status": "experimental", "weights": "missing.pt", "manifest": {}},
        {"available": False, "quality_status": "not_configured", "manifest": {}},
    )


def test_failed_pv_reinference_clears_only_pv_and_scenario():
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_function(_failed_pv_reinference_app, default_timeout=10).run()
    app.button[0].click().run()
    state = app.session_state.filtered_state

    assert not app.exception
    assert any("PV checkpoint failed" in value.value for value in app.error)
    assert "scenario" not in state
    assert state["detections"] == []
    assert state["pv_inference_run"] is False
    assert state["selected_pv_mask_ids"] == []
    assert state["obstacle_detections"]
    assert state["obstacle_inference_run"] is True


def test_same_count_successful_pv_reinference_replaces_masks_and_preserves_obstacles():
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_function(_successful_same_count_pv_reinference_app, default_timeout=10).run()
    app.button[0].click().run()
    state = app.session_state.filtered_state

    assert not app.exception
    assert "scenario" not in state
    assert state["detections"][0]["polygon_pixels"] == ((1.0, 1.0), (8.0, 1.0), (8.0, 8.0), (1.0, 8.0))
    assert state["pv_inference_run"] is True
    assert state["obstacle_detections"]
    assert state["obstacle_inference_run"] is True


def test_streamlit_manual_mode_runs_a_mocked_cached_pvgis_scenario():
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_function(_mocked_manual_app, default_timeout=20).run()
    app.radio[0].set_value("Manuell").run()
    for widget, value in zip(app.number_input[:3], [100.0, 47.483116, 8.207111]):
        widget.set_value(value)
    app.button[0].click().run()
    app.button[1].click().run()

    assert not app.exception
    assert not app.error
    assert any(metric.label == "Jahresenergie" for metric in app.metric)
    assert any("Szenario berechnet" in value.value for value in app.success)
    assert any("PVGIS 5.3 TMY" in value.value for value in app.caption)
    app.number_input[0].set_value(200.0)
    app.button[0].click().run()
    assert not app.exception
    assert "scenario" not in app.session_state.filtered_state


def test_failed_recalculation_does_not_keep_previous_scenario(app_module, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_function(_mocked_manual_app, default_timeout=20).run()
    app.radio[0].set_value("Manuell").run()
    for widget, value in zip(app.number_input[:3], [100.0, 47.483116, 8.207111]):
        widget.set_value(value)
    app.button[0].click().run()
    app.button[1].click().run()
    assert "scenario" in app.session_state.filtered_state

    def fail(*args, **kwargs):
        raise ValueError("weather unavailable")

    monkeypatch.setattr(app_module, "simulate_pv", fail)
    app.button[1].click().run()
    assert not app.exception
    assert any("weather unavailable" in error.value for error in app.error)
    assert "scenario" not in app.session_state.filtered_state
