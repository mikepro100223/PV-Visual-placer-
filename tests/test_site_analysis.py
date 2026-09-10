import io
import json
from dataclasses import asdict

import pytest
from PIL import Image
from shapely.geometry import box, shape
from shapely.ops import unary_union

from rooftop_pv import site_analysis as site
from rooftop_pv.inference import Detection
from rooftop_pv.sources import OrthophotoResult, RoofFeature


def test_tile_plan_covers_every_target_with_training_scale_and_bounded_work():
    bounds = (2500000.03, 1118000.07, 2500213.8, 1118115.3)
    frame, size, tiles = site.plan_tiles(bounds)
    union = unary_union([box(*tile) for tile in tiles])
    assert union.covers(box(*bounds))
    assert union.area == pytest.approx(box(*frame).area)
    assert size[0] == pytest.approx((frame[2] - frame[0]) / .1)
    assert size[1] == pytest.approx((frame[3] - frame[1]) / .1)
    assert len(tiles) > 1
    assert all(tile[2] - tile[0] == pytest.approx(100) for tile in tiles)
    with pytest.raises(ValueError, match="64"):
        site.plan_tiles((2500000, 1118000, 2502000, 1120000))


def test_tile_coordinates_bind_to_the_complete_north_up_frame():
    d = Detection("chimney", .8, ((0, 0), (100, 0), (100, 100)))
    result = site.shift_detection(d, (2500080, 1118000, 2500180, 1118100),
                                  (2500000, 1118000, 2500180, 1118180))
    assert result.polygon == ((800, 800), (900, 800), (900, 900))
    assert result.confidence == d.confidence


def _fake_inputs(monkeypatch, tmp_path):
    roof = RoofFeature(7, box(2499980, 1118010, 2500070, 1118060).__geo_interface__,
                       {"flaeche": 4500, "neigung": 0, "ausrichtung": 0}, "roof-url", None, "today")
    second = RoofFeature(8, box(2500080, 1118020, 2500090, 1118030).__geo_interface__,
                         {"flaeche": 100, "neigung": 0, "ausrichtung": 0}, "roof-url", None, "today")
    monkeypatch.setattr(site.sources, "fetch_roofs_bbox", lambda bbox: [roof, second], raising=False)
    checkpoints = {}
    for role in ("pv", "obstacles"):
        path = tmp_path / f"{role}.pt"
        path.write_bytes(role.encode())
        checkpoints[role] = {"weights": str(path), "weights_sha256": site.sha256(path),
                             "quality_status": "experimental_gates_failed"}
    monkeypatch.setattr(site, "registered_models", lambda: checkpoints)
    monkeypatch.setattr(site, "load_segmenter", lambda path: object())
    image = io.BytesIO()
    Image.new("RGB", (1000, 1000), "gray").save(image, format="JPEG")

    def photo(bbox, width=1000, height=1000):
        return OrthophotoResult(image.getvalue(), "image/jpeg", bbox, width, height,
                                (.1, .1), True, "swissimage-url", "cache-date", "today")

    monkeypatch.setattr(site.sources, "fetch_orthophoto", photo)
    monkeypatch.setattr(site, "predict_tiled_pv", lambda *args, **kwargs: [])
    monkeypatch.setattr(site, "predict_tiled_obstacles", lambda *args, **kwargs: [
        Detection("chimney", .8, ((100, 100), (150, 100), (150, 150), (100, 150)))])
    return [roof, second]


def test_live_pipeline_contract_with_real_files_and_stubbed_remote_models(tmp_path, monkeypatch):
    roofs = _fake_inputs(monkeypatch, tmp_path)
    output = tmp_path / "result"
    report = site.run_site_analysis((2500000, 1118000, 2500100, 1118100), output)
    assert report["roof_count"] == 2
    assert report["target_roof_ids"] == ["7", "8"]
    assert report["complete_roof_count"] == 2
    assert report["tile_count"] > 1
    assert report["accuracy_validated"] is False
    assert set(report["models"]) == {"pv", "obstacles"}
    assert (output / "roofs.csv").is_file()
    assert (output / "usable-roofs-lv95.geojson").is_file()
    assert (output / "overview.jpg").is_file()
    persisted = json.loads((output / "report.json").read_text())
    assert persisted["roof_count"] == 2
    saved_sources = json.loads((output / "sources.json").read_text())
    assert saved_sources["roofs"] == json.loads(json.dumps([asdict(roof) for roof in roofs]))
    collection = json.loads((output / "usable-roofs-lv95.geojson").read_text())
    assert len(collection["features"]) == 2
    for feature in collection["features"]:
        if feature["geometry"]:
            assert shape(feature["geometry"]).area >= 0
    with pytest.raises(FileExistsError):
        site.run_site_analysis((2500000, 1118000, 2500100, 1118100), output)


def test_empty_roof_query_stays_explicit_without_running_models(tmp_path, monkeypatch):
    monkeypatch.setattr(site.sources, "fetch_roofs_bbox", lambda bbox: [], raising=False)
    monkeypatch.setattr(site, "registered_models", lambda: pytest.fail("No model load for empty area"))
    with pytest.raises(ValueError, match="No Sonnendach"):
        site.run_site_analysis((2500000, 1118000, 2500100, 1118100), tmp_path / "empty")


def test_registry_rejects_mutated_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(site, "ROOT", tmp_path)
    index = tmp_path / "artifacts/models"
    index.mkdir(parents=True)
    path = tmp_path / "weights.pt"
    path.write_bytes(b"changed")
    (index / "current.json").write_text(json.dumps({"best_weights": str(path),
        "best_weights_sha256": "not-current", "classes": {"0": "solar_panel"}}))
    with pytest.raises(ValueError, match="hash"):
        site.registered_models()


def test_saved_analysis_replays_geometry_without_models_or_network(tmp_path, monkeypatch):
    _fake_inputs(monkeypatch, tmp_path)
    source, output = tmp_path / "original", tmp_path / "replay"
    original = site.run_site_analysis((2500000, 1118000, 2500100, 1118100), source)
    monkeypatch.setattr(site, "load_segmenter", lambda *args: pytest.fail("Replay must not run models"))
    monkeypatch.setattr(site.sources, "fetch_roofs_bbox", lambda *args: pytest.fail("No source queries"))
    replay = site.recalculate_site_analysis(source, output)
    assert replay["roof_count"] == original["roof_count"]
    assert replay["models"] == original["models"]
    assert replay["replay_source"] == str(source.resolve())
    assert (output / "roofs.csv").read_text() == (source / "roofs.csv").read_text()
    assert (output / "usable-roofs-lv95.geojson").is_file()
    assert replay["replay_input_sha256"]["predictions.json"] == site.sha256(source / "predictions.json")
    with pytest.raises(FileExistsError):
        site.recalculate_site_analysis(source, output)
