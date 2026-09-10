import numpy as np
import pytest
from PIL import Image
from types import SimpleNamespace

from rooftop_pv import inference
from rooftop_pv.inference import Detection, pixel_to_map, union_mask


class _Values:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)


class _FakeModel:
    names = {0: "solar_panel"}

    def __init__(self, result):
        self.result = result
        self.calls = []

    def predict(self, image, **kwargs):
        self.calls.append((image.size, kwargs))
        return [self.result]


def test_map_transform_is_north_up_and_uses_original_dimensions():
    geometry = pixel_to_map([(0, 0), (100, 0), (100, 50), (0, 50)],
                            (2600000, 1200000, 2600020, 1200010), 100, 50)
    assert geometry.area == pytest.approx(200)
    assert geometry.exterior.coords[0] == (2600000, 1200010)


def test_reject_bad_georeference():
    with pytest.raises(ValueError):
        pixel_to_map([(0, 0), (101, 0), (100, 50)], (0, 0, 20, 10), 100, 50)


def test_union_overlaps_are_not_added_twice():
    detection = Detection("solar_panel", .9, ((1, 1), (4, 1), (4, 4), (1, 4)))
    once = union_mask([detection], 10, 10)
    twice = union_mask([detection, detection], 10, 10)
    assert np.array_equal(once, twice)


def test_predict_keeps_disconnected_mask_islands_separate(monkeypatch):
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:6, 2:6] = 1
    mask[12:16, 12:16] = 1
    # This is the kind of merged contour returned by Ultralytics' masks.xy;
    # extraction must use masks.data instead and never create this bridge.
    merged_xy = [np.asarray([(2, 2), (16, 2), (16, 16), (2, 16)], dtype=np.float32)]
    result = SimpleNamespace(
        masks=SimpleNamespace(data=mask[None, ...], xy=merged_xy),
        boxes=SimpleNamespace(cls=_Values([0]), conf=_Values([.87])),
    )
    model = _FakeModel(result)

    detections = inference.predict(Image.new("RGB", (20, 20)), model=model, device="cpu")

    assert len(detections) == 2
    assert all(detection.class_name == "solar_panel" for detection in detections)
    assert all(detection.confidence == pytest.approx(.87) for detection in detections)
    raster = union_mask(detections, 20, 20)
    assert raster[3, 3]
    assert raster[13, 13]
    assert not raster[9, 9]
    assert all(max(x for x, _ in detection.polygon) <= 16 for detection in detections)
    assert model.calls[0][1]["retina_masks"] is True


def test_predict_rejects_mask_dimensions_that_are_not_original_image_pixels():
    result = SimpleNamespace(
        masks=SimpleNamespace(data=np.ones((1, 8, 8), dtype=np.uint8), xy=[]),
        boxes=SimpleNamespace(cls=_Values([0]), conf=_Values([.9])),
    )
    model = _FakeModel(result)

    with pytest.raises(ValueError, match="original image dimensions"):
        inference.predict(Image.new("RGB", (20, 20)), model=model, device="cpu")


def test_tiled_obstacles_keep_training_scale_and_original_pixel_coordinates(monkeypatch):
    crops = []

    def fake_predict(image, **kwargs):
        crops.append(image.size)
        return [Detection("chimney", .8, ((1, 1), (10, 1), (10, 10))),
                Detection("solar_panel", .9, ((1, 1), (10, 1), (10, 10)))]

    monkeypatch.setattr(inference, "predict", fake_predict)
    predictions = inference.predict_tiled_obstacles(
        Image.new("RGB", (1000, 1000)), bbox=(2600000, 1200000, 2600100, 1200100),
        model=object(), device="cpu",
    )
    assert len(crops) == 9
    assert set(crops) == {(410, 410)}
    assert len(predictions) == 9
    assert all(prediction.class_name == "chimney" for prediction in predictions)
    assert max(p.polygon[0][0] for p in predictions) == 591
    assert max(p.polygon[0][1] for p in predictions) == 591


def test_tiled_obstacles_reject_unbounded_work_and_invalid_georeference():
    image = Image.new("RGB", (1000, 1000))
    with pytest.raises(ValueError, match="tiles"):
        inference.predict_tiled_obstacles(image, bbox=(0, 0, 500, 500), model=object())
    with pytest.raises(ValueError, match="bounds"):
        inference.predict_tiled_obstacles(image, bbox=(0, 0, 0, 100), model=object())


def test_tiled_pv_uses_100m_context_and_filters_other_classes(monkeypatch):
    crops = []

    def fake_predict(image, **kwargs):
        crops.append(image.size)
        return [Detection("solar_panel", .9, ((1, 1), (10, 1), (10, 10))),
                Detection("chimney", .8, ((1, 1), (10, 1), (10, 10)))]

    monkeypatch.setattr(inference, "predict", fake_predict)
    predictions = inference.predict_tiled_pv(
        Image.new("RGB", (1000, 1000)), bbox=(2600000, 1200000, 2600100, 1200100),
        model=object(), device="cpu",
    )

    assert crops == [(1000, 1000)]
    assert len(predictions) == 1
    assert predictions[0].class_name == "solar_panel"
    assert predictions[0].polygon[0] == (1, 1)


def test_tiled_pv_offsets_overlapping_crops_in_original_image_bounds(monkeypatch):
    crop_origins = []

    def fake_predict(image, **kwargs):
        crop_origins.append(image.size)
        width, height = image.size
        return [Detection("solar_panel", .9,
                          ((1, 1), (width - 1, 1), (width - 1, height - 1)))]

    monkeypatch.setattr(inference, "predict", fake_predict)
    predictions = inference.predict_tiled_pv(
        Image.new("RGB", (1200, 1000)), bbox=(2600000, 1200000, 2600120, 1200100),
        model=object(), device="cpu",
    )

    assert crop_origins == [(1000, 1000), (1000, 1000)]
    assert len(predictions) == 2
    assert max(x for detection in predictions for x, _ in detection.polygon) <= 1200
    assert max(y for detection in predictions for _, y in detection.polygon) <= 1000
    assert min(x for detection in predictions for x, _ in detection.polygon) >= 0
    assert min(y for detection in predictions for _, y in detection.polygon) >= 0
    assert max(x for x, _ in predictions[1].polygon) > max(x for x, _ in predictions[0].polygon)


def test_tiled_pv_rejects_unbounded_work_and_invalid_georeference():
    image = Image.new("RGB", (1000, 1000))
    with pytest.raises(ValueError, match="tiles"):
        inference.predict_tiled_pv(image, bbox=(0, 0, 1000, 1000), model=object())
    with pytest.raises(ValueError, match="bounds"):
        inference.predict_tiled_pv(image, bbox=(0, 0, 0, 100), model=object())
