import json

from PIL import Image
from shapely.geometry import box

from scripts import evaluate_reference_roofs as evaluator


def detection(label, geometry, confidence, model):
    return {
        "label": label,
        "kind": label,
        "geometry": geometry,
        "confidence": confidence,
        "model": model,
        "source": model,
        "sources": [model],
    }


def test_summarise_image_reports_union_largest_count_confidence_and_overlap():
    raw = [
        detection("pv_installation", box(0, 0, 4, 4), 0.91, "swiss"),
        detection("skylight", box(2, 0, 6, 2), 0.61, "rid"),
        detection("skylight", box(5, 0, 7, 2), 0.72, "rid"),
    ]

    summary = evaluator.summarise_image(raw, raw, (10, 10))

    assert summary["classes"]["skylight"] == {
        "mask_count": 2,
        "total_mask_area_pixels": 10.0,
        "total_mask_area_fraction": 0.10,
        "largest_mask_area_pixels": 8.0,
        "largest_mask_area_fraction": 0.08,
        "max_confidence": 0.72,
        "sources": ["rid"],
    }
    assert summary["overlap"]["raw_swiss_pv_vs_rid_obstacles_pixels"] == 4.0
    assert summary["overlap"]["raw_swiss_pv_vs_rid_obstacles_fraction"] == 0.04
    assert summary["overlap"]["raw_overlap_relative_to_swiss_pv"] == 0.25


def test_infer_reference_image_uses_app_predict_and_restores_model_override(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.touch()
    calls = []

    def fake_predict(image, bounds, confidence, obstacle_confidence):
        calls.append(
            {
                "size": image.size,
                "bounds": bounds,
                "confidence": confidence,
                "obstacle_confidence": obstacle_confidence,
                "checkpoint": evaluator.app_inference.checkpoint_path("rid"),
            }
        )
        return [detection("pv_installation", box(0, 0, 2, 2), 0.9, "swiss")]

    monkeypatch.setattr(evaluator.app_inference, "predict", fake_predict)
    monkeypatch.setenv("OBSTACLE_MODEL_PATH", "keep-this-value.pt")

    raw, displayed = evaluator.infer_reference_image(
        Image.new("RGB", (8, 6)), checkpoint, 0.25, 0.30
    )

    assert calls == [
        {
            "size": (8, 6),
            "bounds": (0.0, 0.0, 8.0, 6.0),
            "confidence": 0.25,
            "obstacle_confidence": 0.30,
            "checkpoint": checkpoint.resolve(),
        }
    ]
    assert raw[0]["model"] == "swiss"
    assert displayed[0]["blocks_placement"] is True
    assert evaluator.os.environ["OBSTACLE_MODEL_PATH"] == "keep-this-value.pt"


def test_promotion_checks_reject_growth_disappearance_and_failed_six_roof_gate():
    production = {
        "pv_reference": {
            "classes": {
                "pv_installation": class_metrics(1, 0.014, "swiss"),
                "skylight": class_metrics(2, 0.001, "rid"),
                "other_obstacle": class_metrics(1, 0.016, "rid"),
            }
        },
        "small_objects": {
            "classes": {"chimney": class_metrics(1, 0.002, "rid")}
        },
    }
    candidate = {
        "pv_reference": {
            "classes": {
                "pv_installation": class_metrics(1, 0.014, "swiss"),
                "other_obstacle": class_metrics(1, 0.017, "rid"),
            }
        },
        "small_objects": {"classes": {}},
    }

    result = evaluator.promotion_checks(production, candidate, six_roof_passed=False)

    assert result["checks"]["swiss_pv_present_on_pv_reference"] is True
    assert result["checks"]["large_rid_obstacle_masks_do_not_grow"] is False
    assert result["checks"]["small_skylights_and_chimneys_do_not_disappear"] is False
    assert result["checks"]["six_roof_gate_passes"] is False
    assert result["passed"] is False
    assert result["violations"]["large_rid_obstacle_masks_do_not_grow"]
    assert result["violations"]["small_skylights_and_chimneys_do_not_disappear"]


def test_promotion_checks_accept_a_candidate_that_preserves_reference_behavior():
    production = {
        "pv_reference": {
            "classes": {
                "pv_installation": class_metrics(1, 0.014, "swiss"),
                "skylight": class_metrics(1, 0.001, "rid"),
                "other_obstacle": class_metrics(1, 0.016, "rid"),
            }
        }
    }
    candidate = {
        "pv_reference": {
            "classes": {
                "pv_installation": class_metrics(1, 0.014, "swiss"),
                "skylight": class_metrics(1, 0.0008, "rid"),
                "other_obstacle": class_metrics(1, 0.015, "rid"),
            }
        }
    }

    result = evaluator.promotion_checks(production, candidate, six_roof_passed=True)

    assert result["passed"] is True
    assert all(result["checks"].values())


def test_render_overlay_writes_a_traceable_image(tmp_path):
    output = tmp_path / "overlay.png"
    image = Image.new("RGB", (20, 20), "white")
    detections = [
        detection("pv_installation", box(2, 2, 8, 8), 0.9, "swiss"),
        detection("chimney", box(10, 10, 14, 14), 0.6, "rid"),
    ]

    evaluator.render_overlay(image, detections, output, "candidate / roof")

    assert output.is_file()
    with Image.open(output) as rendered:
        assert rendered.size == image.size
        assert rendered.getpixel((3, 16)) != (255, 255, 255)
        assert rendered.getpixel((11, 8)) != (255, 255, 255)


def test_load_six_roof_gate_requires_production_thresholds_and_quality_pass(tmp_path):
    path = tmp_path / "six-roof.json"
    path.write_text(
        json.dumps(
            {
                "parameters": {"confidence": 0.25, "obstacle_confidence": 0.30},
                "quality_gates": {"passed": True, "checks": {"example": True}},
            }
        ),
        encoding="utf-8",
    )

    loaded = evaluator.load_six_roof_gate(path)

    assert loaded["passed"] is True
    assert loaded["source"] == str(path.resolve())
    assert len(loaded["sha256"]) == 64


def class_metrics(count, largest, source):
    return {
        "mask_count": count,
        "total_mask_area_pixels": largest * 100,
        "total_mask_area_fraction": largest,
        "largest_mask_area_pixels": largest * 100,
        "largest_mask_area_fraction": largest,
        "max_confidence": 0.9,
        "sources": [source],
    }
