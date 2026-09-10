import pytest
import yaml

from rooftop_pv.training import dataset_paths, load_config


def test_reject_detection_model_config(tmp_path):
    config = tmp_path / "train.yaml"
    config.write_text(yaml.safe_dump(dict(model="yolo11n.pt", epochs=5, imgsz=320, batch=4)))
    with pytest.raises(ValueError, match="segmentation"):
        load_config(config)


def test_image_dimensions_must_match_stride(tmp_path):
    config = tmp_path / "train.yaml"
    config.write_text(yaml.safe_dump(dict(model="yolo11n-seg.pt", epochs=5, imgsz=321, batch=4)))
    with pytest.raises(ValueError, match="divisible"):
        load_config(config)


def test_split_rejects_missing_images(tmp_path):
    config = tmp_path / "data.yaml"
    config.write_text(yaml.safe_dump(dict(path=str(tmp_path), test="missing")))
    with pytest.raises(ValueError, match="not a local"):
        dataset_paths(config, "test")


def test_every_dataset_entry_point_rejects_executable_download_yaml(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "example.jpg").write_bytes(b"fixture")
    config = tmp_path / "data.yaml"
    config.write_text(yaml.safe_dump(dict(test="images", download="print('untrusted')")))
    with pytest.raises(ValueError, match="download"):
        dataset_paths(config, "test")


def test_split_rejects_non_mapping_yaml(tmp_path):
    config = tmp_path / "data.yaml"
    config.write_text("- invalid\n")
    with pytest.raises(ValueError, match="mapping"):
        dataset_paths(config, "test")
