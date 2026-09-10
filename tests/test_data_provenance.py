import io
import json
import zipfile

import pytest

from rooftop_pv import data


class Response:
    def __init__(self, *, content=b"", document=None):
        self.content, self.document = content, document

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        yield self.content[:10]
        yield b""
        yield self.content[10:]

    def json(self):
        return self.document


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.urls = []

    def get(self, url, **kwargs):
        self.urls.append(url)
        return next(self.responses)


def source_metadata(**overrides):
    return dict(ref=data.DATASET_REF, licenseName=data.DATASET_LICENSE,
                versions=[{"versionNumber": 1}], **overrides)


def zip_bytes(member="images/example.jpg"):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(member, b"fixture")
    return output.getvalue()


def test_download_pins_version_records_hash_and_reuses_verified_cache(tmp_path):
    session = Session([Response(content=zip_bytes()),
                       Response(document=source_metadata(token="must not persist"))])
    archive = data.download_dataset(tmp_path, session=session)
    metadata = json.loads((tmp_path / "source-metadata.json").read_text())
    assert all("datasetVersionNumber=1" in url for url in session.urls)
    assert metadata["archive_sha256"] == data.sha256_file(archive)
    assert metadata["license"] == data.DATASET_LICENSE
    assert "must not persist" not in json.dumps(metadata)
    no_network = Session([])
    assert data.download_dataset(tmp_path, session=no_network) == archive
    assert no_network.urls == []


@pytest.mark.parametrize("field,value", [("ref", "different/source"),
                                         ("licenseName", "unknown"), ("versions", [])])
def test_download_rejects_changed_provenance(tmp_path, field, value):
    metadata = source_metadata()
    metadata[field] = value
    session = Session([Response(content=zip_bytes()), Response(document=metadata)])
    with pytest.raises(ValueError):
        data.download_dataset(tmp_path, session=session)
    assert not (tmp_path / "source-metadata.json").exists()


def test_download_rejects_unsafe_zip_without_publishing_it(tmp_path):
    session = Session([Response(content=zip_bytes("../../outside"))])
    with pytest.raises(ValueError, match="unsafe"):
        data.download_dataset(tmp_path, session=session)
    assert not (tmp_path / data.DEFAULT_ARCHIVE_NAME).exists()
    assert not list(tmp_path.glob("*.part"))


def test_cached_metadata_must_match_actual_archive(tmp_path):
    archive = tmp_path / data.DEFAULT_ARCHIVE_NAME
    archive.write_bytes(zip_bytes())
    metadata = data._metadata_document(source_metadata(), archive)
    metadata["archive_sha256"] = "wrong"
    (tmp_path / "source-metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="hash"):
        data.download_dataset(tmp_path, session=Session([]))
    with pytest.raises(ValueError, match="hash"):
        data._load_source_metadata(tmp_path, archive)


def test_unverified_local_archive_does_not_inherit_source_license(tmp_path):
    archive = tmp_path / "local.zip"
    archive.write_bytes(zip_bytes())
    metadata = data._load_source_metadata(tmp_path, archive)
    assert metadata["license"] is None
    assert metadata["provenance_status"].startswith("unverified")
    assert metadata["archive_sha256"] == data.sha256_file(archive)


@pytest.mark.parametrize("ratios", [(1, 0, 0), (.8, .2, .1), (float("nan"), .1, .1)])
def test_invalid_split_ratios_are_rejected(ratios):
    with pytest.raises(ValueError, match="ratios"):
        data.build_split_manifest([], ratios=ratios)


def test_too_few_geographic_groups_and_duplicates_are_rejected():
    names = [f"swissimage-dop10_2021_2575.{n}-1206.0.jpg" for n in range(3)]
    with pytest.raises(ValueError, match="geographic groups"):
        data.build_split_manifest(names)
    with pytest.raises(ValueError, match="unique"):
        data.build_split_manifest([names[0]] * 3)
