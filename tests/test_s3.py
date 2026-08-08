from pathlib import Path

import pytest

from llmtrain.s3 import is_s3_uri, resolve_local_path, sibling_path


def test_is_s3_uri_detects_scheme():
    assert is_s3_uri("s3://my-bucket/checkpoints/step_10.pt")
    assert not is_s3_uri("/workspace/checkpoints/step_10.pt")
    assert not is_s3_uri("checkpoints/step_10.pt")


def test_sibling_path_for_local_path_matches_pathlib_parent_behavior():
    assert sibling_path("/workspace/checkpoints/step_10.pt", "tokenizer.json") == str(
        Path("/workspace/checkpoints/tokenizer.json")
    )


def test_sibling_path_for_s3_uri_stays_in_the_same_prefix():
    assert (
        sibling_path("s3://my-bucket/checkpoints/step_10.pt", "tokenizer.json")
        == "s3://my-bucket/checkpoints/tokenizer.json"
    )


def test_resolve_local_path_passes_through_non_s3_paths_unchanged(tmp_path):
    local_file = tmp_path / "step_10.pt"
    local_file.write_bytes(b"fake checkpoint")

    resolved = resolve_local_path(str(local_file))

    assert resolved == local_file


def test_resolve_local_path_downloads_s3_uri_to_cache_dir(tmp_path, monkeypatch):
    downloaded = []

    class FakeS3Client:
        def download_file(self, bucket, key, dest):
            downloaded.append((bucket, key, dest))
            Path(dest).write_bytes(b"fake checkpoint from s3")

    monkeypatch.setattr("boto3.client", lambda service_name: FakeS3Client())

    resolved = resolve_local_path("s3://my-bucket/checkpoints/step_10.pt", cache_dir=tmp_path)

    assert resolved == tmp_path / "my-bucket" / "checkpoints" / "step_10.pt"
    assert resolved.read_bytes() == b"fake checkpoint from s3"
    assert downloaded == [("my-bucket", "checkpoints/step_10.pt", str(resolved))]


def test_resolve_local_path_skips_download_when_already_cached(tmp_path, monkeypatch):
    cached = tmp_path / "my-bucket" / "checkpoints" / "step_10.pt"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"already here")

    def fail_if_called(service_name):
        raise AssertionError("boto3.client should not be called when already cached")

    monkeypatch.setattr("boto3.client", fail_if_called)

    resolved = resolve_local_path("s3://my-bucket/checkpoints/step_10.pt", cache_dir=tmp_path)

    assert resolved == cached
    assert resolved.read_bytes() == b"already here"


def test_resolve_local_path_raises_helpful_error_without_boto3(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "boto3", None)

    with pytest.raises(ImportError, match="s3"):
        resolve_local_path("s3://my-bucket/checkpoints/step_10.pt", cache_dir=tmp_path)
