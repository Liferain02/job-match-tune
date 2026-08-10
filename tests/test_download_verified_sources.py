import hashlib

import pytest

from jobmatch_tune.dataset.download_verified_sources import verify_or_download_source


def _source(source_path, target_path, digest):
    return {
        "name": "example",
        "local_path": str(target_path),
        "source_url": source_path.as_uri(),
        "artifact_sha256": digest,
    }


def test_verify_or_download_source_downloads_and_reuses_verified_file(tmp_path):
    source_path = tmp_path / "source.bin"
    target_path = tmp_path / "downloads" / "target.bin"
    content = b"fixed artifact"
    source_path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()

    downloaded = verify_or_download_source(_source(source_path, target_path, digest))
    cached = verify_or_download_source(_source(source_path, target_path, digest), verify_only=True)

    assert downloaded["status"] == "downloaded"
    assert cached["status"] == "verified_cached"
    assert target_path.read_bytes() == content


def test_verify_or_download_source_rejects_checksum_mismatch(tmp_path):
    source_path = tmp_path / "source.bin"
    target_path = tmp_path / "target.bin"
    source_path.write_bytes(b"source")
    target_path.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_or_download_source(_source(source_path, target_path, "a" * 64))
