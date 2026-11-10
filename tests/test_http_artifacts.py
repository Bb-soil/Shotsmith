from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest

from anime_studio.artifacts import HttpArtifactStore, validate_mp4_structure
from anime_studio.errors import ArtifactError
from tests.comfy_stub import MP4_FIXTURE, StubState, run_stub


def _store(tmp_path: Path, base_url: str, *, timeout: float = 2.0, max_bytes: int = 1024 * 1024) -> HttpArtifactStore:
    return HttpArtifactStore(
        tmp_path / "artifacts",
        base_url=base_url,
        timeout_seconds=timeout,
        max_bytes=max_bytes,
    )


def _ref(**overrides):
    value = {
        "scheme": "comfy-http",
        "filename": "result.mp4",
        "subfolder": "",
        "type": "output",
        "media_type": "video/mp4",
    }
    value.update(overrides)
    return value


def test_valid_structural_mp4_is_materialized_with_hash_and_no_mock_marker(tmp_path) -> None:
    with run_stub() as (base_url, _state):
        store = _store(tmp_path, base_url)
        record = store.fetch_result(_ref(), "job-1")
        assert record["mock"] is False
        assert record["kind"] == "video"
        assert record["media_type"] == "video/mp4"
        assert record["relative_path"] == "job-1/result.mp4"
        path = Path(record["path"])
        assert path.read_bytes() == MP4_FIXTURE
        assert record["sha256"] == hashlib.sha256(MP4_FIXTURE).hexdigest()
        assert record["size_bytes"] == len(MP4_FIXTURE)
        assert not list(path.parent.glob("*.tmp"))
        assert "result.mock.json" not in str(path)


@pytest.mark.parametrize(
    "filename",
    [
        "../result.mp4",
        "..\\result.mp4",
        "C:\\temp\\result.mp4",
        "bad\x00.mp4",
        "bad\u202e.mp4",
        "folder/result.mp4",
    ],
)
def test_unsafe_filenames_are_rejected(tmp_path, filename) -> None:
    with run_stub() as (base_url, _state):
        store = _store(tmp_path, base_url)
        with pytest.raises(ArtifactError):
            store.fetch_result(_ref(filename=filename), "job-2")


def test_unsafe_subfolder_and_type_are_rejected(tmp_path) -> None:
    with run_stub() as (base_url, _state):
        store = _store(tmp_path, base_url)
        with pytest.raises(ArtifactError):
            store.fetch_result(_ref(subfolder="../outside"), "job-3")
        with pytest.raises(ArtifactError):
            store.fetch_result(_ref(type="input"), "job-3")
        with pytest.raises(ArtifactError):
            store.fetch_result(_ref(scheme="mock", content={}), "job-3")


def test_redirect_is_rejected(tmp_path) -> None:
    with run_stub() as (base_url, state):
        state.view_redirect = "https://example.invalid/other"
        store = _store(tmp_path, base_url)
        with pytest.raises(ArtifactError) as exc_info:
            store.fetch_result(_ref(), "job-4")
        assert exc_info.value.code == "ARTIFACT_REDIRECT_REJECTED"


def test_oversized_declared_and_absent_content_length(tmp_path) -> None:
    with run_stub() as (base_url, state):
        state.view_body = b"x" * 200
        state.view_content_length = "1000000"
        store = _store(tmp_path, base_url, max_bytes=100)
        with pytest.raises(ArtifactError) as exc_info:
            store.fetch_result(_ref(), "job-5")
        assert exc_info.value.code == "ARTIFACT_TOO_LARGE"

    with run_stub() as (base_url, state):
        state.view_body = b"x" * 200
        state.view_content_length = "OMIT"
        store = _store(tmp_path, base_url, max_bytes=100)
        with pytest.raises(ArtifactError) as exc_info:
            store.fetch_result(_ref(), "job-6")
        assert exc_info.value.code == "ARTIFACT_TOO_LARGE"


def test_total_timeout_and_truncated_stream_cleanup(tmp_path) -> None:
    with run_stub() as (base_url, state):
        state.view_delay_seconds = 0.5
        store = _store(tmp_path, base_url, timeout=0.1)
        with pytest.raises(ArtifactError):
            store.fetch_result(_ref(), "job-7")
        assert not list((tmp_path / "artifacts").rglob("*.tmp"))

    with run_stub() as (base_url, state):
        state.view_content_length = str(len(MP4_FIXTURE) + 100)
        store = _store(tmp_path, base_url)
        with pytest.raises(ArtifactError):
            store.fetch_result(_ref(), "job-8")
        assert not list((tmp_path / "artifacts").rglob("*.tmp"))


def test_invalid_media_and_empty_body_are_rejected(tmp_path) -> None:
    with run_stub() as (base_url, state):
        state.view_body = b"not-an-mp4"
        store = _store(tmp_path, base_url)
        with pytest.raises(ArtifactError) as exc_info:
            store.fetch_result(_ref(), "job-9")
        assert exc_info.value.code == "ARTIFACT_INVALID_MEDIA"

    with run_stub() as (base_url, state):
        state.view_body = b""
        store = _store(tmp_path, base_url)
        with pytest.raises(ArtifactError) as exc_info:
            store.fetch_result(_ref(), "job-10")
        assert exc_info.value.code == "ARTIFACT_EMPTY"


def test_failed_download_does_not_replace_existing_target(tmp_path) -> None:
    with run_stub() as (base_url, state):
        state.view_redirect = "https://example.invalid/other"
        store = _store(tmp_path, base_url)
        target = tmp_path / "artifacts" / "job-11" / "result.mp4"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"old")
        with pytest.raises(ArtifactError):
            store.fetch_result(_ref(), "job-11")
        assert target.read_bytes() == b"old"
        assert not list(target.parent.glob("*.tmp"))


def test_download_stall_near_budget_is_bounded_and_cleaned(tmp_path) -> None:
    with run_stub() as (base_url, state):
        chunks = [MP4_FIXTURE[:20], MP4_FIXTURE[20:40], MP4_FIXTURE[40:]]
        state.view_chunks = chunks
        state.view_chunk_delay_seconds = 0.15
        store = _store(tmp_path, base_url, timeout=0.25)
        started = time.monotonic()
        with pytest.raises(ArtifactError):
            store.fetch_result(_ref(), "job-stall")
        elapsed = time.monotonic() - started
        assert elapsed < 1.5
        assert not list((tmp_path / "artifacts").rglob("*.tmp"))


def test_structural_mp4_requires_ftyp_moov_and_mdat(tmp_path) -> None:
    valid = tmp_path / "valid.mp4"
    valid.write_bytes(MP4_FIXTURE)
    validate_mp4_structure(valid)

    no_moov = tmp_path / "no-moov.mp4"
    no_moov.write_bytes(MP4_FIXTURE.replace(b"moov", b"free"))
    with pytest.raises(ArtifactError):
        validate_mp4_structure(no_moov)

    truncated = tmp_path / "truncated.mp4"
    truncated.write_bytes(MP4_FIXTURE[:12])
    with pytest.raises(ArtifactError):
        validate_mp4_structure(truncated)

    overflow = tmp_path / "overflow.mp4"
    overflow.write_bytes((1000).to_bytes(4, "big") + b"ftyp" + b"isom")
    with pytest.raises(ArtifactError):
        validate_mp4_structure(overflow)
