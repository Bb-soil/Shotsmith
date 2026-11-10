from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.p1 import setup_plan


class Response:
    status = 200
    headers: dict[str, str] = {}

    def __init__(self, body: bytes) -> None:
        self.body = body
        self.pos = 0
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        if self.pos >= len(self.body):
            return b""
        if size < 0:
            size = len(self.body) - self.pos
        chunk = self.body[self.pos : self.pos + size]
        self.pos += len(chunk)
        return chunk

    def close(self) -> None:
        self.closed = True


def _spec(body: bytes) -> setup_plan.DownloadSpec:
    return setup_plan.DownloadSpec(
        model_id="tiny",
        url="https://example.invalid/tiny.bin",
        target="models/tiny.bin",
        size_bytes=len(body),
        sha256=hashlib.sha256(body).hexdigest(),
    )


def test_cancel_before_open_keeps_part_and_never_finalizes(tmp_path: Path) -> None:
    body = b"0123456789"
    spec = _spec(body)
    root = tmp_path / "ComfyUI"
    target = root / spec.target
    target.parent.mkdir(parents=True)
    part = target.with_name(target.name + ".part")
    part.write_bytes(b"0123")

    def opener(url, headers):
        raise AssertionError("cancel must be observed before the network request")

    with pytest.raises(setup_plan.DownloadCancelled):
        setup_plan.download_verified(spec, root=root, opener=opener, cancel=lambda: True)
    assert part.read_bytes() == b"0123"
    assert not target.exists()


def test_cancel_during_complete_part_hash_keeps_part(tmp_path: Path) -> None:
    body = b"x" * (3 * 1024 * 1024 + 1)
    spec = _spec(body)
    root = tmp_path / "ComfyUI"
    target = root / spec.target
    target.parent.mkdir(parents=True)
    part = target.with_name(target.name + ".part")
    part.write_bytes(body)
    calls = [0]

    def cancel() -> bool:
        calls[0] += 1
        # First two checks are entry/verify gates; the fourth is inside
        # _hash_file after one chunk was actually read.
        return calls[0] >= 4

    with pytest.raises(setup_plan.DownloadCancelled):
        setup_plan.download_verified(
            spec,
            root=root,
            opener=lambda url, headers: Response(body),
            cancel=cancel,
        )
    assert calls[0] >= 4
    assert part.read_bytes() == body
    assert not target.exists()


def test_cancel_before_final_replace_never_creates_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"verified-body"
    spec = _spec(body)
    root = tmp_path / "ComfyUI"
    target = root / spec.target
    target.parent.mkdir(parents=True)
    part = target.with_name(target.name + ".part")
    state = {"cancel": False}

    def hash_then_cancel(path, cancel=None):
        value = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        state["cancel"] = True
        return value

    monkeypatch.setattr(setup_plan, "_hash_file", hash_then_cancel)
    with pytest.raises(setup_plan.DownloadCancelled):
        setup_plan.download_verified(
            spec,
            root=root,
            opener=lambda url, headers: Response(body),
            cancel=lambda: state["cancel"],
        )
    assert part.read_bytes() == body
    assert not target.exists()


def test_cached_skip_emits_no_network_progress(tmp_path: Path) -> None:
    body = b"cached"
    spec = _spec(body)
    root = tmp_path / "ComfyUI"
    target = root / spec.target
    target.parent.mkdir(parents=True)
    target.write_bytes(body)
    network: list[tuple[int, int]] = []
    result = setup_plan.download_verified(
        spec,
        root=root,
        opener=lambda url, headers: (_ for _ in ()).throw(AssertionError("no request")),
        network_progress=lambda written, total: network.append((written, total)),
    )
    assert result.skipped is True
    assert network == []


def test_default_download_has_no_cancel_and_unchanged_network_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"plain"
    spec = _spec(body)
    seen: list[float] = []

    def open_with_timeout(url, headers, timeout=60.0):
        seen.append(float(timeout))
        return Response(body)

    monkeypatch.setattr(setup_plan, "_urllib_open", open_with_timeout)
    result = setup_plan.download_verified(spec, root=tmp_path)
    assert result.sha256 == spec.sha256
    assert seen == [60.0]


@pytest.mark.parametrize("offset", [10, 100, 150])
def test_ignored_range_network_bytes_are_exact_deltas(
    tmp_path: Path, offset: int
) -> None:
    size = 300
    body = bytes(range(256)) + bytes(range(size - 256))
    spec = _spec(body)
    root = tmp_path / f"ComfyUI-{offset}"
    target = root / spec.target
    target.parent.mkdir(parents=True)
    target.with_name(target.name + ".part").write_bytes(body[:offset])
    deltas: list[int] = []
    positions: list[int] = []
    result = setup_plan.download_verified(
        spec,
        root=root,
        opener=lambda url, headers: Response(body),
        chunk_size=100,
        network_bytes=deltas.append,
        progress=lambda written, total: positions.append(written),
    )
    assert sum(deltas) == len(body)
    assert all(delta == 100 for delta in deltas)
    assert positions[-1] == len(body)
    assert result.path.read_bytes() == body
