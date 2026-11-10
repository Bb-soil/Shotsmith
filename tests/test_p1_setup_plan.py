from __future__ import annotations

import copy
import hashlib
import os
import sys
import types
from pathlib import Path

import pytest

from scripts.p1.setup_plan import (
    DownloadError,
    DownloadSpec,
    SetupPlanError,
    build_setup_plan,
    download_verified,
    generate_constraints,
    load_manifest,
    manifest_download_specs,
    render_constraints_snippet,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "P1.wan22-5b.draft.json"


class FakeResponse:
    def __init__(self, status: int, body: bytes, headers: dict[str, str] | None = None):
        self.status = status
        self._body = body
        self._pos = 0
        self.headers = headers or {}
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        if self._pos >= len(self._body):
            return b""
        if size < 0:
            size = len(self._body) - self._pos
        chunk = self._body[self._pos : self._pos + size]
        self._pos += len(chunk)
        return chunk

    def close(self) -> None:
        self.closed = True


def _spec(body: bytes, *, target: str = "models/diffusion_models/test.bin") -> DownloadSpec:
    return DownloadSpec(
        model_id="fixture",
        url="https://example.invalid/model.bin",
        target=target,
        size_bytes=len(body),
        sha256=hashlib.sha256(body).hexdigest(),
    )


def test_manifest_has_exactly_three_verified_download_specs() -> None:
    manifest = load_manifest(MANIFEST)
    specs = manifest_download_specs(manifest)
    assert len(specs) == 3
    assert {spec.target for spec in specs} == {
        "models/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
        "models/vae/wan2.2_vae.safetensors",
        "models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    }
    assert sum(spec.size_bytes for spec in specs) == 18_144_966_705
    assert all(spec.sha256 for spec in specs)


def test_setup_plan_is_dry_run_and_preserves_base_torch() -> None:
    plan = build_setup_plan(load_manifest(MANIFEST), workspace="/workspace")
    rendered = plan.render()
    assert "runpod/pytorch:1.4.1-cu1281-torch291-ubuntu2404@sha256:8ab44dd" in rendered
    assert "checkout --detach 7a28332047b6af638dc6c7f72ef3481a718769fd" in rendered
    assert "python3 -m venv --system-site-packages" in rendered
    assert "--no-cache-dir -r \"$COMFY_DIR/requirements.txt\" -c \"$CONSTRAINTS\"" in rendered
    assert "import torch" in rendered
    assert "custom_nodes" in rendered
    assert "--execute" in rendered
    command_lines = [
        line for line in rendered.splitlines() if not line.lstrip().startswith("#")
    ]
    assert not any(
        "snapshot_download" in line or "huggingface-cli" in line
        for line in command_lines
    )


def test_constraints_generation_is_data_not_python_source() -> None:
    text = generate_constraints(
        {"torch": "2.9.1+cu128", "torchvision": "0.24.1", "torchaudio": "2.9.1"}
    )
    assert text == (
        "torch==2.9.1+cu128\n"
        "torchvision==0.24.1\n"
        "torchaudio==2.9.1\n"
    )
    assert "import " not in text
    assert all("==" in line for line in text.splitlines())

    with pytest.raises(SetupPlanError):
        generate_constraints({"torchvision": "0.24.1"})
    with pytest.raises(SetupPlanError):
        generate_constraints({"torch": "2.9.1; echo unsafe"})
    with pytest.raises(SetupPlanError):
        generate_constraints({"torch": "2.9.1\nimport os"})


def test_rendered_constraints_snippet_executes_with_fake_metadata(monkeypatch, capsys) -> None:
    versions = {
        "torch": "2.9.1+cu128",
        "torchvision": "0.24.1",
        "torchaudio": "2.9.1",
    }

    class Missing(Exception):
        pass

    fake = types.ModuleType("importlib.metadata")
    fake.PackageNotFoundError = Missing  # type: ignore[attr-defined]
    fake.version = lambda name: versions[name] if name in versions else (_ for _ in ()).throw(Missing(name))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "importlib.metadata", fake)

    exec(
        compile(render_constraints_snippet(), "constraints-snippet", "exec"),
        {"__name__": "__main__"},
    )
    output = capsys.readouterr().out
    assert output == generate_constraints(versions)
    assert "import " not in output

    missing_torch = types.ModuleType("importlib.metadata")
    missing_torch.PackageNotFoundError = Missing  # type: ignore[attr-defined]
    missing_torch.version = lambda name: (  # type: ignore[attr-defined]
        (_ for _ in ()).throw(Missing(name)) if name == "torch" else "0.24.1"
    )
    monkeypatch.setitem(sys.modules, "importlib.metadata", missing_torch)
    with pytest.raises(SetupPlanError):
        exec(
            compile(render_constraints_snippet(), "constraints-snippet", "exec"),
            {"__name__": "__main__"},
        )


def test_setup_plan_renders_executed_constraints_not_python_source() -> None:
    rendered = build_setup_plan(load_manifest(MANIFEST), workspace="/workspace").render()
    assert "cat > \"$CONSTRAINTS\"" not in rendered
    assert "<<'PY' > \"$CONSTRAINTS\"" in rendered
    assert 'grep -q \'^torch==\' "$CONSTRAINTS"' in rendered
    assert "test -s \"$CONSTRAINTS\"" in rendered


def test_manifest_rejects_wrong_model_count_and_unsafe_targets() -> None:
    manifest = load_manifest(MANIFEST)
    with pytest.raises(SetupPlanError):
        build_setup_plan(
            {**manifest, "models": manifest["models"][:2]},
            workspace="/workspace",
        )

    for bad_target in ("../escape.bin", "/absolute.bin", "weights/model.bin", "models/../x"):
        mutated = copy.deepcopy(manifest)
        mutated["models"][0]["target"] = bad_target
        with pytest.raises(SetupPlanError):
            build_setup_plan(mutated, workspace="/workspace")


def test_manifest_rejects_non_https_or_credential_urls() -> None:
    manifest = load_manifest(MANIFEST)
    for bad_url in (
        "http://example.invalid/model.bin",
        "https://user:pass@example.invalid/model.bin",
        "https://example.invalid/model.bin?token=x",
    ):
        mutated = copy.deepcopy(manifest)
        mutated["models"][0]["download_url"] = bad_url
        with pytest.raises(SetupPlanError):
            build_setup_plan(mutated, workspace="/workspace")


def test_download_fresh_then_skip_verified_without_second_request(tmp_path) -> None:
    body = b"tiny-model-bytes"
    spec = _spec(body)
    root = tmp_path / "ComfyUI"
    root.mkdir()
    calls: list[tuple[str, dict[str, str]]] = []

    def opener(url: str, headers: dict[str, str]):
        calls.append((url, dict(headers)))
        return FakeResponse(200, body)

    result = download_verified(spec, root=root, opener=opener)
    assert result.skipped is False
    assert result.path.read_bytes() == body
    assert not result.path.with_name(result.path.name + ".part").exists()
    assert calls == [(spec.url, {})]

    def must_not_call(url: str, headers: dict[str, str]):
        raise AssertionError("verified final file must be skipped")

    again = download_verified(spec, root=root, opener=must_not_call)
    assert again.skipped is True
    assert again.path.read_bytes() == body


def test_download_resumes_part_with_206_content_range(tmp_path) -> None:
    body = b"0123456789abcdef"
    prefix = body[:7]
    spec = _spec(body)
    root = tmp_path / "ComfyUI"
    target = root / spec.target
    target.parent.mkdir(parents=True)
    target.with_name(target.name + ".part").write_bytes(prefix)
    seen: dict[str, str] = {}

    def opener(url: str, headers: dict[str, str]):
        seen.update(headers)
        return FakeResponse(
            206,
            body[len(prefix) :],
            {"Content-Range": f"bytes {len(prefix)}-{len(body) - 1}/{len(body)}"},
        )

    result = download_verified(spec, root=root, opener=opener)
    assert seen == {"Range": f"bytes={len(prefix)}-"}
    assert result.path.read_bytes() == body
    assert result.skipped is False


def test_exact_complete_part_valid_finalizes_without_opener(tmp_path) -> None:
    body = b"already-complete"
    spec = _spec(body)
    root = tmp_path / "ComfyUI"
    target = root / spec.target
    target.parent.mkdir(parents=True)
    target.with_name(target.name + ".part").write_bytes(body)

    def opener(url: str, headers: dict[str, str]):
        raise AssertionError("a complete verified .part must not open a request")

    result = download_verified(spec, root=root, opener=opener)
    assert result.path.read_bytes() == body
    assert result.skipped is False
    assert not target.with_name(target.name + ".part").exists()


def test_exact_complete_bad_part_removed_then_clean_retry(tmp_path) -> None:
    body = b"expected-body!"
    spec = _spec(body)
    root = tmp_path / "ComfyUI"
    target = root / spec.target
    target.parent.mkdir(parents=True)
    part = target.with_name(target.name + ".part")
    part.write_bytes(b"wrong-body!!!!")

    with pytest.raises(DownloadError):
        download_verified(
            spec,
            root=root,
            opener=lambda url, headers: FakeResponse(200, body),
        )
    assert not target.exists()
    assert not part.exists()

    retry = download_verified(
        spec,
        root=root,
        opener=lambda url, headers: FakeResponse(200, body),
    )
    assert retry.path.read_bytes() == body


def test_oversized_part_is_discarded_before_clean_download(tmp_path) -> None:
    body = b"fresh"
    spec = _spec(body)
    root = tmp_path / "ComfyUI"
    target = root / spec.target
    target.parent.mkdir(parents=True)
    part = target.with_name(target.name + ".part")
    part.write_bytes(b"too-long-part")
    result = download_verified(
        spec,
        root=root,
        opener=lambda url, headers: FakeResponse(200, body),
    )
    assert result.path.read_bytes() == body
    assert not part.exists()


def test_download_200_ignoring_range_resets_safely(tmp_path) -> None:
    body = b"complete-body"
    prefix = body[:4]
    spec = _spec(body)
    root = tmp_path / "ComfyUI"
    target = root / spec.target
    target.parent.mkdir(parents=True)
    target.with_name(target.name + ".part").write_bytes(prefix)

    def opener(url: str, headers: dict[str, str]):
        assert headers == {"Range": f"bytes={len(prefix)}-"}
        return FakeResponse(200, body)

    result = download_verified(spec, root=root, opener=opener)
    assert result.path.read_bytes() == body


def test_download_bad_content_range_or_hash_never_creates_final(tmp_path) -> None:
    body = b"0123456789"
    root = tmp_path / "ComfyUI"
    target = root / "models/diffusion_models/test.bin"
    target.parent.mkdir(parents=True)
    spec = _spec(body)

    def bad_range(url: str, headers: dict[str, str]):
        return FakeResponse(206, body, {"Content-Range": "bytes 1-9/10"})

    with pytest.raises(DownloadError):
        download_verified(spec, root=root, opener=bad_range)
    assert not target.exists()

    def bad_hash(url: str, headers: dict[str, str]):
        return FakeResponse(200, b"012345678X")

    with pytest.raises(DownloadError):
        download_verified(spec, root=root, opener=bad_hash)
    assert not target.exists()
    assert not target.with_name(target.name + ".part").exists()


def test_download_existing_mismatching_final_is_refused(tmp_path) -> None:
    body = b"expected"
    root = tmp_path / "ComfyUI"
    target = root / "models/diffusion_models/test.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"wrong")
    spec = _spec(body)

    with pytest.raises(DownloadError):
        download_verified(spec, root=root, opener=lambda url, headers: FakeResponse(200, body))
    assert target.read_bytes() == b"wrong"


def test_download_rejects_traversal_and_oversized_streams(tmp_path) -> None:
    root = tmp_path / "ComfyUI"
    root.mkdir()
    with pytest.raises((SetupPlanError, DownloadError)):
        download_verified(
            _spec(b"x", target="../escape.bin"),
            root=root,
            opener=lambda url, headers: FakeResponse(200, b"x"),
        )

    spec = _spec(b"1234")
    with pytest.raises(DownloadError):
        download_verified(
            spec,
            root=root,
            opener=lambda url, headers: FakeResponse(200, b"12345"),
        )
    assert not (root / spec.target).exists()


def test_download_rejects_symlink_escape(tmp_path) -> None:
    root = tmp_path / "ComfyUI"
    (root / "models").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "models" / "link"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not available on this Windows host")
    with pytest.raises(DownloadError):
        download_verified(
            _spec(b"x", target="models/link/file.bin"),
            root=root,
            opener=lambda url, headers: FakeResponse(200, b"x"),
        )


def test_contained_path_retries_transient_windows_resolve_disagreement(
    tmp_path, monkeypatch
) -> None:
    from scripts.p1 import setup_plan as sp

    root = tmp_path / "ComfyUI"
    root.mkdir()
    real = sp.os.path.commonpath
    calls = {"n": 0}

    def flaky(paths):
        calls["n"] += 1
        if calls["n"] == 1:
            return str(tmp_path / "outside")
        return real(paths)

    monkeypatch.setattr(sp.os.path, "commonpath", flaky)
    target = sp._contained_path(root, "models/race/file.bin")
    assert root.resolve() in target.parents
    assert calls["n"] >= 2


def test_rendered_setup_has_single_pip_continuation_and_lf() -> None:
    plan = build_setup_plan(load_manifest(MANIFEST))
    text = plan.render()
    assert "\r\n" not in text
    assert "\\\\\n" not in text
    assert "pip install \\\n  --no-cache-dir" in text


def test_rendered_setup_passes_actual_bash_syntax_if_available(tmp_path) -> None:
    import shutil
    import subprocess

    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash is absent: SHELL_EXECUTION_NOT_RUN")
    plan = build_setup_plan(load_manifest(MANIFEST))
    script = tmp_path / "setup.sh"
    script.write_bytes(plan.render().encode("utf-8"))
    path = str(script)
    if os.name == "nt":
        # WSL bash needs /mnt/<drive>/... rather than a Windows path.
        drive = Path(path).drive.rstrip(":").lower()
        rest = path.split(":", 1)[1].replace("\\", "/")
        path = f"/mnt/{drive}{rest}"
    completed = subprocess.run(
        [bash, "-n", path],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_download_progress_is_monotonic_and_bounded(tmp_path) -> None:
    body = b"0123456789" * 100
    spec = _spec(body)
    seen: list[tuple[int, int]] = []

    def progress(written: int, total: int) -> None:
        seen.append((written, total))

    result = download_verified(
        spec,
        root=tmp_path,
        opener=lambda url, headers: FakeResponse(200, body),
        chunk_size=64,
        progress=progress,
    )
    assert result.skipped is False
    assert seen
    assert all(total == len(body) for _written, total in seen)
    assert [written for written, _total in seen] == sorted(
        written for written, _total in seen
    )
    assert seen[-1] == (len(body), len(body))


def test_download_rejects_unbounded_chunk_size(tmp_path) -> None:
    with pytest.raises(DownloadError):
        download_verified(
            _spec(b"x"),
            root=tmp_path,
            opener=lambda url, headers: FakeResponse(200, b"x"),
            chunk_size=0,
        )
    with pytest.raises(DownloadError):
        download_verified(
            _spec(b"x"),
            root=tmp_path,
            opener=lambda url, headers: FakeResponse(200, b"x"),
            chunk_size=9 * 1024 * 1024,
        )
