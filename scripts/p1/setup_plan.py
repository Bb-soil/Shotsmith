"""Offline P1 worker setup plan and a verified model downloader.

The default CLI is dry-run: it reads the local draft manifest, validates the
three declared downloads and prints a bash plan. It never starts a subprocess,
opens a network connection or inspects credentials.

``--execute`` is intentionally limited to the three manifest downloads. It is
not a setup runner, it does not install packages or clone ComfyUI, and it must
only be used after a separate owner approval and a real Pod is available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, TextIO
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "manifests" / "P1.wan22-5b.draft.json"
DEFAULT_WORKSPACE = "/workspace"
MAX_SESSION_SECONDS = 60 * 60
HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
HEX40 = re.compile(r"^[0-9a-fA-F]{40}$")
VERSION_RE = re.compile(r"^[0-9][0-9A-Za-z.+_-]*$")
CHUNK_SIZE = 1024 * 1024

CONSTRAINTS_SNIPPET = """\
from importlib.metadata import PackageNotFoundError, version

from scripts.p1.setup_plan import generate_constraints

versions = {}
for name in ("torch", "torchvision", "torchaudio"):
    try:
        versions[name] = version(name)
    except PackageNotFoundError:
        pass

print(generate_constraints(versions), end="")
"""


def generate_constraints(versions: Mapping[str, str]) -> str:
    """Return pip constraint lines, requiring the base image torch install."""

    if not isinstance(versions, Mapping) or "torch" not in versions:
        raise SetupPlanError(
            "base image torch is required; refusing to generate empty constraints"
        )
    lines: list[str] = []
    for name in ("torch", "torchvision", "torchaudio"):
        if name not in versions:
            continue
        value = versions[name]
        if not isinstance(value, str) or not VERSION_RE.fullmatch(value):
            raise SetupPlanError(f"invalid {name} version for a pip constraint")
        lines.append(f"{name}=={value}")
    return "\n".join(lines) + "\n"


def render_constraints_snippet() -> str:
    return CONSTRAINTS_SNIPPET

Opener = Callable[[str, Mapping[str, str]], Any]


class SetupPlanError(Exception):
    """A manifest or setup plan is unsafe or incomplete."""


class DownloadError(Exception):
    """A download failed a size, status, range or checksum check."""


class DownloadCancelled(DownloadError):
    """A caller-requested cancellation left the resumable part untouched."""


@dataclass(frozen=True)
class DownloadSpec:
    model_id: str
    url: str
    target: str
    size_bytes: int
    sha256: str

    @property
    def label(self) -> str:
        return f"{self.model_id} -> {self.target}"


@dataclass(frozen=True)
class DownloadResult:
    spec: DownloadSpec
    path: Path
    skipped: bool
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class SetupPlan:
    image: str
    image_digest: str
    comfyui_source: str
    comfyui_commit: str
    workspace: str
    comfyui_dir: str
    venv_dir: str
    constraints_path: str
    manifest_path: str
    downloads: tuple[DownloadSpec, ...]

    def render(self) -> str:
        workspace = shlex.quote(self.workspace)
        comfy = shlex.quote(self.comfyui_dir)
        venv = shlex.quote(self.venv_dir)
        constraints = shlex.quote(self.constraints_path)
        manifest = shlex.quote(self.manifest_path)
        source = shlex.quote(self.comfyui_source)
        commit = shlex.quote(self.comfyui_commit)
        image = shlex.quote(self.image + "@" + self.image_digest)
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "# P1 worker setup plan. Review before running; this file is not executed by setup_plan.py.",
            f"# Official base image, pinned by digest: {image}",
            "# Do not substitute the public ComfyUI template, a community image or :latest.",
            f"WORKSPACE={workspace}",
            f"COMFY_DIR={comfy}",
            f"VENV={venv}",
            f"CONSTRAINTS={constraints}",
            f"MANIFEST={manifest}",
            'mkdir -p "$WORKSPACE" "$WORKSPACE/p1-evidence" "$WORKSPACE/.cache/pip"',
            'if [ ! -d "$COMFY_DIR/.git" ]; then',
            f'  git clone --filter=blob:none --no-checkout {source} "$COMFY_DIR"',
            "fi",
            'git -C "$COMFY_DIR" fetch --all --tags',
            f'git -C "$COMFY_DIR" checkout --detach {commit}',
            'mkdir -p "$COMFY_DIR/custom_nodes"',
            "# custom_nodes intentionally stays empty: the prepared Wan graph is core-only.",
            'python3 -m venv --system-site-packages "$VENV"',
            'PIP_CACHE_DIR="$WORKSPACE/.cache/pip" "$VENV/bin/python" -m pip install --upgrade pip',
            'test -x "$VENV/bin/python"',
            '"$VENV/bin/python" -c "import torch; print(torch.__version__)"',
            "# Execute this Python snippet; its stdout is the constraints file, not Python source.",
            '"$VENV/bin/python" - <<\'PY\' > "$CONSTRAINTS"',
            *CONSTRAINTS_SNIPPET.splitlines(),
            "PY",
            'test -s "$CONSTRAINTS"',
            'grep -q \'^torch==\' "$CONSTRAINTS" || { echo "base image torch constraint missing" >&2; exit 1; }',
            'PIP_CACHE_DIR="$WORKSPACE/.cache/pip" "$VENV/bin/python" -m pip install \\',
            '  --no-cache-dir -r "$COMFY_DIR/requirements.txt" -c "$CONSTRAINTS"',
            "# The constraints preserve the base image torch/torchvision/torchaudio CUDA wheels.",
            "# If pip reports a conflict, stop and obtain a new approved plan; do not install torch manually.",
            "# Source requirements are not a fully locked environment. Future preflight must record pip check and pip freeze.",
            "# Model downloads are exactly the three manifest entries; run from the repository root:",
            '"$VENV/bin/python" scripts/p1/setup_plan.py --manifest "$MANIFEST" --workspace "$WORKSPACE" --execute',
            "# Do not use huggingface-cli/snapshot_download, a second HF cache or a copied torch wheel.",
            "# Then run the preflight plan; no generation is part of setup.",
        ]
        return "\n".join(lines) + "\n"


def _validate_workspace(value: str) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise SetupPlanError("workspace must be a non-empty path string")
    path = PurePosixPath(value)
    if not path.is_absolute() or any(part in ("", ".", "..") for part in path.parts[1:]):
        raise SetupPlanError("workspace must be an absolute normalized POSIX path")
    return str(path)


def _validate_target(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise SetupPlanError("model target must be a non-empty string")
    if "\\" in value or ":" in value:
        raise SetupPlanError("model target must use POSIX separators only")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise SetupPlanError("model target must be a normalized relative path")
    if path.parts[0] != "models":
        raise SetupPlanError("model target must be under models/")
    return str(path)


def _validate_url(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise SetupPlanError("download URL must be a non-empty string")
    parts = urlsplit(value)
    if parts.scheme != "https" or not parts.hostname:
        raise SetupPlanError("download URL must be https with a host")
    if parts.username or parts.password:
        raise SetupPlanError("download URL must not contain userinfo")
    if parts.query or parts.fragment:
        raise SetupPlanError("download URL must not contain query/fragment")
    return value


def _validate_spec(data: Mapping[str, Any]) -> DownloadSpec:
    model_id = data.get("id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise SetupPlanError("model id must be a non-empty string")
    size = data.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise SetupPlanError(f"{model_id}: size_bytes must be a positive integer")
    checksum = data.get("sha256")
    if not isinstance(checksum, str) or not HEX64.fullmatch(checksum):
        raise SetupPlanError(f"{model_id}: sha256 must be 64 hex characters")
    return DownloadSpec(
        model_id=model_id,
        url=_validate_url(data.get("download_url")),
        target=_validate_target(data.get("target")),
        size_bytes=size,
        sha256=checksum.lower(),
    )


def manifest_download_specs(manifest: Mapping[str, Any]) -> tuple[DownloadSpec, ...]:
    if not isinstance(manifest, Mapping):
        raise SetupPlanError("manifest must be an object")
    models = manifest.get("models")
    if not isinstance(models, list) or len(models) != 3:
        raise SetupPlanError("manifest.models must contain exactly the three prepared files")
    specs = tuple(_validate_spec(model) for model in models if isinstance(model, Mapping))
    if len(specs) != 3:
        raise SetupPlanError("every manifest model entry must be an object")
    targets = [spec.target for spec in specs]
    if len(set(targets)) != len(targets):
        raise SetupPlanError("manifest model targets must be unique")
    return specs


def build_setup_plan(
    manifest: Mapping[str, Any],
    *,
    workspace: str = DEFAULT_WORKSPACE,
    manifest_path: str = "manifests/P1.wan22-5b.draft.json",
) -> SetupPlan:
    workspace_text = _validate_workspace(workspace)
    container = manifest.get("container")
    if not isinstance(container, Mapping):
        raise SetupPlanError("manifest.container is missing")
    image = container.get("image")
    digest = container.get("digest")
    if not isinstance(image, str) or not image.strip():
        raise SetupPlanError("manifest.container.image is missing")
    if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest):
        raise SetupPlanError("manifest.container.digest must be sha256:<64 hex>")
    comfyui = manifest.get("comfyui")
    if not isinstance(comfyui, Mapping):
        raise SetupPlanError("manifest.comfyui is missing")
    source = comfyui.get("source_url")
    commit = comfyui.get("commit")
    if not isinstance(source, str) or not source.startswith("https://"):
        raise SetupPlanError("manifest.comfyui.source_url must be https")
    if not isinstance(commit, str) or not HEX40.fullmatch(commit):
        raise SetupPlanError("manifest.comfyui.commit must be 40 hex characters")
    comfyui_dir = str(PurePosixPath(workspace_text) / "ComfyUI")
    venv_dir = str(PurePosixPath(workspace_text) / "p1-venv")
    constraints = str(PurePosixPath(workspace_text) / "p1-torch-constraints.txt")
    return SetupPlan(
        image=image,
        image_digest=digest,
        comfyui_source=source,
        comfyui_commit=commit,
        workspace=workspace_text,
        comfyui_dir=comfyui_dir,
        venv_dir=venv_dir,
        constraints_path=constraints,
        manifest_path=manifest_path,
        downloads=manifest_download_specs(manifest),
    )


def _contained_path(root: Path, target: str) -> Path:
    """Resolve a manifest target and reject traversal/symlink escapes.

    The lexical check runs first.  The resolved check is retried because on
    Windows a concurrent creator of a parent directory can transiently make
    ``Path.resolve`` disagree with the just-created stable path; retrying never
    weakens the escape check.
    """

    root_resolved = root.resolve()
    candidate_lexical = root_resolved / Path(target)
    try:
        candidate_lexical.relative_to(root_resolved)
    except ValueError as exc:
        raise DownloadError(f"download target escapes the model root: {target!r}") from exc
    root_key = os.path.normcase(str(root_resolved))
    last_candidate = candidate_lexical
    for _ in range(3):
        candidate = candidate_lexical.resolve()
        last_candidate = candidate
        try:
            common = os.path.commonpath(
                [root_key, os.path.normcase(str(candidate))]
            )
        except ValueError:
            common = ""
        if common == root_key and candidate != root_resolved:
            if candidate.exists() and candidate.is_symlink():
                raise DownloadError(f"download target must not be a symlink: {target!r}")
            return candidate
    if last_candidate.exists() and last_candidate.is_symlink():
        raise DownloadError(f"download target must not be a symlink: {target!r}")
    raise DownloadError(f"download target escapes the model root: {target!r}")


def _hash_file(path: Path, cancel: Callable[[], bool] | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            if cancel is not None and cancel():
                raise DownloadCancelled("download cancelled during hash verification")
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _parse_content_range(value: str | None, *, offset: int, total: int) -> None:
    if not isinstance(value, str):
        raise DownloadError("HTTP 206 response has no Content-Range")
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value.strip())
    if match is None:
        raise DownloadError("HTTP 206 Content-Range is malformed")
    start, end, declared_total = (int(item) for item in match.groups())
    if declared_total != total:
        raise DownloadError("HTTP 206 Content-Range total does not match manifest size")
    if start != offset or end != total - 1:
        raise DownloadError("HTTP 206 Content-Range does not match the requested offset")


def _urllib_open(url: str, headers: Mapping[str, str], timeout: float = 60.0) -> Any:
    request = urllib.request.Request(url, headers=dict(headers), method="GET")
    return urllib.request.urlopen(request, timeout=float(timeout))


def download_verified(
    spec: DownloadSpec,
    *,
    root: Path | str,
    opener: Opener | None = None,
    chunk_size: int = CHUNK_SIZE,
    log: TextIO | None = None,
    progress: Callable[[int, int], None] | None = None,
    network_progress: Callable[[int, int], None] | None = None,
    network_bytes: Callable[[int], None] | None = None,
    stage: Callable[[str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    network_timeout: float = 60.0,
) -> DownloadResult:
    """Download one manifest file with resume and full-file SHA256 verification.

    A verified existing final file is skipped. A mismatching existing final file
    is never overwritten. ``.part`` is kept on an incomplete transfer for a
    future resume, but is removed on overflow/checksum failure. The final file is
    created only by an atomic ``os.replace`` after hashing the complete file.

    ``progress`` is an optional bounded display hook: it receives
    ``(bytes_written, total_bytes)`` and never a URL, token or response body.
    It cannot change the transfer, retries or verification outcome.
    ``network_progress`` is the same shape but is called only for bytes actually
    read from the network, never for an existing final, a resume offset or a
    cached skip. ``network_bytes`` receives the exact positive byte delta read
    from the socket for each chunk, so an ignored Range/reset cannot corrupt
    speed accounting. ``cancel`` is an optional cooperative cancellation
    predicate; when omitted, behavior is identical to the original P1 downloader.
    """

    if isinstance(spec.size_bytes, bool) or not isinstance(spec.size_bytes, int):
        raise DownloadError("manifest size is invalid")
    if spec.size_bytes <= 0 or not HEX64.fullmatch(spec.sha256):
        raise DownloadError("manifest size/checksum is invalid")
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
        or chunk_size > 8 * 1024 * 1024
    ):
        raise DownloadError("chunk_size must be a bounded positive integer")
    if progress is not None and not callable(progress):
        raise DownloadError("progress must be callable or None")
    if network_progress is not None and not callable(network_progress):
        raise DownloadError("network_progress must be callable or None")
    if network_bytes is not None and not callable(network_bytes):
        raise DownloadError("network_bytes must be callable or None")
    if stage is not None and not callable(stage):
        raise DownloadError("stage must be callable or None")
    if cancel is not None and not callable(cancel):
        raise DownloadError("cancel must be callable or None")
    if (
        isinstance(network_timeout, bool)
        or not isinstance(network_timeout, (int, float))
        or not (0 < float(network_timeout) <= 60)
    ):
        raise DownloadError("network_timeout must be within 0..60 seconds")

    def check_cancel() -> None:
        if cancel is not None and cancel():
            raise DownloadCancelled("download cancelled")

    def report_network(written: int, delta: int) -> None:
        if network_progress is not None:
            network_progress(written, spec.size_bytes)
        if network_bytes is not None:
            network_bytes(delta)

    check_cancel()

    def report_stage(name: str) -> None:
        if stage is not None:
            stage(name)

    if not isinstance(spec.target, str):
        raise DownloadError("manifest target is invalid")
    _validate_target(spec.target)
    try:
        _validate_url(spec.url)
    except SetupPlanError as exc:
        raise DownloadError(str(exc)) from exc
    root_path = Path(root)
    final = _contained_path(root_path, spec.target)
    final.parent.mkdir(parents=True, exist_ok=True)
    part = final.with_name(final.name + ".part")
    if part.is_symlink():
        raise DownloadError(f"partial file must not be a symlink: {part}")

    if final.exists():
        if final.is_symlink() or not final.is_file():
            raise DownloadError(f"existing final target is not a regular file: {final}")
        size = final.stat().st_size
        report_stage("VERIFYING")
        check_cancel()
        if size == spec.size_bytes and _hash_file(final, cancel).lower() == spec.sha256:
            if progress is not None:
                progress(size, spec.size_bytes)
            if log is not None:
                log.write(f"skip verified: {spec.label}\n")
            report_stage("DONE")
            return DownloadResult(spec, final, True, spec.sha256, size)
        raise DownloadError(f"existing final file does not match the manifest: {spec.label}")

    offset = 0
    if part.exists():
        if part.is_symlink() or not part.is_file():
            raise DownloadError(f"partial path is not a regular file: {part}")
        offset = part.stat().st_size
        if offset > spec.size_bytes:
            part.unlink()
            offset = 0
        elif offset == spec.size_bytes:
            report_stage("VERIFYING")
            check_cancel()
            actual_sha = _hash_file(part, cancel)
            if actual_sha.lower() == spec.sha256:
                check_cancel()
                os.replace(part, final)
                if progress is not None:
                    progress(offset, spec.size_bytes)
                if log is not None:
                    log.write(f"verified from complete .part: {spec.label}\n")
                report_stage("DONE")
                return DownloadResult(spec, final, False, actual_sha, offset)
            # A complete but corrupt part cannot be resumed with Range (a
            # bytes=<size>- request is invalid). Remove it to allow a clean
            # retry on the next explicit execution; never finalize it.
            part.unlink(missing_ok=True)
            raise DownloadError(
                f"complete .part failed SHA256 for {spec.label}; removed for a clean retry"
            )

    if progress is not None:
        progress(offset, spec.size_bytes)
    report_stage("DOWNLOADING")

    headers: dict[str, str] = {}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    check_cancel()
    if opener is None:
        opener = lambda url, request_headers: _urllib_open(  # noqa: E731
            url, request_headers, timeout=network_timeout
        )
    try:
        response = opener(spec.url, headers)
    except (OSError, urllib.error.URLError) as exc:
        raise DownloadError(f"download request failed for {spec.label}: {type(exc).__name__}") from exc

    try:
        status = int(getattr(response, "status", 0) or getattr(response, "getcode", lambda: 0)())
        if status == 206:
            _parse_content_range(
                response.headers.get("Content-Range"),
                offset=offset,
                total=spec.size_bytes,
            )
            mode = "ab" if offset else "wb"
        elif status == 200:
            # A server that ignores Range is safe only if we restart the file.
            offset = 0
            mode = "wb"
        else:
            raise DownloadError(f"unexpected HTTP status {status} for {spec.label}")

        written = offset
        with part.open(mode) as handle:
            while True:
                check_cancel()
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                check_cancel()
                written += len(chunk)
                if written > spec.size_bytes:
                    handle.close()
                    try:
                        part.unlink(missing_ok=True)
                    except OSError:
                        pass
                    raise DownloadError(f"download exceeded manifest size: {spec.label}")
                handle.write(chunk)
                report_network(written, len(chunk))
                if progress is not None:
                    progress(written, spec.size_bytes)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()

    if written != spec.size_bytes:
        raise DownloadError(
            f"incomplete download for {spec.label}: {written} of {spec.size_bytes} bytes"
        )
    report_stage("VERIFYING")
    check_cancel()
    actual_sha = _hash_file(part, cancel)
    if actual_sha.lower() != spec.sha256:
        try:
            part.unlink(missing_ok=True)
        except OSError:
            pass
        raise DownloadError(f"SHA256 mismatch for {spec.label}; final file not created")
    check_cancel()
    os.replace(part, final)
    if progress is not None:
        progress(written, spec.size_bytes)
    if log is not None:
        log.write(f"verified: {spec.label} ({written} bytes)\n")
    report_stage("DONE")
    return DownloadResult(spec, final, False, actual_sha, written)


def load_manifest(path: Path | str) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SetupPlanError(f"cannot load manifest {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SetupPlanError("manifest must be a JSON object")
    return data


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/p1/setup_plan.py",
        description="Render a dry-run P1 worker setup plan; download only with --execute.",
    )
    parser.add_argument(
        "--manifest",
        default="manifests/P1.wan22-5b.draft.json",
        help="manifest path; run from the repository root for the default",
    )
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="download the three verified manifest files (requires separate approval)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        plan = build_setup_plan(
            manifest,
            workspace=args.workspace,
            manifest_path=args.manifest,
        )
        if not args.execute:
            print(plan.render(), end="")
            return 0
        for spec in plan.downloads:
            download_verified(spec, root=Path(plan.comfyui_dir), log=sys.stdout)
        return 0
    except (SetupPlanError, DownloadError) as exc:
        print(f"setup plan refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
