"""Render the owned POSIX setup script (no model downloads, no torch reinstall)."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .errors import BlockedError
from .profile import Profile
from .stock_nodes import VALIDATOR_SOURCE
from .util import validate_workspace


def _quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


def render_setup(profile: Profile, *, workspace: str) -> str:
    workspace = validate_workspace(workspace)
    root = str(PurePosixPath(workspace) / ".avs-i1")
    comfy = str(PurePosixPath(workspace) / "ComfyUI")
    venv = str(PurePosixPath(workspace) / "p1-venv")
    constraints = str(PurePosixPath(workspace) / "p1-torch-constraints.txt")
    commit = str(profile.comfy["commit"])
    source = str(profile.comfy["source_url"])
    return "\n".join(
        (
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "# I1 owned setup: no model downloads and no PyTorch reinstall.",
            "umask 077",
            f"WORKSPACE={_quote(workspace)}",
            f"ROOT={_quote(root)}",
            f"COMFY_DIR={_quote(comfy)}",
            f"VENV={_quote(venv)}",
            f"CONSTRAINTS={_quote(constraints)}",
            f"PINNED_COMMIT={_quote(commit)}",
            f"PINNED_SOURCE={_quote(source)}",
            'test -d "$WORKSPACE" || { echo "workspace is missing" >&2; exit 1; }',
            'test -w "$WORKSPACE" || { echo "workspace is not writable" >&2; exit 1; }',
            'command -v bash >/dev/null && command -v git >/dev/null && command -v python3 >/dev/null && command -v tee >/dev/null || { echo "bash/git/python3/tee missing" >&2; exit 1; }',
            'python3 -c "import venv" >/dev/null 2>&1 || { echo "python3 venv module missing" >&2; exit 1; }',
            'WORKSPACE_REAL=$(realpath "$WORKSPACE")',
            'for path in "$ROOT" "$COMFY_DIR" "$VENV" "$VENV/bin" "$ROOT/logs" "$CONSTRAINTS" "$ROOT/logs/setup.log" "$ROOT/logs/torch-before.json" "$ROOT/logs/torch-after.json"; do',
            '  if [ -L "$path" ]; then echo "unsafe symlink before mutation: $path" >&2; exit 1; fi',
            "done",
            'for path in "$ROOT" "$COMFY_DIR" "$VENV"; do',
            '  if [ -e "$path" ]; then',
            '    real=$(realpath "$path")',
            '    case "$real" in "$WORKSPACE_REAL"/*) ;; *) echo "path escapes workspace: $path -> $real" >&2; exit 1;; esac',
            "  fi",
            "done",
            'if [ -e "$VENV/bin/python" ] && [ ! -x "$VENV/bin/python" ]; then',
            '  echo "venv/bin/python is not executable" >&2; exit 1',
            "fi",
            'if [ -e "$VENV" ] && [ ! -f "$VENV/pyvenv.cfg" ]; then',
            '  echo "foreign venv without pyvenv.cfg; refusing to mutate" >&2; exit 1',
            "fi",
            'if [ -e "$COMFY_DIR" ] && [ ! -d "$COMFY_DIR/.git" ]; then',
            '  echo "ComfyUI path exists but is not a git checkout; refusing to mutate" >&2',
            "  exit 1",
            "fi",
            'if [ ! -d "$COMFY_DIR/.git" ]; then',
            '  git clone --filter=blob:none --no-checkout "$PINNED_SOURCE" "$COMFY_DIR"',
            '  git -C "$COMFY_DIR" checkout --detach "$PINNED_COMMIT"',
            "fi",
            'ACTUAL_COMMIT=$(git -C "$COMFY_DIR" rev-parse HEAD 2>/dev/null || true)',
            'if [ "$ACTUAL_COMMIT" != "$PINNED_COMMIT" ]; then',
            '  echo "ComfyUI commit mismatch: expected $PINNED_COMMIT, actual ${ACTUAL_COMMIT:-missing}; no reset/clean performed" >&2',
            "  exit 1",
            "fi",
            'if [ -n "$(git -C "$COMFY_DIR" status --porcelain --untracked-files=no)" ]; then',
            '  echo "ComfyUI checkout has modified tracked files; refusing to mutate" >&2',
            "  exit 1",
            "fi",
            'if ! NODES_JSON=$(python3 -B - "$COMFY_DIR" "$PINNED_COMMIT" <<\'I1_STOCK_NODES_PY\'',
            *VALIDATOR_SOURCE.splitlines(),
            "I1_STOCK_NODES_PY",
            "); then",
            '  echo "custom_nodes: проверка не пройдена: $NODES_JSON" >&2',
            "  exit 1",
            "fi",
            'test -n "$NODES_JSON" || { echo "custom_nodes: validator вернул пустой результат" >&2; exit 1; }',
            'test -f "$COMFY_DIR/requirements.txt" || { echo "ComfyUI requirements.txt missing" >&2; exit 1; }',
            'mkdir -p "$ROOT/logs"',
            'LOG="$ROOT/logs/setup.log"',
            'exec > >(tee -a "$LOG") 2>&1',
            'python3 -B -c \'import json,sys; d=json.loads(sys.argv[1]); c=list(d.get("cache_unverified") or [])+list(d.get("cache_dirs") or []); print("custom_nodes: runtime cache (не проверено): " + ", ".join(c)) if c else None\' "$NODES_JSON" || true',
            'if [ ! -x "$VENV/bin/python" ]; then',
            '  python3 -m venv --system-site-packages "$VENV"',
            "fi",
            '"$VENV/bin/python" - <<\'PY\' > "$ROOT/logs/torch-before.json"',
            "import json",
            "import torch",
            'print(json.dumps({"version": torch.__version__, "cuda": torch.version.cuda, "available": torch.cuda.is_available()}, sort_keys=True))',
            "PY",
            'test -s "$ROOT/logs/torch-before.json" || { echo "base torch is unavailable before setup" >&2; exit 1; }',
            '"$VENV/bin/python" - "$CONSTRAINTS" <<\'PY\'',
            "import sys",
            "from importlib.metadata import PackageNotFoundError, version",
            "from setup_plan import generate_constraints",
            "versions = {}",
            'for name in ("torch", "torchvision", "torchaudio"):',
            "    try:",
            "        versions[name] = version(name)",
            "    except PackageNotFoundError:",
            "        pass",
            'with open(sys.argv[1], "w", encoding="utf-8") as handle:',
            "    handle.write(generate_constraints(versions))",
            "PY",
            'test -s "$CONSTRAINTS" || { echo "constraints file is empty" >&2; exit 1; }',
            'grep -q \'^torch==\' "$CONSTRAINTS" || { echo "base torch constraint missing" >&2; exit 1; }',
            'PIP_CACHE_DIR="$ROOT/pip-cache" "$VENV/bin/python" -m pip install --disable-pip-version-check --no-cache-dir -r "$COMFY_DIR/requirements.txt" -c "$CONSTRAINTS"',
            '"$VENV/bin/python" -m pip check',
            '"$VENV/bin/python" - <<\'PY\' > "$ROOT/logs/torch-after.json"',
            "import json",
            "import torch",
            'print(json.dumps({"version": torch.__version__, "cuda": torch.version.cuda, "available": torch.cuda.is_available()}, sort_keys=True))',
            "PY",
            'if ! cmp -s "$ROOT/logs/torch-before.json" "$ROOT/logs/torch-after.json"; then',
            '  echo "base torch/CUDA changed during setup; refusing to continue" >&2',
            "  exit 1",
            "fi",
            'echo "SETUP_COMPLETE"',
            "",
        )
    )


def setup_text_for(profile: Profile, *, workspace: str) -> str:
    if not profile.comfy.get("commit"):
        raise BlockedError("profile ComfyUI commit is missing")
    return render_setup(profile, workspace=workspace)


def local_setup_path(repo_root: Path | str, workspace: str) -> Path:
    # Used only as a local staging name; never written into the repository.
    return Path(repo_root) / "data" / "i1" / "staging" / "setup.sh"


__all__ = ["local_setup_path", "render_setup", "setup_text_for"]
