"""Capture a command's raw bytes without PowerShell UTF-16/UTF-8 corruption.

Usage::

    python scripts/kf1/evidence_capture.py --out DIR --label NAME -- CMD ARGS...

Writes exact ``.stdout.bin``/``.stderr.bin`` bytes, a decoded UTF-8 convenience
copy (BOM-aware; UTF-8 with replacement only when the stream is not text),
``meta.json`` with argv/cwd/start/end/python/exit, and
``conversion-index.json`` describing raw vs decoded files and hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _decode_text(data: bytes, requested: str) -> tuple[str, str]:
    if data.startswith(b"\xff\xfe"):
        return data[2:].decode("utf-16-le", errors="replace"), "utf-16-le-bom"
    if data.startswith(b"\xfe\xff"):
        return data[2:].decode("utf-16-be", errors="replace"), "utf-16-be-bom"
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8", errors="replace"), "utf-8-bom"
    if requested and requested != "auto":
        return data.decode(requested, errors="replace"), requested
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace"), "utf-8-replace"


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kf1-evidence-capture")
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--encoding", default="auto")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("command is required after --")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started_at = _now_iso()
    started_epoch = time.time()
    completed = subprocess.run(command, capture_output=True, check=False)
    finished_epoch = time.time()
    finished_at = _now_iso()
    stdout = completed.stdout
    stderr = completed.stderr
    stdout_text, stdout_encoding = _decode_text(stdout, args.encoding)
    stderr_text, stderr_encoding = _decode_text(stderr, args.encoding)
    prefix = out / args.label

    _write_bytes(prefix.with_suffix(".stdout.bin"), stdout)
    _write_bytes(prefix.with_suffix(".stderr.bin"), stderr)
    _write_bytes(
        prefix.with_suffix(".stdout.utf8.txt"),
        stdout_text.encode("utf-8", errors="strict"),
    )
    _write_bytes(
        prefix.with_suffix(".stderr.utf8.txt"),
        stderr_text.encode("utf-8", errors="strict"),
    )
    meta: dict[str, Any] = {
        "schema": "anime-studio/kf1-evidence-capture/v1",
        "label": args.label,
        "argv": command,
        "cwd": str(Path.cwd()),
        "started_at": started_at,
        "started_epoch": started_epoch,
        "finished_at": finished_at,
        "finished_epoch": finished_epoch,
        "duration_seconds": round(finished_epoch - started_epoch, 3),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "exit_code": int(completed.returncode),
        "stdout_bytes": len(stdout),
        "stderr_bytes": len(stderr),
        "stdout_sha256": _sha(stdout),
        "stderr_sha256": _sha(stderr),
        "stdout_encoding_used": stdout_encoding,
        "stderr_encoding_used": stderr_encoding,
        "note": "raw .bin files are exact; .utf8.txt are explicit decoded copies",
    }
    _write_bytes(
        prefix.with_suffix(".meta.json"),
        (json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
            "utf-8"
        ),
    )
    _write_bytes(
        prefix.with_suffix(".conversion-index.json"),
        (
            json.dumps(
                {
                    "label": args.label,
                    "raw_stdout": f"{args.label}.stdout.bin",
                    "raw_stderr": f"{args.label}.stderr.bin",
                    "decoded_stdout": f"{args.label}.stdout.utf8.txt",
                    "decoded_stderr": f"{args.label}.stderr.utf8.txt",
                    "stdout_encoding": stdout_encoding,
                    "stderr_encoding": stderr_encoding,
                    "raw_stdout_sha256": meta["stdout_sha256"],
                    "raw_stderr_sha256": meta["stderr_sha256"],
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8"),
    )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
