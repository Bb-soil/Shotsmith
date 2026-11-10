"""Staged as ``v3_download_worker.py``; attests birth then runs the real worker."""

from __future__ import annotations

import hashlib
import json
import runpy
import sys
from pathlib import Path


def main() -> int:
    import i1_child_entry  # staged next to this shim

    attempt = Path(__file__).resolve().parent
    i1_child_entry.write_birth(
        kind="download_worker", manifest_path=attempt / "launch.json"
    )
    impl = attempt / "i1_download_worker_impl.py"
    if not impl.is_file() or impl.is_symlink():
        print("download implementation is missing", file=sys.stderr)
        return 2
    try:
        job = json.loads(Path(sys.argv[1]).read_bytes().decode("utf-8"))
        expected = str(job["worker_impl_sha256"])
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        print("download job implementation hash is missing", file=sys.stderr)
        return 2
    digest = hashlib.sha256()
    with impl.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        print("download implementation hash mismatch", file=sys.stderr)
        return 2
    sys.argv = [str(impl), *sys.argv[1:]]
    runpy.run_path(str(impl), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
