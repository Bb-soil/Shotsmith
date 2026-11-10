from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_i1_does_not_import_clore_or_owner_risk_through_side_effect() -> None:
    code = (
        "import sys;"
        "import scripts.i1.ops, scripts.i1.cli, scripts.i1.tunnel, scripts.i1.comfy;"
        "bad=[m for m in sys.modules if m.startswith('scripts.p1.clore') "
        "or m.startswith('scripts.p1.owner_risk') "
        "or m.startswith('scripts.p1.v3_lifecycle')];"
        "print(','.join(bad));"
        "raise SystemExit(1 if bad else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
