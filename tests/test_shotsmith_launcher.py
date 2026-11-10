import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("shell", ["powershell", "pwsh"])
@pytest.mark.parametrize("args", [[], ["worker", "--help"], ["project", "preview", "my project"]])
def test_launcher_preserves_empty_and_explicit_arguments(tmp_path, shell, args):
    executable = shutil.which(shell)
    if not executable:
        pytest.skip(f"{shell} unavailable")
    shutil.copyfile(ROOT / "shotsmith.ps1", tmp_path / "shotsmith.ps1")
    (tmp_path / "avs.ps1").write_text(
        "[CmdletBinding()] param([Parameter(ValueFromRemainingArguments=$true)][string[]]$CommandArgs)\n"
        "$values = @(); if ($null -ne $CommandArgs) { $values = @($CommandArgs) }\n"
        "ConvertTo-Json -InputObject $values -Compress\nexit 0\n",
        encoding="utf-8-sig",
    )
    result = subprocess.run(
        [executable, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(tmp_path / "shotsmith.ps1"), *args],
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == args
