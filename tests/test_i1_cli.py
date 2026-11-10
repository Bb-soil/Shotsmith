from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts.i1 import cli


def test_parser_has_exactly_seven_commands() -> None:
    parser = cli._build_parser()
    actions = [action for action in parser._actions if action.dest == "command"]
    assert actions
    assert set(actions[0].choices) == {
        "connect",
        "inspect",
        "install",
        "status",
        "resume",
        "open",
        "disconnect",
    }


def test_install_without_execute_is_json_plan_only(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    ctx = SimpleNamespace(profile=SimpleNamespace(profile_id="p"))
    plan = {
        "inspection": {"blockers": [], "warnings": [], "facts": {}},
        "plan": {"profile_id": "p", "models": [], "blockers": []},
    }
    monkeypatch.setattr(cli.ops, "load_context", lambda **kwargs: ctx)
    monkeypatch.setattr(cli.ops, "plan_install", lambda *args, **kwargs: plan)
    assert cli.main(["install", "--name", "x", "--json"]) == 0
    output = capsys.readouterr().out.strip()
    payload = json.loads(output)
    assert payload["status"] == "PLAN_ONLY"
    assert output.count("\n{") == 0


def test_status_offline_is_one_json_object(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    ctx = SimpleNamespace(profile=SimpleNamespace(profile_id="p"))
    monkeypatch.setattr(cli.ops, "load_context", lambda **kwargs: ctx)
    monkeypatch.setattr(cli.ops, "status", lambda ctx, offline=False: {"status": "INSTALLED", "offline": offline})
    assert cli.main(["status", "--name", "x", "--offline", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "INSTALLED"


def test_connect_noninteractive_requires_fingerprint(
    monkeypatch: pytest.MonkeyPatch, capsys, tmp_path
) -> None:
    from scripts.i1 import hosttrust

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    b64 = "aTEtaG9zdC1rZXk="
    monkeypatch.setattr(
        cli.ops,
        "scan_host",
        lambda **kwargs: [
            hosttrust.HostKey(
                "ssh-ed25519", b64, "SHA256:fake", "example.test ssh-ed25519 " + b64
            )
        ],
    )
    code = cli.main(
        [
            "connect",
            "--host",
            "example.test",
            "--user",
            "root",
            "--port",
            "22",
            "--workspace",
            "/workspace",
            "--agent",
            "--state-dir",
            str(tmp_path),
            "--json",
        ]
    )
    # It fails before a real SSH attempt because unknown host has no accepted fingerprint.
    assert code in {2, 3}
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "ERROR"


def test_progress_prefers_worker_verify_stage_and_current_file(capsys) -> None:
    printer = cli.ProgressPrinter()
    printer.last_at = 0.0
    printer(
        {
            "stage": "DOWNLOAD",
            "progress": {
                "stage": "VERIFYING",
                "current_file": "wan-high.safetensors",
                "files_done": 1,
                "files_total": 4,
                "bytes_done": 10,
                "bytes_total": 100,
                "heartbeat_at": 0,
            },
        }
    )
    out = capsys.readouterr().out
    assert "VERIFYING" in out
    assert "wan-high.safetensors" in out
    assert "1/4" in out


def test_redact_removes_secret_values_from_cli_errors() -> None:
    from scripts.i1.util import redact

    text = (
        "Authorization: Bearer sk-SYNTHETIC-123\n"
        "token=SYNTHETIC-456 password=hunter2\n"
        "https://user:pass@example.test/x?api_key=SYNTHETIC-789\n"
        "-----BEGIN OPENSSH PRIVATE KEY-----\nBODY-SECRET\n"
        "-----END OPENSSH PRIVATE KEY-----"
    )
    safe = redact(text)
    for secret in ("sk-SYNTHETIC-123", "SYNTHETIC-456", "hunter2", "pass@", "SYNTHETIC-789", "BODY-SECRET"):
        assert secret not in safe
