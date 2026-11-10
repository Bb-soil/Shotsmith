from __future__ import annotations

import json
from pathlib import Path

import pytest

from anime_studio.__main__ import main
from anime_studio.errors import P1ConfigError
from anime_studio.p1_config import (
    P1RuntimeConfig,
    check_readiness,
    load_config,
    validate_manifest,
)
from anime_studio.settings import Settings
from anime_studio.workflow import workflow_sha256
from tests.p1_helpers import (
    PNG_FIXTURE,
    make_manifest,
    write_config,
    write_manifest,
    write_workflow,
)


ROOT = Path(__file__).resolve().parents[1]


def _valid_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    workflow_path = tmp_path / "workflow.json"
    write_workflow(workflow_path)
    workflow_hash = workflow_sha256(workflow_path)
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, make_manifest(workflow_sha256=workflow_hash))
    input_path = tmp_path / "input.png"
    input_path.write_bytes(PNG_FIXTURE)
    config_path = write_config(
        tmp_path,
        base_url="http://127.0.0.1:8188",
        workflow_path=workflow_path,
        manifest_path=manifest_path,
        input_image_path=input_path,
        workflow_sha256=workflow_hash,
    )
    return workflow_path, manifest_path, config_path


def test_repository_draft_is_structurally_valid_but_live_blocked() -> None:
    readiness = check_readiness(ROOT / "config" / "p1.example.json")
    assert readiness.live_ready is False
    assert readiness.workflow_sha256
    assert any("TEMPLATE_NOT_EXECUTABLE" in issue for issue in readiness.issues)
    assert any("input image" in issue for issue in readiness.issues)


def test_fully_pinned_manifest_can_be_live_ready() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        tmp_path = Path(raw)
        _, _, config_path = _valid_files(tmp_path)
        readiness = check_readiness(config_path, base_dir=tmp_path)
        assert readiness.live_ready is True, readiness.issues


def test_core_only_empty_custom_nodes_is_allowed() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        tmp_path = Path(raw)
        workflow_path, manifest_path, config_path = _valid_files(tmp_path)
        manifest = make_manifest(workflow_sha256=workflow_sha256(workflow_path))
        manifest["custom_nodes"] = []
        write_manifest(manifest_path, manifest)
        readiness = check_readiness(config_path, base_dir=tmp_path)
        assert readiness.live_ready is True, readiness.issues


@pytest.mark.parametrize("bad_value", [None, {}, "node", 0, True])
def test_custom_nodes_must_be_an_explicit_list(tmp_path, bad_value) -> None:
    workflow_path, manifest_path, config_path = _valid_files(tmp_path)
    manifest = make_manifest(workflow_sha256=workflow_sha256(workflow_path))
    manifest["custom_nodes"] = bad_value
    write_manifest(manifest_path, manifest)
    readiness = check_readiness(config_path, base_dir=tmp_path)
    assert readiness.live_ready is False
    assert any("custom_nodes" in issue for issue in readiness.issues)


def test_missing_custom_nodes_key_is_rejected(tmp_path) -> None:
    workflow_path, manifest_path, config_path = _valid_files(tmp_path)
    manifest = make_manifest(workflow_sha256=workflow_sha256(workflow_path))
    del manifest["custom_nodes"]
    write_manifest(manifest_path, manifest)
    readiness = check_readiness(config_path, base_dir=tmp_path)
    assert readiness.live_ready is False
    assert any("custom_nodes" in issue for issue in readiness.issues)


def test_nonempty_custom_nodes_keep_pin_source_and_credential_checks(tmp_path) -> None:
    workflow_path, manifest_path, config_path = _valid_files(tmp_path)
    manifest = make_manifest(workflow_sha256=workflow_sha256(workflow_path))
    manifest["custom_nodes"] = [
        {
            "name": "",
            "commit": "not-a-commit",
            "source_url": "NOT_VERIFIED",
            "token": "must-not-be-accepted",
        }
    ]
    write_manifest(manifest_path, manifest)
    readiness = check_readiness(config_path, base_dir=tmp_path)
    assert readiness.live_ready is False
    assert any("custom_nodes[0].name" in issue for issue in readiness.issues)
    assert any("custom_nodes[0].commit" in issue for issue in readiness.issues)
    assert any("custom_nodes[0].source_url" in issue for issue in readiness.issues)
    assert any("credential-like" in issue for issue in readiness.issues)


def test_prepared_wan_draft_is_blocked_only_by_draft_status() -> None:
    readiness = check_readiness(ROOT / "config" / "p1.wan22-5b.draft.json")
    assert readiness.live_ready is False
    assert list(readiness.issues) == ["manifest.status is not VERIFIED"]
    assert (
        readiness.workflow_sha256
        == "abac4f35c05f22e0d8d517bdc569bee8f7419e48c81753c90ad8eedf310b2576"
    )
    assert readiness.workflow["58"]["inputs"]["format.codec"] == "h264"


def test_status_verified_is_not_trusted_when_fields_are_unknown() -> None:
    with pytest.raises(P1ConfigError):
        # A malformed config must fail before a manifest can be trusted.
        P1RuntimeConfig.from_dict({})

    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        tmp_path = Path(raw)
        workflow_path, manifest_path, config_path = _valid_files(tmp_path)
        manifest = make_manifest(
            workflow_sha256=workflow_sha256(workflow_path),
            status="VERIFIED",
            verified_fields=False,
        )
        write_manifest(manifest_path, manifest)
        readiness = check_readiness(config_path, base_dir=tmp_path)
        assert readiness.live_ready is False
        assert any("sha256" in issue for issue in readiness.issues)


def test_manifest_hash_mismatch_and_license_unknown_block_live() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        tmp_path = Path(raw)
        workflow_path, manifest_path, config_path = _valid_files(tmp_path)
        manifest = make_manifest(workflow_sha256="f" * 64)
        write_manifest(manifest_path, manifest)
        readiness = check_readiness(config_path, base_dir=tmp_path)
        assert readiness.live_ready is False
        assert any("does not match" in issue for issue in readiness.issues)

    with tempfile.TemporaryDirectory() as raw:
        tmp_path = Path(raw)
        workflow_path, manifest_path, config_path = _valid_files(tmp_path)
        manifest = make_manifest(workflow_sha256=workflow_sha256(workflow_path))
        manifest["models"][0]["hosted_use"] = "unknown"
        manifest["models"][0]["license_url"] = "NOT_VERIFIED"
        write_manifest(manifest_path, manifest)
        readiness = check_readiness(config_path, base_dir=tmp_path)
        assert readiness.live_ready is False
        assert any("hosted_use" in issue for issue in readiness.issues)


def test_manifest_credential_like_key_blocks_live() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        tmp_path = Path(raw)
        workflow_path, manifest_path, config_path = _valid_files(tmp_path)
        manifest = make_manifest(workflow_sha256=workflow_sha256(workflow_path))
        manifest["api_token"] = "should-never-be-committed"
        write_manifest(manifest_path, manifest)
        readiness = check_readiness(config_path, base_dir=tmp_path)
        assert readiness.live_ready is False
        assert any("credential-like" in issue for issue in readiness.issues)


@pytest.mark.parametrize(
    "base_url",
    [
        "https://user:pass@example.invalid",
        "https://example.invalid?token=x",
        "https://example.invalid/#frag",
    ],
)
def test_config_rejects_userinfo_and_query_credentials(tmp_path, base_url) -> None:
    workflow_path = tmp_path / "workflow.json"
    write_workflow(workflow_path)
    manifest_path = tmp_path / "manifest.json"
    write_manifest(
        manifest_path,
        make_manifest(workflow_sha256=workflow_sha256(workflow_path)),
    )
    config_path = write_config(
        tmp_path,
        base_url=base_url,
        workflow_path=workflow_path,
        manifest_path=manifest_path,
        workflow_sha256=workflow_sha256(workflow_path),
    )
    with pytest.raises(P1ConfigError):
        load_config(config_path, base_dir=tmp_path)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_timeout_is_rejected(tmp_path, bad_value) -> None:
    workflow_path = tmp_path / "workflow.json"
    write_workflow(workflow_path)
    manifest_path = tmp_path / "manifest.json"
    write_manifest(
        manifest_path,
        make_manifest(workflow_sha256=workflow_sha256(workflow_path)),
    )
    config_path = write_config(
        tmp_path,
        base_url="http://127.0.0.1:8188",
        workflow_path=workflow_path,
        manifest_path=manifest_path,
        workflow_sha256=workflow_sha256(workflow_path),
    )
    data = json.loads(config_path.read_text(encoding="utf-8"))
    data["runtime"]["read_timeout_seconds"] = bad_value
    config_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(P1ConfigError):
        load_config(config_path, base_dir=tmp_path)


def test_non_loopback_without_token_env_is_rejected(tmp_path) -> None:
    workflow_path = tmp_path / "workflow.json"
    write_workflow(workflow_path)
    manifest_path = tmp_path / "manifest.json"
    write_manifest(
        manifest_path,
        make_manifest(workflow_sha256=workflow_sha256(workflow_path)),
    )
    config_path = write_config(
        tmp_path,
        base_url="https://example.invalid",
        workflow_path=workflow_path,
        manifest_path=manifest_path,
        workflow_sha256=workflow_sha256(workflow_path),
    )
    data = json.loads(config_path.read_text(encoding="utf-8"))
    data["runtime"]["token_env"] = ""
    config_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(P1ConfigError):
        load_config(config_path, base_dir=tmp_path)


def test_settings_require_explicit_opt_in_and_config(tmp_path) -> None:
    with pytest.raises(ValueError):
        Settings(
            db_path=tmp_path / "db.sqlite3",
            artifacts_dir=tmp_path / "artifacts",
            adapter_mode="comfy-http",
        )
    with pytest.raises(ValueError):
        Settings(
            db_path=tmp_path / "db.sqlite3",
            artifacts_dir=tmp_path / "artifacts",
            adapter_mode="comfy-http",
            allow_live_transport=True,
        )
    default = Settings(
        db_path=tmp_path / "db.sqlite3",
        artifacts_dir=tmp_path / "artifacts",
    )
    assert default.adapter_mode == "mock"


def test_p1_check_cli_draft_exit_zero_and_malformed_exit_two(tmp_path, capsys) -> None:
    assert main(["p1-check", "--config", str(ROOT / "config" / "p1.example.json")]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["live_ready"] is False

    bad = tmp_path / "bad.json"
    bad.write_text("{not-json", encoding="utf-8")
    assert main(["p1-check", "--config", str(bad)]) == 2


def test_serve_real_mode_refuses_without_opt_in_or_config(tmp_path) -> None:
    common = [
        "serve",
        "--host",
        "127.0.0.1",
        "--db",
        str(tmp_path / "db.sqlite3"),
        "--artifacts",
        str(tmp_path / "artifacts"),
    ]
    assert main([*common, "--adapter", "comfy-http"]) == 2
    assert main([*common, "--adapter", "comfy-http", "--allow-live-transport"]) == 2
