from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.p1.preflight import (
    PreflightError,
    PreflightExecutionError,
    build_preflight_checks,
    build_preflight_plan,
    run_preflight_checks,
    validate_object_info,
)

ROOT = Path(__file__).resolve().parents[1]
OBJECT_INFO_FIXTURE = (
    ROOT / "tests" / "fixtures" / "p1" / "savevideo-object_info-7a283320.json"
)


def _object_info() -> dict[str, object]:
    return json.loads(OBJECT_INFO_FIXTURE.read_text(encoding="utf-8"))


def _savevideo(data: dict[str, object]) -> dict[str, object]:
    return data["SaveVideo"]["input"]  # type: ignore[index]


def _format_options(data: dict[str, object]) -> list[dict[str, object]]:
    return _savevideo(data)["required"]["format"][1]["options"]  # type: ignore[index]


def _option(options: list[dict[str, object]], key: str) -> dict[str, object]:
    return next(option for option in options if option.get("key") == key)


def _mp4_codec(data: dict[str, object]) -> list[object]:
    return _option(_format_options(data), "mp4")["inputs"]["required"]["codec"]  # type: ignore[index]


def _h264_encoding(data: dict[str, object]) -> list[object]:
    return _option(_mp4_codec(data)[1]["options"], "h264")["inputs"]["optional"]["encoding"]  # type: ignore[index]


def test_preflight_plan_is_dry_run_and_loopback_only() -> None:
    plan = build_preflight_plan(workspace="/workspace")
    rendered = "\n".join(plan)
    assert "nvidia-smi" in rendered
    assert "ffmpeg -version" in rendered
    assert "ffprobe -version" in rendered
    assert "pip check" in rendered
    assert "pip freeze" in rendered
    assert "import comfy, nodes" in rendered
    assert "/object_info" in rendered
    assert "/system_stats" in rendered
    assert "127.0.0.1:8189:127.0.0.1:8188" in rendered
    assert "127.0.0.1" in rendered and "8188" in rendered
    assert "public" in rendered.lower()
    assert "| tee" not in rendered
    assert "--execute" in rendered
    with pytest.raises(PreflightError):
        build_preflight_plan(workspace="/workspace", host="0.0.0.0")


def test_original_preflight_output_paths_support_filesystem_execution() -> None:
    # Do not replace output_path here: fixture conversion hid PurePosixPath
    # reaching run_preflight_checks, where parent.mkdir() failed on the worker.
    checks = build_preflight_checks(workspace="/workspace")
    for check in checks:
        assert isinstance(check.output_path, Path), check.name


def _checks_in_tmp(tmp_path):
    return tuple(
        replace(check, output_path=tmp_path / check.output_path.name)
        for check in build_preflight_checks(workspace="/workspace")
    )


def _runner(fail_index: int, mode: str):
    calls: list[tuple[str, ...]] = []

    def run(argv, **kwargs):
        calls.append(tuple(argv))
        assert kwargs.get("shell", False) is False
        index = len(calls) - 1
        if index == fail_index:
            if mode == "not-found":
                raise FileNotFoundError(argv[0])
            return SimpleNamespace(
                returncode=9,
                stdout="failed stdout",
                stderr="failed stderr",
            )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    return run, calls


@pytest.mark.parametrize("mode", ["failure", "not-found"])
def test_each_preflight_check_failure_stops_and_persists_evidence(tmp_path, mode) -> None:
    checks = _checks_in_tmp(tmp_path)
    for index, check in enumerate(checks):
        run, calls = _runner(index, mode)
        with pytest.raises(PreflightExecutionError) as exc_info:
            run_preflight_checks(checks, runner=run)
        assert exc_info.value.returncode == (127 if mode == "not-found" else 9)
        assert len(calls) == index + 1
        assert check.output_path.exists()
        assert check.output_path.with_name(check.output_path.name + ".stderr.txt").exists()
        if mode == "not-found":
            assert "command not found" in check.output_path.read_text(encoding="utf-8")
        else:
            assert "failed stdout" in check.output_path.read_text(encoding="utf-8")


def test_object_info_validator_accepts_source_shaped_dynamic_path() -> None:
    data = _object_info()
    assert data["_fixture_provenance"]["status"] == "SOURCE_RECONSTRUCTION_NOT_LIVE"
    assert validate_object_info(data) == []


def test_object_info_validator_reports_missing_class() -> None:
    data = _object_info()
    del data["Wan22ImageToVideoLatent"]
    issues = validate_object_info(data)
    assert any("Wan22ImageToVideoLatent" in issue for issue in issues)


def test_object_info_validator_rejects_flat_savevideo_schema() -> None:
    data = _object_info()
    data["SaveVideo"]["input"] = {
        "required": {
            "video": ["VIDEO", {}],
            "filename_prefix": ["STRING", {}],
            "format": ["COMBO"],
            "format.codec": ["COMBO"],
            "format.codec.encoding": ["COMBO"],
        },
        "optional": {},
    }
    issues = validate_object_info(data)
    assert issues
    assert any("COMFY_DYNAMICCOMBO_V3" in issue for issue in issues)


def test_object_info_validator_rejects_missing_mp4_branch() -> None:
    data = _object_info()
    options = _format_options(data)
    options[:] = [option for option in options if option.get("key") != "mp4"]
    issues = validate_object_info(data)
    assert any("format" in issue and "mp4" in issue for issue in issues)


def test_object_info_validator_rejects_missing_codec_in_mp4_branch() -> None:
    data = _object_info()
    mp4 = _option(_format_options(data), "mp4")
    del mp4["inputs"]["required"]["codec"]
    issues = validate_object_info(data)
    assert any("format" in issue and "codec" in issue for issue in issues)


def test_top_level_optional_codec_does_not_substitute_for_format_codec() -> None:
    data = _object_info()
    mp4 = _option(_format_options(data), "mp4")
    del mp4["inputs"]["required"]["codec"]
    assert "codec" in data["SaveVideo"]["input"]["optional"]
    issues = validate_object_info(data)
    assert issues
    assert any("option 'mp4'" in issue and "codec" in issue for issue in issues)


def test_codec_in_foreign_format_branch_does_not_confirm_mp4() -> None:
    data = _object_info()
    mp4 = _option(_format_options(data), "mp4")
    del mp4["inputs"]["required"]["codec"]
    mkv = _option(_format_options(data), "mkv")
    assert "codec" in mkv["inputs"]["required"]
    issues = validate_object_info(data)
    assert any(
        "format" in issue and "mp4" in issue and "codec" in issue for issue in issues
    )


def test_object_info_validator_rejects_missing_h264_option() -> None:
    data = _object_info()
    options = _mp4_codec(data)[1]["options"]
    options[:] = [option for option in options if option.get("key") != "h264"]
    issues = validate_object_info(data)
    assert any("h264" in issue for issue in issues)


def test_object_info_validator_rejects_missing_encoding_input() -> None:
    data = _object_info()
    h264 = _option(_mp4_codec(data)[1]["options"], "h264")
    del h264["inputs"]["optional"]["encoding"]
    issues = validate_object_info(data)
    assert any("encoding" in issue for issue in issues)


def test_object_info_validator_rejects_missing_auto_encoding() -> None:
    data = _object_info()
    options = _h264_encoding(data)[1]["options"]
    options[:] = [option for option in options if option.get("key") != "auto"]
    issues = validate_object_info(data)
    assert any("auto" in issue for issue in issues)


def _format_descriptor_none(data: dict[str, object]) -> None:
    _savevideo(data)["required"]["format"] = None


def _format_descriptor_short(data: dict[str, object]) -> None:
    _savevideo(data)["required"]["format"] = ["COMFY_DYNAMICCOMBO_V3"]


def _format_descriptor_wrong_marker(data: dict[str, object]) -> None:
    _savevideo(data)["required"]["format"] = ["WRONG_MARKER", {"options": []}]


def _format_options_mapping(data: dict[str, object]) -> None:
    _savevideo(data)["required"]["format"] = [
        "COMFY_DYNAMICCOMBO_V3",
        {"options": {}},
    ]


def _format_options_scalars(data: dict[str, object]) -> None:
    _savevideo(data)["required"]["format"] = [
        "COMFY_DYNAMICCOMBO_V3",
        {"options": [None, 1, "mp4"]},
    ]


def _mp4_option_scalar(data: dict[str, object]) -> None:
    options = _format_options(data)
    for index, option in enumerate(options):
        if option.get("key") == "mp4":
            options[index] = "mp4"  # type: ignore[call-overload]


def _mp4_inputs_none(data: dict[str, object]) -> None:
    _option(_format_options(data), "mp4")["inputs"] = None


def _mp4_inputs_empty(data: dict[str, object]) -> None:
    _option(_format_options(data), "mp4")["inputs"] = {}


def _mp4_required_none(data: dict[str, object]) -> None:
    _option(_format_options(data), "mp4")["inputs"]["required"] = None


def _mp4_required_list(data: dict[str, object]) -> None:
    _option(_format_options(data), "mp4")["inputs"]["required"] = []


def _h264_optional_none(data: dict[str, object]) -> None:
    _option(_mp4_codec(data)[1]["options"], "h264")["inputs"]["optional"] = None


def _h264_optional_list(data: dict[str, object]) -> None:
    _option(_mp4_codec(data)[1]["options"], "h264")["inputs"]["optional"] = []


@pytest.mark.parametrize(
    "mutator",
    [
        _format_descriptor_none,
        _format_descriptor_short,
        _format_descriptor_wrong_marker,
        _format_options_mapping,
        _format_options_scalars,
        _mp4_option_scalar,
        _mp4_inputs_none,
        _mp4_inputs_empty,
        _mp4_required_none,
        _mp4_required_list,
        _h264_optional_none,
        _h264_optional_list,
    ],
)
def test_object_info_validator_handles_malformed_dynamic_schema(mutator) -> None:
    data = _object_info()
    mutator(data)
    issues = validate_object_info(data)
    assert isinstance(issues, list)
    assert issues


def test_object_info_validator_rejects_duplicate_mp4_option() -> None:
    data = _object_info()
    options = _format_options(data)
    options.append(json.loads(json.dumps(_option(options, "mp4"))))
    issues = validate_object_info(data)
    assert any("duplicate" in issue.lower() for issue in issues)


def test_object_info_validator_rejects_ambiguous_child_in_required_and_optional() -> None:
    data = _object_info()
    mp4 = _option(_format_options(data), "mp4")
    mp4["inputs"]["optional"] = {"codec": mp4["inputs"]["required"]["codec"]}
    issues = validate_object_info(data)
    assert any("ambiguous" in issue.lower() for issue in issues)


@pytest.mark.parametrize("value", [None, [], "SaveVideo", 1])
def test_object_info_validator_handles_malformed_savevideo_container(value) -> None:
    data = _object_info()
    data["SaveVideo"] = value
    issues = validate_object_info(data)
    assert issues


def test_object_info_validator_handles_missing_savevideo_input() -> None:
    data = _object_info()
    data["SaveVideo"]["input"] = None
    issues = validate_object_info(data)
    assert any("format" in issue for issue in issues)
