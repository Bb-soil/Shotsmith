"""Dry-run P1 worker preflight checks and offline object_info validation.

The default CLI only prints an explicit argv plan. ``--execute`` is for a future,
separately approved Pod session and never performs a generation. Each check is
run without a shell; stdout/stderr are persisted and the original exit code is
honoured. A failed check stops the sequence before any later check.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

REQUIRED_CLASSES = (
    "KSampler",
    "CLIPTextEncode",
    "VAEDecode",
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "ModelSamplingSD3",
    "Wan22ImageToVideoLatent",
    "LoadImage",
    "CreateVideo",
    "SaveVideo",
)
REQUIRED_INPUTS = {
    "Wan22ImageToVideoLatent": {"width", "height", "length", "start_image"},
    "CreateVideo": {"fps"},
    "CLIPLoader": {"type", "device"},
}
SAVEVIDEO_INPUTS = {"format"}
_DYNAMIC_COMBO_MARKER = "COMFY_DYNAMICCOMBO_V3"
_SAVEVIDEO_DYNAMIC_PATH = (
    ("format", "mp4"),
    ("codec", "h264"),
    ("encoding", "auto"),
)
_MISSING = object()


class PreflightError(Exception):
    """A preflight plan is invalid or a saved object_info file is unusable."""


class PreflightExecutionError(PreflightError):
    """A mandatory preflight command failed; later checks must not run."""

    def __init__(self, check: "PreflightCheck", returncode: int, message: str) -> None:
        super().__init__(message)
        self.check = check
        self.returncode = returncode
        self.output_path = check.output_path


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    argv: tuple[str, ...]
    output_path: Path
    cwd: Path | None = None


@dataclass(frozen=True)
class PreflightResult:
    check: PreflightCheck
    returncode: int
    stdout_path: Path
    stderr_path: Path


def _validate_workspace(value: str) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise PreflightError("workspace must be a non-empty path string")
    path = PurePosixPath(value)
    if not path.is_absolute() or any(part in ("", ".", "..") for part in path.parts[1:]):
        raise PreflightError("workspace must be an absolute normalized POSIX path")
    return str(path)


def _json_fetch_code(url: str) -> str:
    return (
        "import json, urllib.request; "
        f"print(json.dumps(json.load(urllib.request.urlopen({url!r}, timeout=30)), indent=2))"
    )


def build_preflight_checks(
    *,
    workspace: str = "/workspace",
    comfyui_dir: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8188,
) -> tuple[PreflightCheck, ...]:
    if host != "127.0.0.1":
        raise PreflightError("preflight only targets the loopback worker")
    if isinstance(port, bool) or not isinstance(port, int) or not (1 <= port <= 65535):
        raise PreflightError("port must be a valid integer")
    workspace_text = _validate_workspace(workspace)
    comfy = (
        _validate_workspace(comfyui_dir)
        if comfyui_dir is not None
        else str(PurePosixPath(workspace_text) / "ComfyUI")
    )
    venv = PurePosixPath(workspace_text) / "p1-venv"
    evidence = PurePosixPath(workspace_text) / "p1-evidence"
    python = str(venv / "bin/python")
    local_url = f"http://{host}:{port}"
    object_info = Path(str(evidence / "object_info.json"))

    def output(name: str) -> Path:
        return Path(str(evidence / name))

    return (
        PreflightCheck("nvidia-smi", ("nvidia-smi",), output("nvidia-smi.txt")),
        PreflightCheck("ffmpeg-version", ("ffmpeg", "-version"), output("ffmpeg-version.txt")),
        PreflightCheck("ffprobe-version", ("ffprobe", "-version"), output("ffprobe-version.txt")),
        PreflightCheck(
            "pip-check",
            (python, "-m", "pip", "check"),
            output("pip-check.txt"),
        ),
        PreflightCheck(
            "pip-freeze",
            (python, "-m", "pip", "freeze"),
            output("pip-freeze.txt"),
        ),
        PreflightCheck(
            "core-imports",
            (
                python,
                "-c",
                "import sys; sys.path.insert(0, '.'); import comfy, nodes",
            ),
            output("core-imports.txt"),
            cwd=Path(comfy),
        ),
        PreflightCheck(
            "system-stats",
            (python, "-c", _json_fetch_code(f"{local_url}/system_stats")),
            output("system_stats.json"),
        ),
        PreflightCheck(
            "object-info",
            (python, "-c", _json_fetch_code(f"{local_url}/object_info")),
            object_info,
        ),
        PreflightCheck(
            "object-info-validation",
            (python, "scripts/p1/preflight.py", "--validate-object-info", str(object_info)),
            output("object-info-validation.txt"),
        ),
    )


def render_preflight_plan(checks: tuple[PreflightCheck, ...], *, host: str, port: int, comfy: str, python: str) -> tuple[str, ...]:
    lines = ["# P1 preflight checks (dry-run; explicit --execute is a separate approval)"]
    for check in checks:
        lines.append(f"# check: {check.name}")
        if check.cwd is not None:
            lines.append(f"# cwd: {check.cwd}")
        lines.append(shlex.join(check.argv))
        lines.append(f"# stdout evidence: {check.output_path}")
        lines.append(f"# stderr evidence: {_stderr_path(check.output_path)}")
    lines.extend(
        (
            f"# Worker command: {python} {shlex.quote(str(PurePosixPath(comfy) / 'main.py'))} --listen {host} --port {port}",
            "# PC tunnel: ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8189:127.0.0.1:8188 -p <SSH_PORT> -i <LOCAL_KEY> root@<POD_IP>",
            "# Public ports 8188/8080/8888 and Jupyter are not exposed; only SSH TCP22 is opened.",
            "# This plan performs no generation and does not prove GPU, VRAM, timing or encoding.",
        )
    )
    return tuple(lines)


def build_preflight_plan(
    *,
    workspace: str = "/workspace",
    comfyui_dir: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8188,
) -> tuple[str, ...]:
    checks = build_preflight_checks(
        workspace=workspace,
        comfyui_dir=comfyui_dir,
        host=host,
        port=port,
    )
    workspace_text = _validate_workspace(workspace)
    comfy = comfyui_dir or str(PurePosixPath(workspace_text) / "ComfyUI")
    python = str(PurePosixPath(workspace_text) / "p1-venv" / "bin" / "python")
    return render_preflight_plan(checks, host=host, port=port, comfy=comfy, python=python)


def _stderr_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".stderr.txt")


def run_preflight_checks(
    checks: tuple[PreflightCheck, ...],
    *,
    runner: Any | None = None,
    timeout_seconds: float = 120.0,
) -> list[PreflightResult]:
    """Run checks in order without a shell, persisting stdout/stderr and exit status."""

    run = runner or subprocess.run
    results: list[PreflightResult] = []
    for check in checks:
        check.output_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path = _stderr_path(check.output_path)
        try:
            completed = run(
                list(check.argv),
                cwd=str(check.cwd) if check.cwd is not None else None,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
        except FileNotFoundError as exc:
            message = f"{check.name}: command not found ({check.argv[0]})"
            check.output_path.write_text(message + "\n", encoding="utf-8")
            stderr_path.write_text(message + "\n", encoding="utf-8")
            raise PreflightExecutionError(check, 127, message) from exc
        except OSError as exc:
            message = f"{check.name}: could not start check: {type(exc).__name__}"
            check.output_path.write_text(message + "\n", encoding="utf-8")
            stderr_path.write_text(message + "\n", encoding="utf-8")
            raise PreflightExecutionError(check, 126, message) from exc

        stdout = completed.stdout if isinstance(completed.stdout, str) else ""
        stderr = completed.stderr if isinstance(completed.stderr, str) else ""
        check.output_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        returncode = int(getattr(completed, "returncode", 1))
        result = PreflightResult(check, returncode, check.output_path, stderr_path)
        results.append(result)
        if returncode != 0:
            raise PreflightExecutionError(
                check,
                returncode,
                f"{check.name} exited {returncode}; stopped before later checks",
            )
    return results


def _input_names(spec: Mapping[str, Any]) -> set[str]:
    inputs = spec.get("input")
    if not isinstance(inputs, Mapping):
        return set()
    names: set[str] = set()
    for section in ("required", "optional"):
        values = inputs.get(section)
        if isinstance(values, Mapping):
            names.update(str(key) for key in values)
        elif isinstance(values, list):
            names.update(str(item) for item in values if isinstance(item, str))
    # Some saved object_info formats return a flat input mapping.
    if not names:
        names.update(str(key) for key in inputs)
    return names


def _dynamic_options(spec: Any) -> list[Any] | None:
    """Return options for an exact ``COMFY_DYNAMICCOMBO_V3`` descriptor."""

    if not isinstance(spec, (list, tuple)) or len(spec) != 2:
        return None
    if spec[0] != _DYNAMIC_COMBO_MARKER:
        return None
    payload = spec[1]
    if not isinstance(payload, Mapping):
        return None
    options = payload.get("options")
    if not isinstance(options, list):
        return None
    return options


def _selected_dynamic_option(
    options: list[Any], selected: str
) -> tuple[Mapping[str, Any] | None, str | None]:
    matches = [
        option
        for option in options
        if isinstance(option, Mapping) and option.get("key") == selected
    ]
    if not matches:
        return None, "missing"
    if len(matches) > 1:
        return None, "duplicate"
    return matches[0], None


def _child_dynamic_input(
    option: Mapping[str, Any], field: str
) -> tuple[Any, str | None]:
    inputs = option.get("inputs", _MISSING)
    if not isinstance(inputs, Mapping):
        return _MISSING, "inputs must be an object"
    present = False
    found: list[Any] = []
    for section in ("required", "optional"):
        if section not in inputs:
            continue
        present = True
        values = inputs[section]
        if not isinstance(values, Mapping):
            return _MISSING, f"inputs.{section} must be an object"
        if field in values:
            found.append(values[field])
    if not present:
        return _MISSING, "selected option has no required/optional inputs"
    if not found:
        return _MISSING, f"missing input {field!r}"
    if len(found) > 1:
        return _MISSING, f"ambiguous input {field!r} in required/optional"
    return found[0], None


def _top_level_dynamic_input(
    save: Mapping[str, Any], field: str
) -> tuple[Any, str | None]:
    inputs = save.get("input")
    if not isinstance(inputs, Mapping):
        return _MISSING, "input must be an object"
    found: list[Any] = []
    for section in ("required", "optional"):
        values = inputs.get(section)
        if isinstance(values, Mapping) and field in values:
            found.append(values[field])
    if not found:
        return _MISSING, f"missing input {field!r}"
    if len(found) > 1:
        return _MISSING, f"ambiguous input {field!r} in required/optional"
    return found[0], None


def _validate_savevideo_dynamic(save: Mapping[str, Any], issues: list[str]) -> None:
    """Follow only the selected mp4 -> h264 -> auto DynamicCombo path."""

    current, error = _top_level_dynamic_input(save, "format")
    if error is not None:
        issues.append(f"SaveVideo dynamic field 'format' is unavailable: {error}")
        return
    labels: list[str] = []
    for index, (field, selected) in enumerate(_SAVEVIDEO_DYNAMIC_PATH):
        labels.append(field)
        label = ".".join(labels)
        options = _dynamic_options(current)
        if options is None:
            issues.append(
                f"SaveVideo dynamic field {label!r} is not a valid "
                "'COMFY_DYNAMICCOMBO_V3' descriptor with an options list"
            )
            return
        option, error = _selected_dynamic_option(options, selected)
        if option is None:
            if error == "duplicate":
                issues.append(
                    f"SaveVideo dynamic field {label!r} has duplicate/ambiguous "
                    f"selected option {selected!r}"
                )
            else:
                issues.append(
                    f"SaveVideo dynamic field {label!r} is missing selected "
                    f"option {selected!r}"
                )
            return
        if index == len(_SAVEVIDEO_DYNAMIC_PATH) - 1:
            return
        next_field = _SAVEVIDEO_DYNAMIC_PATH[index + 1][0]
        current, error = _child_dynamic_input(option, next_field)
        if error is not None:
            issues.append(
                f"SaveVideo dynamic field {label!r} option {selected!r} has "
                f"malformed inputs: {error}"
            )
            return


def validate_object_info(data: Mapping[str, Any]) -> list[str]:
    if not isinstance(data, Mapping):
        return ["object_info must be an object"]
    issues: list[str] = []
    for class_name in REQUIRED_CLASSES:
        if class_name not in data:
            issues.append(f"object_info is missing native class {class_name!r}")
            continue
        spec = data[class_name]
        if not isinstance(spec, Mapping):
            issues.append(f"object_info class {class_name!r} is not an object")
            continue
        names = _input_names(spec)
        for required in REQUIRED_INPUTS.get(class_name, set()):
            if required not in names:
                issues.append(f"{class_name} is missing input {required!r}")
    save = data.get("SaveVideo")
    if isinstance(save, Mapping):
        names = _input_names(save)
        for required in SAVEVIDEO_INPUTS:
            if required not in names:
                issues.append(f"SaveVideo is missing dynamic field {required!r}")
        if "format" in names:
            _validate_savevideo_dynamic(save, issues)
    return issues


def _savevideo_dynamic_facts(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return exact nested DynamicCombo facts for the selected mp4/h264 path."""

    save = data.get("SaveVideo")
    if not isinstance(save, Mapping):
        return {"path": list(SAVEVIDEO_INPUTS), "selected": {}, "issues": ["SaveVideo is missing"]}
    selected: dict[str, str] = {}
    issues: list[str] = []
    current, error = _top_level_dynamic_input(save, "format")
    if error is not None:
        return {
            "path": [field for field, _ in _SAVEVIDEO_DYNAMIC_PATH],
            "selected": selected,
            "issues": [f"SaveVideo dynamic field 'format' is unavailable: {error}"],
        }
    for index, (field, selected_value) in enumerate(_SAVEVIDEO_DYNAMIC_PATH):
        options = _dynamic_options(current)
        if options is None:
            issues.append(
                f"SaveVideo dynamic field {field!r} is not a valid "
                "'COMFY_DYNAMICCOMBO_V3' descriptor"
            )
            break
        option, option_error = _selected_dynamic_option(options, selected_value)
        if option is None:
            issues.append(
                f"SaveVideo dynamic field {field!r} missing/ambiguous option {selected_value!r}"
            )
            break
        selected[field] = selected_value
        if index == len(_SAVEVIDEO_DYNAMIC_PATH) - 1:
            break
        next_field = _SAVEVIDEO_DYNAMIC_PATH[index + 1][0]
        current, child_error = _child_dynamic_input(option, next_field)
        if child_error is not None:
            issues.append(
                f"SaveVideo dynamic field {field!r} option {selected_value!r} "
                f"has malformed inputs: {child_error}"
            )
            break
    return {
        "path": [field for field, _ in _SAVEVIDEO_DYNAMIC_PATH],
        "selected": selected,
        "issues": issues,
    }


def summarize_object_info(data: Mapping[str, Any]) -> dict[str, Any]:
    """Compact, hash-friendly facts; never returns the full object_info dump."""

    if not isinstance(data, Mapping):
        return {
            "classes": [],
            "missing_classes": list(REQUIRED_CLASSES),
            "validation_issues": ["object_info must be an object"],
            "input_names": {},
            "dynamic_combo": _savevideo_dynamic_facts({}),
        }
    present = [name for name in REQUIRED_CLASSES if isinstance(data.get(name), Mapping)]
    missing = [name for name in REQUIRED_CLASSES if name not in data]
    input_names: dict[str, list[str]] = {}
    for name in present:
        spec = data.get(name)
        if isinstance(spec, Mapping):
            input_names[name] = sorted(_input_names(spec))
    return {
        "classes": present,
        "missing_classes": missing,
        "validation_issues": validate_object_info(data),
        "input_names": input_names,
        "dynamic_combo": _savevideo_dynamic_facts(data),
    }


def load_object_info(path: Path | str) -> Mapping[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"cannot read object_info {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise PreflightError("object_info must be a JSON object")
    return data


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/p1/preflight.py",
        description="Print dry-run P1 worker preflight commands; execute only separately.",
    )
    parser.add_argument("--workspace", default="/workspace")
    parser.add_argument("--comfyui-dir", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--validate-object-info", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.validate_object_info:
            issues = validate_object_info(load_object_info(args.validate_object_info))
            for issue in issues:
                print(issue)
            return 1 if issues else 0
        plan = build_preflight_plan(
            workspace=args.workspace,
            comfyui_dir=args.comfyui_dir,
        )
        if not args.execute:
            print("\n".join(plan))
            return 0
        checks = build_preflight_checks(
            workspace=args.workspace,
            comfyui_dir=args.comfyui_dir,
        )
        for result in run_preflight_checks(checks):
            print(f"{result.check.name}: exit {result.returncode}")
        return 0
    except (PreflightError, PreflightExecutionError, OSError) as exc:
        print(f"preflight failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
