"""P1 access configuration, manifest gate and offline readiness checks.

No secrets are stored or read here. A config names an environment variable
(``token_env``); the live transport resolves it only when it is explicitly
constructed. Offline ``p1-check`` never looks at the variable's value.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from .errors import P1ConfigError, P1ManifestError, WorkflowBindingError
from .workflow import (
    TEMPLATE_MARKER,
    WorkflowBinding,
    load_workflow,
    workflow_sha256,
)


ROOT = Path(__file__).resolve().parent.parent
CONFIG_SCHEMA = "anime-video-studio/p1-config/v1"
MANIFEST_SCHEMA = "anime-video-studio/p1-manifest/v1"
CANDIDATE_SCHEMA = "anime-video-studio/p1-live-candidate/v1"
SOURCE_APPROVAL_SCHEMA = "anime-video-studio/p1-source-approval/v1"
RUNTIME_EVIDENCE_SCHEMA = "anime-video-studio/p1-runtime-evidence/v1"
SOURCE_BUNDLE_ROLES = (
    "config",
    "workflow",
    "manifest",
    "job",
    "source",
    "derived",
    "policy",
)
CANDIDATE_CURRENCY = "USD-Blockchain"
HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
HEX40 = re.compile(r"^[0-9a-fA-F]{40}$")
PLACEHOLDERS = {
    "",
    "NOT_VERIFIED",
    "UNVERIFIED",
    "UNKNOWN",
    "TODO",
    "TBD",
    "PLACEHOLDER",
    "NONE",
    "NULL",
}
_CREDENTIAL_KEYS = {
    "token",
    "password",
    "secret",
    "authorization",
    "apikey",
    "api_key",
    "credential",
    "credentials",
}


def _reject_unknown_keys(data: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise P1ConfigError(f"{where} has unknown keys: {sorted(unknown)}")


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise P1ConfigError(f"{name} must be a number")
    if not math.isfinite(float(value)) or value <= 0:
        raise P1ConfigError(f"{name} must be finite and > 0")
    return float(value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise P1ConfigError(f"{name} must be an integer")
    if value <= 0:
        raise P1ConfigError(f"{name} must be > 0")
    return value


def _string(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise P1ConfigError(f"{name} must be a string")
    if not allow_empty and not value.strip():
        raise P1ConfigError(f"{name} must not be empty")
    return value


def _safe_path(value: Any, name: str, base_dir: Path) -> Path:
    text = _string(value, name)
    if "\x00" in text:
        raise P1ConfigError(f"{name} contains a NUL character")
    path = Path(text)
    if not path.is_absolute():
        path = base_dir / path
    try:
        return path.resolve()
    except OSError as exc:
        raise P1ConfigError(f"{name} cannot be resolved: {exc}") from exc


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _validate_url(value: str, name: str) -> str:
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"}:
        raise P1ConfigError(f"{name} must use http or https")
    if not parts.hostname:
        raise P1ConfigError(f"{name} must include a host")
    if parts.username or parts.password:
        raise P1ConfigError(f"{name} must not contain userinfo")
    if parts.query or parts.fragment:
        raise P1ConfigError(f"{name} must not contain query/fragment credentials")
    return value.rstrip("/")


@dataclass(frozen=True)
class P1RuntimeConfig:
    base_url: str
    token_env: str
    exclusive_worker: bool
    client_id: str
    connect_timeout_seconds: float
    read_timeout_seconds: float
    upload_timeout_seconds: float
    artifact_timeout_seconds: float
    max_upload_bytes: int
    max_artifact_bytes: int
    width: int
    height: int
    frames: int
    fps: int
    input_image_path: Path | None
    input_image_suffixes: tuple[str, ...]
    workflow_path: Path
    workflow_sha256: str | None
    manifest_path: Path
    binding: WorkflowBinding
    binding_raw: Mapping[str, Any]
    base_dir: Path

    @property
    def loopback(self) -> bool:
        return _is_loopback(urlsplit(self.base_url).hostname or "")

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], *, base_dir: Path = ROOT
    ) -> "P1RuntimeConfig":
        if not isinstance(data, Mapping):
            raise P1ConfigError("P1 config must be an object")
        _reject_unknown_keys(data, {"schema", "runtime", "binding"}, "config")
        if data.get("schema") != CONFIG_SCHEMA:
            raise P1ConfigError(f"config.schema must be {CONFIG_SCHEMA!r}")
        raw = data.get("runtime")
        if not isinstance(raw, Mapping):
            raise P1ConfigError("config.runtime must be an object")
        _reject_unknown_keys(
            raw,
            {
                "base_url",
                "token_env",
                "exclusive_worker",
                "client_id",
                "connect_timeout_seconds",
                "read_timeout_seconds",
                "upload_timeout_seconds",
                "artifact_timeout_seconds",
                "max_upload_bytes",
                "max_artifact_bytes",
                "width",
                "height",
                "frames",
                "fps",
                "input_image_path",
                "input_image_suffixes",
                "workflow_path",
                "workflow_sha256",
                "manifest_path",
            },
            "config.runtime",
        )

        base_url = _validate_url(_string(raw.get("base_url"), "runtime.base_url"), "runtime.base_url")
        token_env = _string(raw.get("token_env", ""), "runtime.token_env", allow_empty=True)
        if token_env and not re.match(r"^[A-Z][A-Z0-9_]*$", token_env):
            raise P1ConfigError("runtime.token_env must look like an environment name")
        exclusive_worker = raw.get("exclusive_worker")
        if not isinstance(exclusive_worker, bool):
            raise P1ConfigError("runtime.exclusive_worker must be a boolean")
        client_id = _string(raw.get("client_id", "anime-video-studio-p1"), "runtime.client_id")

        host = urlsplit(base_url).hostname or ""
        if not _is_loopback(host):
            if urlsplit(base_url).scheme != "https":
                raise P1ConfigError("non-loopback ComfyUI base_url must use https")
            if not token_env:
                raise P1ConfigError(
                    "non-loopback ComfyUI requires runtime.token_env for an "
                    "authenticated closed channel"
                )

        input_suffixes_raw = raw.get("input_image_suffixes", [".png", ".jpg", ".jpeg", ".webp"])
        if (
            not isinstance(input_suffixes_raw, list)
            or not input_suffixes_raw
            or any(not isinstance(item, str) or not item.startswith(".") for item in input_suffixes_raw)
        ):
            raise P1ConfigError("runtime.input_image_suffixes must be a non-empty list")
        input_suffixes = tuple(item.lower() for item in input_suffixes_raw)

        input_image_raw = raw.get("input_image_path")
        input_image_path = None
        if input_image_raw not in (None, ""):
            input_image_path = _safe_path(input_image_raw, "runtime.input_image_path", base_dir)

        workflow_path = _safe_path(raw.get("workflow_path"), "runtime.workflow_path", base_dir)
        manifest_path = _safe_path(raw.get("manifest_path"), "runtime.manifest_path", base_dir)
        workflow_hash_raw = raw.get("workflow_sha256")
        workflow_hash = None
        if workflow_hash_raw not in (None, "", "NOT_VERIFIED"):
            workflow_hash = _string(workflow_hash_raw, "runtime.workflow_sha256")
            if not HEX64.match(workflow_hash):
                raise P1ConfigError("runtime.workflow_sha256 must be 64 hex characters")

        binding_raw = data.get("binding")
        if not isinstance(binding_raw, Mapping):
            raise P1ConfigError("config.binding must be an object")
        try:
            binding = WorkflowBinding.from_dict(binding_raw)
        except WorkflowBindingError as exc:
            raise P1ConfigError(f"config.binding is invalid: {exc}") from exc

        return cls(
            base_url=base_url,
            token_env=token_env,
            exclusive_worker=exclusive_worker,
            client_id=client_id,
            connect_timeout_seconds=_positive_number(
                raw.get("connect_timeout_seconds", 5.0),
                "runtime.connect_timeout_seconds",
            ),
            read_timeout_seconds=_positive_number(
                raw.get("read_timeout_seconds", 30.0),
                "runtime.read_timeout_seconds",
            ),
            upload_timeout_seconds=_positive_number(
                raw.get("upload_timeout_seconds", 60.0),
                "runtime.upload_timeout_seconds",
            ),
            artifact_timeout_seconds=_positive_number(
                raw.get("artifact_timeout_seconds", 120.0),
                "runtime.artifact_timeout_seconds",
            ),
            max_upload_bytes=_positive_int(
                raw.get("max_upload_bytes", 20 * 1024 * 1024),
                "runtime.max_upload_bytes",
            ),
            max_artifact_bytes=_positive_int(
                raw.get("max_artifact_bytes", 100 * 1024 * 1024),
                "runtime.max_artifact_bytes",
            ),
            width=_positive_int(raw.get("width", 704), "runtime.width"),
            height=_positive_int(raw.get("height", 1280), "runtime.height"),
            frames=_positive_int(raw.get("frames", 81), "runtime.frames"),
            fps=_positive_int(raw.get("fps", 24), "runtime.fps"),
            input_image_path=input_image_path,
            input_image_suffixes=input_suffixes,
            workflow_path=workflow_path,
            workflow_sha256=workflow_hash,
            manifest_path=manifest_path,
            binding=binding,
            binding_raw=binding_raw,
            base_dir=base_dir,
        )


def _placeholder(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        stripped = value.strip()
        return stripped.upper() in PLACEHOLDERS or stripped.startswith("<")
    return False


def _check_url(value: Any, name: str, issues: list[str]) -> None:
    if not isinstance(value, str) or _placeholder(value):
        issues.append(f"{name} is missing or NOT_VERIFIED")
        return
    try:
        parts = urlsplit(value)
    except ValueError:
        issues.append(f"{name} is not a valid URL")
        return
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        issues.append(f"{name} must be an http(s) URL")
    if parts.username or parts.password:
        issues.append(f"{name} must not contain userinfo")
    if parts.query or parts.fragment:
        issues.append(f"{name} must not contain query/fragment credentials")


def _check_no_secret_keys(value: Any, where: str, issues: list[str]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_lower = str(key).lower().replace("-", "_")
            if any(marker in key_lower for marker in _CREDENTIAL_KEYS):
                issues.append(f"{where} contains a credential-like key {key!r}")
            _check_no_secret_keys(child, f"{where}.{key}", issues)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_no_secret_keys(child, f"{where}[{index}]", issues)


def _check_date(value: Any, name: str, issues: list[str]) -> None:
    if not isinstance(value, str) or _placeholder(value):
        issues.append(f"{name} is missing or NOT_VERIFIED")
        return
    try:
        date.fromisoformat(value)
    except ValueError:
        issues.append(f"{name} must be an ISO date")


def _manifest_common_issues(
    manifest: Mapping[str, Any],
    *,
    workflow_path: Path,
    actual_workflow_sha256: str,
    config: P1RuntimeConfig,
) -> list[str]:
    """Structural checks shared by source and generation validation.

    This deliberately does not decide whether DRAFT or VERIFIED is acceptable.
    Callers choose the phase-specific status rule explicitly.
    """

    issues: list[str] = []
    if not isinstance(manifest, Mapping):
        return ["manifest must be an object"]
    _check_no_secret_keys(manifest, "manifest", issues)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        issues.append(f"manifest.schema must be {MANIFEST_SCHEMA!r}")
    for key in ("manifest_id", "status"):
        if _placeholder(manifest.get(key)):
            issues.append(f"manifest.{key} is missing or NOT_VERIFIED")
    if not isinstance(manifest.get("status"), str) or manifest.get("status") not in {
        "DRAFT",
        "VERIFIED",
        "BLOCKED",
    }:
        issues.append("manifest.status must be DRAFT, VERIFIED or BLOCKED")

    container = manifest.get("container")
    if not isinstance(container, Mapping):
        issues.append("manifest.container is missing")
    else:
        if _placeholder(container.get("image")):
            issues.append("manifest.container.image is missing or NOT_VERIFIED")
        digest = container.get("digest")
        if not isinstance(digest, str) or not re.match(r"^sha256:[0-9a-fA-F]{64}$", digest):
            issues.append("manifest.container.digest must be sha256:<64 hex>")

    comfyui = manifest.get("comfyui")
    if not isinstance(comfyui, Mapping) or not isinstance(comfyui.get("commit"), str):
        issues.append("manifest.comfyui.commit is missing")
    elif not HEX40.match(comfyui.get("commit", "")):
        issues.append("manifest.comfyui.commit must be 40 hex characters")

    custom_nodes = manifest.get("custom_nodes")
    if not isinstance(custom_nodes, list):
        issues.append(
            "manifest.custom_nodes must be an explicitly pinned list "
            "(use [] for a core-only worker)"
        )
    else:
        # An explicitly empty list is the correct core-only case. Missing,
        # null and non-list values are still rejected above; a non-empty list
        # keeps the existing pin/source/credential checks unchanged.
        for index, node in enumerate(custom_nodes):
            if not isinstance(node, Mapping):
                issues.append(f"manifest.custom_nodes[{index}] must be an object")
                continue
            if _placeholder(node.get("name")):
                issues.append(f"manifest.custom_nodes[{index}].name is missing")
            commit = node.get("commit")
            if not isinstance(commit, str) or not HEX40.match(commit):
                issues.append(f"manifest.custom_nodes[{index}].commit must be 40 hex")
            _check_url(node.get("source_url"), f"manifest.custom_nodes[{index}].source_url", issues)

    workflow = manifest.get("workflow")
    if not isinstance(workflow, Mapping):
        issues.append("manifest.workflow is missing")
    else:
        declared_hash = workflow.get("sha256")
        if not isinstance(declared_hash, str) or not HEX64.match(declared_hash):
            issues.append("manifest.workflow.sha256 must be 64 hex characters")
        elif declared_hash.lower() != actual_workflow_sha256.lower():
            issues.append(
                "manifest.workflow.sha256 does not match the configured workflow file"
            )
        if config.workflow_sha256 and config.workflow_sha256.lower() != actual_workflow_sha256.lower():
            issues.append("config.runtime.workflow_sha256 does not match the workflow file")

    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        issues.append("manifest.models must be a non-empty list")
    else:
        for index, model in enumerate(models):
            if not isinstance(model, Mapping):
                issues.append(f"manifest.models[{index}] must be an object")
                continue
            prefix = f"manifest.models[{index}]"
            for key in ("id", "role", "revision", "filename"):
                if _placeholder(model.get(key)):
                    issues.append(f"{prefix}.{key} is missing or NOT_VERIFIED")
            _check_url(model.get("source_url"), f"{prefix}.source_url", issues)
            _check_url(model.get("license_url"), f"{prefix}.license_url", issues)
            checksum = model.get("sha256")
            if not isinstance(checksum, str) or not HEX64.match(checksum):
                issues.append(f"{prefix}.sha256 must be 64 hex characters")
            _check_date(model.get("license_reviewed_at"), f"{prefix}.license_reviewed_at", issues)
            if model.get("hosted_use") != "allowed":
                issues.append(f"{prefix}.hosted_use must be explicitly 'allowed'")
            if model.get("commercial_use_status") != "allowed":
                issues.append(
                    f"{prefix}.commercial_use_status must be explicitly 'allowed'"
                )

    if _placeholder(manifest.get("checked_at")):
        issues.append("manifest.checked_at is missing or NOT_VERIFIED")
    else:
        _check_date(manifest.get("checked_at"), "manifest.checked_at", issues)

    try:
        workflow_data, _ = load_workflow(workflow_path)
    except WorkflowBindingError as exc:
        issues.append(str(exc))
    else:
        if TEMPLATE_MARKER in json.dumps(workflow_data, ensure_ascii=False):
            issues.append("configured workflow is TEMPLATE_NOT_EXECUTABLE")
        binding_issues = config.binding.validate(workflow_data)
        issues.extend(f"workflow binding: {issue}" for issue in binding_issues)

    if config.input_image_path is None:
        issues.append("config.runtime.input_image_path is missing")
    else:
        if config.input_image_path.suffix.lower() not in config.input_image_suffixes:
            issues.append("configured input image has a disallowed suffix")
        if not config.input_image_path.is_file():
            issues.append("configured input image does not exist")
        else:
            try:
                if config.input_image_path.stat().st_size > config.max_upload_bytes:
                    issues.append("configured input image exceeds max_upload_bytes")
                if config.input_image_path.stat().st_size == 0:
                    issues.append("configured input image is empty")
            except OSError as exc:
                issues.append(f"configured input image cannot be inspected: {exc}")

    return issues


def _parse_iso_moment(value: Any, name: str, issues: list[str]) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        issues.append(f"{name} is missing")
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        issues.append(f"{name} must be an ISO-8601 timestamp")
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _check_hex(value: Any, name: str, issues: list[str], length: int = 64) -> None:
    if not isinstance(value, str) or len(value) != length or not re.fullmatch(
        rf"[0-9a-fA-F]{{{length}}}", value
    ):
        issues.append(f"{name} must be {length} hex characters")


def _mapping_digest(value: Mapping[str, Any]) -> str:
    text = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _descriptor_input_names(descriptor: Mapping[str, Any]) -> set[str]:
    inputs = descriptor.get("input")
    if not isinstance(inputs, Mapping):
        return set()
    names: set[str] = set()
    for section in ("required", "optional"):
        values = inputs.get(section)
        if isinstance(values, Mapping):
            names.update(str(key) for key in values)
    if not names:
        names.update(str(key) for key in inputs)
    return names


def _dynamic_options(spec: Any) -> list[Any] | None:
    if not isinstance(spec, (list, tuple)) or len(spec) != 2:
        return None
    if spec[0] != "COMFY_DYNAMICCOMBO_V3" or not isinstance(spec[1], Mapping):
        return None
    options = spec[1].get("options")
    return options if isinstance(options, list) else None


def _dynamic_option(spec: Any, key: str) -> Mapping[str, Any] | None:
    options = _dynamic_options(spec)
    if options is None:
        return None
    matches = [
        option
        for option in options
        if isinstance(option, Mapping) and option.get("key") == key
    ]
    return matches[0] if len(matches) == 1 else None


def _dynamic_child(option: Mapping[str, Any], field: str) -> Any:
    inputs = option.get("inputs")
    if not isinstance(inputs, Mapping):
        return None
    found: list[Any] = []
    for section in ("required", "optional"):
        values = inputs.get(section)
        if isinstance(values, Mapping) and field in values:
            found.append(values[field])
    return found[0] if len(found) == 1 else None


def _descriptor_dynamic_combo_valid(descriptor: Mapping[str, Any]) -> bool:
    inputs = descriptor.get("input")
    if not isinstance(inputs, Mapping):
        return False
    format_spec = None
    for section in ("required", "optional"):
        values = inputs.get(section)
        if isinstance(values, Mapping) and "format" in values:
            format_spec = values["format"]
            break
    if format_spec is None:
        return False
    mp4 = _dynamic_option(format_spec, "mp4")
    if mp4 is None:
        return False
    codec_spec = _dynamic_child(mp4, "codec")
    if codec_spec is None:
        return False
    h264 = _dynamic_option(codec_spec, "h264")
    if h264 is None:
        return False
    encoding_spec = _dynamic_child(h264, "encoding")
    if encoding_spec is None:
        return False
    return _dynamic_option(encoding_spec, "auto") is not None


def _expected_graph_requirements(
    config: P1RuntimeConfig,
) -> tuple[dict[str, str], dict[str, list[str]], list[dict[str, str]]]:
    raw_nodes = config.binding_raw.get("node_types")
    if not isinstance(raw_nodes, Mapping):
        raise P1ConfigError("config.binding.node_types is missing")
    node_classes = {str(key): str(value) for key, value in raw_nodes.items()}
    workflow_data, _ = load_workflow(config.workflow_path)
    inputs_by_class: dict[str, set[str]] = {}
    checks: list[dict[str, str]] = []
    for node_id, class_name in node_classes.items():
        node = workflow_data.get(node_id)
        if not isinstance(node, Mapping) or node.get("class_type") != class_name:
            raise P1ConfigError(f"workflow node {node_id} class does not match binding")
        inputs = node.get("inputs")
        if not isinstance(inputs, Mapping):
            raise P1ConfigError(f"workflow node {node_id} inputs are missing")
        for input_name in inputs:
            text = str(input_name)
            if class_name == "SaveVideo" and (
                text == "format" or text.startswith("format.")
            ):
                continue
            inputs_by_class.setdefault(class_name, set()).add(text)
            checks.append(
                {"node_id": node_id, "class": class_name, "input": text}
            )
    expected_inputs = {
        class_name: sorted(names)
        for class_name, names in sorted(inputs_by_class.items())
    }
    checks.sort(key=lambda item: (item["node_id"], item["class"], item["input"]))
    return node_classes, expected_inputs, checks


def _expected_torch_profile(manifest: Mapping[str, Any]) -> dict[str, str] | None:
    container = manifest.get("container")
    image = container.get("image") if isinstance(container, Mapping) else None
    if not isinstance(image, str):
        return None
    torch_match = re.search(r"torch(\d)(\d)(\d?)", image)
    cuda_match = re.search(r"cu(\d{3})", image)
    if torch_match is None or cuda_match is None:
        return None
    major, minor, patch = torch_match.groups()
    version = f"{major}.{minor}" + (f".{patch}" if patch else "")
    cuda_digits = cuda_match.group(1)
    return {
        "torch": version,
        "torch_major_minor": f"{major}.{minor}",
        "cuda": f"{cuda_digits[:2]}.{cuda_digits[2]}",
    }


def _candidate_issues(
    manifest: Mapping[str, Any],
    *,
    config: P1RuntimeConfig,
    manifest_path: Path | None,
) -> list[str]:
    """Validate explicit live-candidate provenance and runtime facts.

    This runs only when a manifest carries the explicit ``candidate`` block.
    Established VERIFIED manifests without that block keep the historical
    structural contract; a candidate is never inferred from status alone.
    """

    issues: list[str] = []
    candidate = manifest.get("candidate")
    if not isinstance(candidate, Mapping):
        return ["manifest.candidate must be an object"]
    if candidate.get("schema") != CANDIDATE_SCHEMA:
        issues.append(f"candidate.schema must be {CANDIDATE_SCHEMA!r}")
    if candidate.get("kind") != "live_candidate":
        issues.append("candidate.kind must be 'live_candidate'")
    if candidate.get("source_status") != "DRAFT":
        issues.append("candidate.source_status must be DRAFT")
    if candidate.get("source_promoted") is not False:
        issues.append("candidate.source_promoted must be false")
    for key in ("run_id", "nonce"):
        if not isinstance(candidate.get(key), str) or not candidate.get(key):
            issues.append(f"candidate.{key} is missing")

    verification = manifest.get("verification")
    if not isinstance(verification, Mapping):
        issues.append("manifest.verification is missing")
    else:
        if verification.get("GPU_QUALITY_VERIFIED") is not False:
            issues.append("manifest.verification.GPU_QUALITY_VERIFIED must be false")
        if verification.get("GENERATION_STATE") != "NOT_RUN":
            issues.append("manifest.verification.GENERATION_STATE must be NOT_RUN")
        if verification.get("RUNTIME_READINESS") != "VERIFIED":
            issues.append("manifest.verification.RUNTIME_READINESS must be VERIFIED")
        if verification.get("SOURCE_STATUS") != "DRAFT":
            issues.append("manifest.verification.SOURCE_STATUS must be DRAFT")

    source_manifest = candidate.get("source_manifest")
    if not isinstance(source_manifest, Mapping):
        issues.append("candidate.source_manifest is missing")
        source_manifest = {}
    if source_manifest.get("status") != "DRAFT":
        issues.append("candidate.source_manifest.status must be DRAFT")
    _check_hex(source_manifest.get("sha256"), "candidate.source_manifest.sha256", issues)
    source_path_raw = source_manifest.get("path")
    if not isinstance(source_path_raw, str) or not source_path_raw.strip():
        issues.append("candidate.source_manifest.path is missing")
    else:
        try:
            source_path = _safe_path(source_path_raw, "candidate.source_manifest.path", config.base_dir)
        except P1ConfigError as exc:
            issues.append(str(exc))
        else:
            if not source_path.is_file():
                issues.append("candidate.source_manifest.path does not exist")
            else:
                try:
                    actual_source = sha256_file(source_path)
                except P1ConfigError as exc:
                    issues.append(str(exc))
                else:
                    declared_source = source_manifest.get("sha256")
                    if isinstance(declared_source, str) and actual_source.lower() != declared_source.lower():
                        issues.append("candidate.source_manifest.sha256 does not match the file")

    source_bundle = candidate.get("source_bundle")
    if not isinstance(source_bundle, Mapping):
        issues.append("candidate.source_bundle is missing")
        source_bundle = {}
    else:
        if set(source_bundle) != set(SOURCE_BUNDLE_ROLES):
            issues.append("candidate.source_bundle has missing/unknown roles")
        for role in SOURCE_BUNDLE_ROLES:
            _check_hex(source_bundle.get(role), f"candidate.source_bundle.{role}", issues)
    bundle_hash = candidate.get("source_bundle_sha256")
    _check_hex(bundle_hash, "candidate.source_bundle_sha256", issues)
    if isinstance(bundle_hash, str) and source_bundle and _mapping_digest(source_bundle) != bundle_hash.lower():
        issues.append("candidate.source_bundle_sha256 does not match source_bundle")
    if isinstance(source_bundle.get("manifest"), str) and isinstance(
        source_manifest.get("sha256"), str
    ):
        if source_bundle["manifest"].lower() != source_manifest["sha256"].lower():
            issues.append("candidate.source_bundle.manifest differs from source_manifest.sha256")
    workflow = manifest.get("workflow")
    if isinstance(workflow, Mapping) and isinstance(source_bundle.get("workflow"), str):
        if str(workflow.get("sha256", "")).lower() != source_bundle["workflow"].lower():
            issues.append("candidate.workflow.sha256 differs from source_bundle.workflow")

    approval = candidate.get("source_approval")
    if not isinstance(approval, Mapping):
        issues.append("candidate.source_approval is missing")
    else:
        _check_hex(approval.get("artifact_sha256"), "candidate.source_approval.artifact_sha256", issues)
        if approval.get("schema") != SOURCE_APPROVAL_SCHEMA:
            issues.append(f"candidate.source_approval.schema must be {SOURCE_APPROVAL_SCHEMA!r}")
        if approval.get("source_bundle_sha256") != bundle_hash:
            issues.append("candidate.source_approval.source_bundle_sha256 differs")
        doc = approval.get("approval_doc")
        if not isinstance(doc, Mapping):
            issues.append("candidate.source_approval.approval_doc is missing")
        else:
            _check_hex(doc.get("sha256"), "candidate.source_approval.approval_doc.sha256", issues)
            doc_path = doc.get("path")
            if not isinstance(doc_path, str) or not doc_path.strip():
                issues.append("candidate.source_approval.approval_doc.path is missing")
            else:
                try:
                    resolved_doc = _safe_path(
                        doc_path, "candidate.source_approval.approval_doc.path", config.base_dir
                    )
                except P1ConfigError as exc:
                    issues.append(str(exc))
                else:
                    if not resolved_doc.is_file():
                        issues.append("candidate.source_approval.approval_doc.path does not exist")
                    else:
                        try:
                            actual_doc = sha256_file(resolved_doc)
                        except P1ConfigError as exc:
                            issues.append(str(exc))
                        else:
                            if isinstance(doc.get("sha256"), str) and actual_doc.lower() != doc["sha256"].lower():
                                issues.append("candidate.source_approval.approval_doc.sha256 does not match the file")
        created = _parse_iso_moment(
            approval.get("created_at"), "candidate.source_approval.created_at", issues
        )
        expires = _parse_iso_moment(
            approval.get("expires_at"), "candidate.source_approval.expires_at", issues
        )
        now = datetime.now(timezone.utc)
        if created is not None and created > now:
            issues.append("candidate.source_approval.created_at is in the future")
        if expires is not None and expires <= now:
            issues.append("candidate.source_approval.expires_at has passed")
        if created is not None and expires is not None and expires <= created:
            issues.append("candidate.source_approval.expires_at must be after created_at")
        scope = approval.get("scope")
        if not isinstance(scope, Mapping):
            issues.append("candidate.source_approval.scope is missing")
        else:
            if scope.get("purpose") != "bootstrap_candidate_only":
                issues.append("candidate.source_approval.scope.purpose must be bootstrap_candidate_only")
            if scope.get("orders") != 1:
                issues.append("candidate.source_approval.scope.orders must be 1")
            if scope.get("max_generations") != 1:
                issues.append("candidate.source_approval.scope.max_generations must be 1")
            if scope.get("automatic_retries") != 0:
                issues.append("candidate.source_approval.scope.automatic_retries must be 0")
            if scope.get("max_duration_seconds") != 3600:
                issues.append("candidate.source_approval.scope.max_duration_seconds must be 3600")
            try:
                cap = Decimal(str(scope.get("max_total_usd")))
            except (InvalidOperation, ValueError):
                issues.append("candidate.source_approval.scope.max_total_usd is invalid")
            else:
                if cap <= 0 or cap > Decimal("1.00"):
                    issues.append("candidate.source_approval.scope.max_total_usd must be <= 1.00")

    session = candidate.get("session")
    if not isinstance(session, Mapping):
        issues.append("candidate.session is missing")
    else:
        integer_keys = (
            "order_id",
            "server_id",
            "created_at",
            "duration_seconds",
            "planned_cancel_at",
            "hard_end_at",
        )
        for key in integer_keys:
            value = session.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                issues.append(f"candidate.session.{key} is invalid")
        if session.get("currency") != CANDIDATE_CURRENCY:
            issues.append(f"candidate.session.currency must be {CANDIDATE_CURRENCY!r}")
        _check_hex(session.get("quote_hash"), "candidate.session.quote_hash", issues)
        if "owner_risk_admission_sha256" in session:
            _check_hex(
                session.get("owner_risk_admission_sha256"),
                "candidate.session.owner_risk_admission_sha256",
                issues,
            )
        try:
            cap = Decimal(str(session.get("cap_usd")))
        except (InvalidOperation, ValueError):
            issues.append("candidate.session.cap_usd is invalid")
        else:
            if cap <= 0 or cap > Decimal("1.00"):
                issues.append("candidate.session.cap_usd must be <= 1.00")
            approval_map = candidate.get("source_approval")
            approval_scope = (
                approval_map.get("scope") if isinstance(approval_map, Mapping) else None
            )
            if isinstance(approval_scope, Mapping):
                try:
                    approval_cap = Decimal(str(approval_scope.get("max_total_usd")))
                except (InvalidOperation, ValueError):
                    approval_cap = None
                if approval_cap is not None and cap > approval_cap:
                    issues.append("candidate.session.cap_usd exceeds the source approval cap")
        created_at = session.get("created_at")
        planned = session.get("planned_cancel_at")
        hard_end = session.get("hard_end_at")
        duration = session.get("duration_seconds")
        if isinstance(created_at, int) and isinstance(planned, int) and planned < created_at:
            issues.append("candidate.session.planned_cancel_at precedes creation")
        if isinstance(created_at, int) and isinstance(planned, int) and planned > created_at + 3300:
            issues.append("candidate.session.planned_cancel_at exceeds creation + 55 minutes")
        if isinstance(created_at, int) and isinstance(hard_end, int):
            if hard_end <= created_at:
                issues.append("candidate.session.hard_end_at must be after creation")
            if hard_end > created_at + 3600:
                issues.append("candidate.session.hard_end_at exceeds creation + 60 minutes")
        if isinstance(duration, int) and duration > 3600:
            issues.append("candidate.session.duration_seconds exceeds 60 minutes")
        session_hash = _mapping_digest(session)
        declared_session = candidate.get("session_binding_sha256")
        _check_hex(declared_session, "candidate.session_binding_sha256", issues)
        if isinstance(declared_session, str) and session_hash != declared_session.lower():
            issues.append("candidate.session_binding_sha256 does not match session")

    host_key = candidate.get("host_key")
    if not isinstance(host_key, Mapping):
        issues.append("candidate.host_key is missing")
    else:
        fingerprint = host_key.get("fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint.startswith("SHA256:"):
            issues.append("candidate.host_key.fingerprint is invalid")
        _check_hex(host_key.get("known_hosts_sha256"), "candidate.host_key.known_hosts_sha256", issues)
        checked = _parse_iso_moment(host_key.get("checked_at"), "candidate.host_key.checked_at", issues)
        if checked is not None and checked > datetime.now(timezone.utc):
            issues.append("candidate.host_key.checked_at is in the future")

    collector = candidate.get("collector")
    if not isinstance(collector, Mapping):
        issues.append("candidate.collector is missing")
    else:
        _check_hex(collector.get("sha256"), "candidate.collector.sha256", issues)
        _check_hex(collector.get("plan_sha256"), "candidate.collector.plan_sha256", issues)

    evidence = candidate.get("evidence")
    evidence_doc = None
    raw_doc = None
    raw_size = None
    if not isinstance(evidence, Mapping):
        issues.append("candidate.evidence is missing")
    else:
        raw_path_raw = evidence.get("raw_path")
        _check_hex(evidence.get("raw_sha256"), "candidate.evidence.raw_sha256", issues)
        _check_hex(evidence.get("sha256"), "candidate.evidence.sha256", issues)
        evidence_path_raw = evidence.get("path")
        if not isinstance(evidence_path_raw, str) or not evidence_path_raw.strip():
            issues.append("candidate.evidence.path is missing")
        elif manifest_path is None:
            issues.append("candidate.evidence.path cannot be verified without manifest path")
        else:
            evidence_rel = Path(evidence_path_raw)
            if evidence_rel.is_absolute() or ".." in evidence_rel.parts:
                issues.append("candidate.evidence.path must be a relative owned path")
            else:
                evidence_file = Path(manifest_path).parent / evidence_rel
                if not evidence_file.is_file():
                    issues.append("candidate.evidence.path does not exist")
                else:
                    try:
                        actual_evidence = sha256_file(evidence_file)
                    except P1ConfigError as exc:
                        issues.append(str(exc))
                    else:
                        declared_evidence = evidence.get("sha256")
                        if isinstance(declared_evidence, str) and actual_evidence.lower() != declared_evidence.lower():
                            issues.append("candidate.evidence.sha256 does not match the evidence file")
                        else:
                            try:
                                evidence_doc = json.loads(evidence_file.read_text(encoding="utf-8"))
                            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                                issues.append(f"candidate.evidence.path is not valid JSON: {type(exc).__name__}")
                            else:
                                if not isinstance(evidence_doc, dict):
                                    issues.append("candidate.evidence.path must contain an object")
                                    evidence_doc = None
        observed = _parse_iso_moment(
            evidence.get("observed_at"), "candidate.evidence.observed_at", issues
        )
        expires = _parse_iso_moment(
            evidence.get("expires_at"), "candidate.evidence.expires_at", issues
        )
        now = datetime.now(timezone.utc)
        if observed is not None and observed > now:
            issues.append("candidate.evidence.observed_at is in the future")
        if expires is not None and expires <= now:
            issues.append("candidate.evidence.expires_at has passed")
        if observed is not None and expires is not None and expires <= observed:
            issues.append("candidate.evidence.expires_at must be after observed_at")
        if not isinstance(raw_path_raw, str) or not raw_path_raw.strip():
            issues.append("candidate.evidence.raw_path is missing")
        elif manifest_path is None:
            issues.append("candidate.evidence.raw_path cannot be verified without manifest path")
        else:
            raw_rel = Path(raw_path_raw)
            if raw_rel.is_absolute() or ".." in raw_rel.parts:
                issues.append("candidate.evidence.raw_path must be a relative owned path")
            else:
                raw_path = Path(manifest_path).parent / raw_rel
                if not raw_path.is_file():
                    issues.append("candidate.evidence.raw_path does not exist")
                else:
                    try:
                        raw_bytes = raw_path.read_bytes()
                        actual_raw = hashlib.sha256(raw_bytes).hexdigest()
                    except (OSError, P1ConfigError) as exc:
                        issues.append(f"candidate.evidence.raw_path cannot be read: {type(exc).__name__}")
                    else:
                        declared_raw = evidence.get("raw_sha256")
                        if isinstance(declared_raw, str) and actual_raw.lower() != declared_raw.lower():
                            issues.append("candidate.evidence.raw_sha256 does not match the raw file")
                        else:
                            raw_size = len(raw_bytes)
                            try:
                                raw_doc = json.loads(raw_bytes.decode("utf-8"))
                            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                                issues.append(f"candidate.evidence.raw_path is not valid JSON: {type(exc).__name__}")
                            else:
                                if not isinstance(raw_doc, dict):
                                    issues.append("candidate.evidence.raw_path must contain an object")
                                    raw_doc = None
                                elif raw_doc.get("schema") != RUNTIME_EVIDENCE_SCHEMA:
                                    issues.append("candidate.evidence.raw_path schema is invalid")

    runtime = candidate.get("runtime_evidence")
    if not isinstance(runtime, Mapping):
        issues.append("candidate.runtime_evidence is missing")
        return issues
    if runtime.get("schema") != RUNTIME_EVIDENCE_SCHEMA:
        issues.append(f"candidate.runtime_evidence.schema must be {RUNTIME_EVIDENCE_SCHEMA!r}")
    for key in ("run_id", "nonce"):
        if runtime.get(key) != candidate.get(key):
            issues.append(f"candidate.runtime_evidence.{key} does not match candidate")
    for key in ("session_binding_sha256", "source_bundle_sha256", "source_approval_sha256"):
        expected = candidate.get(key)
        if key == "source_approval_sha256":
            approval_map = candidate.get("source_approval")
            expected = approval_map.get("artifact_sha256") if isinstance(approval_map, Mapping) else None
        if runtime.get(key) != expected:
            issues.append(f"candidate.runtime_evidence.{key} does not match candidate")
    if runtime.get("host_key") != candidate.get("host_key"):
        issues.append("candidate.runtime_evidence.host_key does not match candidate.host_key")
    if runtime.get("collector") != candidate.get("collector"):
        issues.append("candidate.runtime_evidence.collector does not match candidate.collector")
    if isinstance(evidence, Mapping) and runtime.get("observed_at") != evidence.get("observed_at"):
        issues.append("candidate.runtime_evidence.observed_at does not match candidate.evidence")
    if evidence_doc is not None and dict(runtime) != evidence_doc:
        issues.append("candidate.runtime_evidence differs from runtime-evidence.json")
    if raw_doc is not None:
        approval_map = candidate.get("source_approval")
        expected_approval = (
            approval_map.get("artifact_sha256") if isinstance(approval_map, Mapping) else None
        )
        if raw_doc.get("source_approval_sha256") != expected_approval:
            issues.append(
                "candidate.raw runtime_evidence source_approval identity mismatch"
            )
        if raw_doc.get("host_key") != candidate.get("host_key"):
            issues.append("candidate.raw runtime_evidence host_key identity mismatch")
        expected_runtime = dict(raw_doc)
        expected_runtime["raw_sha256"] = evidence.get("raw_sha256")
        expected_runtime["raw_size_bytes"] = raw_size
        if dict(runtime) != expected_runtime:
            issues.append("candidate.runtime_evidence differs from the raw probe JSON")

    python_facts = runtime.get("python")
    torch_facts = runtime.get("torch")
    comfy_facts = runtime.get("comfyui")
    if not isinstance(python_facts, Mapping) or not isinstance(python_facts.get("version"), str):
        issues.append("candidate.runtime_evidence.python.version is missing")
    if not isinstance(comfy_facts, Mapping):
        issues.append("candidate.runtime_evidence.comfyui is missing")
    else:
        _check_hex(comfy_facts.get("commit"), "candidate.runtime_evidence.comfyui.commit", issues, 40)
        declared_manifest_commit = manifest.get("comfyui", {}).get("commit")
        if isinstance(declared_manifest_commit, str) and comfy_facts.get("commit") != declared_manifest_commit:
            issues.append("candidate.runtime_evidence.comfyui.commit differs from manifest.comfyui.commit")
        if comfy_facts.get("import_ok") is not True or comfy_facts.get("import_error") is not None:
            issues.append("candidate.runtime_evidence.comfyui import did not succeed")

    pip_facts = runtime.get("pip")
    if not isinstance(pip_facts, Mapping):
        issues.append("candidate.runtime_evidence.pip is missing")
    else:
        if pip_facts.get("check_exit") != 0:
            issues.append("candidate.runtime_evidence.pip.check_exit must be 0")
        _check_hex(pip_facts.get("freeze_sha256"), "candidate.runtime_evidence.pip.freeze_sha256", issues)
        lines = pip_facts.get("freeze_line_count")
        if isinstance(lines, bool) or not isinstance(lines, int) or lines < 1:
            issues.append("candidate.runtime_evidence.pip.freeze_line_count is invalid")

    evidence_models = runtime.get("models")
    manifest_models = manifest.get("models")
    if not isinstance(evidence_models, list) or not isinstance(manifest_models, list):
        issues.append("candidate.runtime_evidence.models is missing")
    elif len(evidence_models) != len(manifest_models) or not evidence_models:
        issues.append("candidate.runtime_evidence.models must match every manifest model")
    else:
        for index, (actual, expected) in enumerate(zip(evidence_models, manifest_models)):
            prefix = f"candidate.runtime_evidence.models[{index}]"
            if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
                issues.append(f"{prefix} must be an object")
                continue
            for key in ("id", "target", "sha256"):
                if actual.get(key) != expected.get(key):
                    issues.append(f"{prefix}.{key} does not match manifest")
            _check_hex(actual.get("sha256"), f"{prefix}.sha256", issues)
            size = actual.get("size_bytes")
            if isinstance(size, bool) or not isinstance(size, int) or size != expected.get("size_bytes"):
                issues.append(f"{prefix}.size_bytes does not match manifest")
            if actual.get("actual_sha256") != expected.get("sha256"):
                issues.append(f"{prefix}.actual_sha256 does not match manifest sha256")
            if actual.get("actual_size_bytes") != expected.get("size_bytes"):
                issues.append(f"{prefix}.actual_size_bytes does not match manifest size_bytes")

    try:
        expected_node_classes, expected_inputs, expected_checks = (
            _expected_graph_requirements(config)
        )
    except (P1ConfigError, WorkflowBindingError) as exc:
        issues.append(f"candidate graph requirements are unavailable: {exc}")
        expected_node_classes = {}
        expected_inputs = {}
        expected_checks = []

    object_info = runtime.get("object_info")
    if not isinstance(object_info, Mapping):
        issues.append("candidate.runtime_evidence.object_info is missing")
    else:
        _check_hex(object_info.get("sha256"), "candidate.runtime_evidence.object_info.sha256", issues)
        size = object_info.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            issues.append("candidate.runtime_evidence.object_info.size_bytes is invalid")
        classes = object_info.get("classes")
        if not isinstance(classes, list) or set(classes) != set(expected_node_classes.values()):
            issues.append("candidate.runtime_evidence.object_info.classes does not match the workflow graph")
        descriptors = object_info.get("descriptors")
        if not isinstance(descriptors, Mapping):
            issues.append("candidate.runtime_evidence.object_info.descriptors is missing")
        else:
            if set(descriptors) != set(expected_node_classes.values()):
                issues.append("candidate.runtime_evidence.object_info.descriptors does not match the workflow graph")
            for class_name, required_inputs in expected_inputs.items():
                descriptor = descriptors.get(class_name)
                if not isinstance(descriptor, Mapping):
                    issues.append(f"candidate object_info descriptor for {class_name} is missing")
                    continue
                names = _descriptor_input_names(descriptor)
                for input_name in required_inputs:
                    if input_name not in names:
                        issues.append(
                            f"candidate object_info {class_name} is missing input {input_name!r}"
                        )
            save_descriptor = descriptors.get("SaveVideo")
            if not isinstance(save_descriptor, Mapping) or not _descriptor_dynamic_combo_valid(save_descriptor):
                issues.append("candidate object_info SaveVideo dynamic mp4/h264/auto descriptor is invalid")
        if object_info.get("missing_classes") != []:
            issues.append("candidate.runtime_evidence.object_info.missing_classes must be empty")
        if object_info.get("validation_issues") != []:
            issues.append("candidate.runtime_evidence.object_info.validation_issues must be empty")
        dynamic = object_info.get("dynamic_combo")
        if not isinstance(dynamic, Mapping):
            issues.append("candidate.runtime_evidence.object_info.dynamic_combo is missing")
        else:
            selected = dynamic.get("selected")
            if not isinstance(selected, Mapping):
                issues.append("candidate.runtime_evidence.object_info.dynamic_combo.selected is missing")
            else:
                for key, expected in (("format", "mp4"), ("codec", "h264"), ("encoding", "auto")):
                    if selected.get(key) != expected:
                        issues.append(
                            f"candidate.runtime_evidence.object_info.dynamic_combo.selected.{key} must be {expected!r}"
                        )
            if dynamic.get("issues") != []:
                issues.append("candidate.runtime_evidence.object_info.dynamic_combo.issues must be empty")

    graph = runtime.get("graph")
    if not isinstance(graph, Mapping):
        issues.append("candidate.runtime_evidence.graph is missing")
    else:
        node_classes = graph.get("node_classes")
        if not isinstance(node_classes, Mapping) or dict(node_classes) != expected_node_classes:
            issues.append("candidate.runtime_evidence.graph.node_classes does not match config binding")
        if graph.get("missing_graph_classes") != []:
            issues.append("candidate.runtime_evidence.graph.missing_graph_classes must be empty")
        checks = graph.get("input_checks")
        if not isinstance(checks, list):
            issues.append("candidate.runtime_evidence.graph.input_checks is missing")
        else:
            actual: dict[tuple[str, str, str], bool] = {}
            for item in checks:
                if not isinstance(item, Mapping):
                    issues.append("candidate.runtime_evidence.graph.input_checks contains a non-object")
                    continue
                key = (
                    str(item.get("node_id", "")),
                    str(item.get("class", "")),
                    str(item.get("input", "")),
                )
                if key in actual:
                    issues.append("candidate.runtime_evidence.graph.input_checks has duplicates")
                actual[key] = item.get("present") is True
            expected = {
                (item["node_id"], item["class"], item["input"]): True
                for item in expected_checks
            }
            if actual != expected:
                issues.append("candidate.runtime_evidence.graph.input_checks does not match all workflow inputs")

    torch_profile = _expected_torch_profile(manifest)
    torch_facts = runtime.get("torch")
    if not isinstance(torch_facts, Mapping):
        issues.append("candidate.runtime_evidence.torch is missing")
    else:
        actual_torch = torch_facts.get("version")
        actual_cuda = torch_facts.get("cuda_version")
        if torch_profile is None:
            issues.append("manifest.container.image does not pin a torch/CUDA profile")
        else:
            if not isinstance(actual_torch, str):
                issues.append("candidate.runtime_evidence.torch.version is missing")
            else:
                normalized_torch = actual_torch.split("+", 1)[0]
                if not (
                    normalized_torch == torch_profile["torch"]
                    or (
                        len(torch_profile["torch"].split(".")) == 2
                        and normalized_torch.startswith(torch_profile["torch_major_minor"] + ".")
                    )
                ):
                    issues.append(
                        "candidate.runtime_evidence.torch.version is incompatible with the source image pin"
                    )
            if not isinstance(actual_cuda, str):
                issues.append("candidate.runtime_evidence.torch.cuda_version is missing")
            else:
                cuda_match = re.fullmatch(r"(\d+)\.(\d+)(?:\.\d+)?", actual_cuda)
                expected_cuda = tuple(int(part) for part in torch_profile["cuda"].split("."))
                if cuda_match is None or (
                    int(cuda_match.group(1)), int(cuda_match.group(2))
                ) != expected_cuda:
                    issues.append(
                        "candidate.runtime_evidence.torch.cuda_version is incompatible with the source image pin"
                    )
        if torch_facts.get("cuda_available") is not True:
            issues.append("candidate.runtime_evidence.torch.cuda_available must be true")
        device_count = torch_facts.get("device_count")
        if isinstance(device_count, bool) or not isinstance(device_count, int) or device_count <= 0:
            issues.append("candidate.runtime_evidence.torch.device_count must be positive")
        if not isinstance(torch_facts.get("device_name"), str) or not torch_facts.get("device_name"):
            issues.append("candidate.runtime_evidence.torch.device_name is missing")
        vram = torch_facts.get("total_vram_bytes")
        if isinstance(vram, bool) or not isinstance(vram, int) or vram <= 0:
            issues.append("candidate.runtime_evidence.torch.total_vram_bytes must be positive")

    system_stats = runtime.get("system_stats")
    if not isinstance(system_stats, Mapping):
        issues.append("candidate.runtime_evidence.system_stats is missing")
    else:
        _check_hex(system_stats.get("sha256"), "candidate.runtime_evidence.system_stats.sha256", issues)
        devices = system_stats.get("devices")
        if not isinstance(devices, list) or not devices:
            issues.append("candidate.runtime_evidence.system_stats.devices is missing")
        else:
            for index, device in enumerate(devices):
                prefix = f"candidate.runtime_evidence.system_stats.devices[{index}]"
                if not isinstance(device, Mapping):
                    issues.append(f"{prefix} must be an object")
                    continue
                if not isinstance(device.get("name"), str) or not device.get("name"):
                    issues.append(f"{prefix}.name is missing")
                if str(device.get("type", "")).lower() not in {"cuda", "gpu"}:
                    issues.append(f"{prefix}.type must be CUDA")
                memory = device.get("vram_total")
                if isinstance(memory, bool) or not isinstance(memory, int) or memory <= 0:
                    issues.append(f"{prefix}.vram_total must be positive")
            torch_name = (
                torch_facts.get("device_name") if isinstance(torch_facts, Mapping) else None
            )
            system_names = [
                str(device.get("name", "")).lower()
                for device in devices
                if isinstance(device, Mapping)
            ]
            if isinstance(torch_name, str) and torch_name:
                lowered = torch_name.lower()
                if not any(lowered in name or name in lowered for name in system_names):
                    issues.append(
                        "candidate.runtime_evidence.system_stats devices do not bind to the torch CUDA device"
                    )

    python_facts = runtime.get("python")
    torch_facts = runtime.get("torch")
    comfy_facts = runtime.get("comfyui")
    if not isinstance(python_facts, Mapping) or not isinstance(python_facts.get("version"), str):
        issues.append("candidate.runtime_evidence.python.version is missing")
    if not isinstance(comfy_facts, Mapping):
        issues.append("candidate.runtime_evidence.comfyui is missing")
    else:
        _check_hex(comfy_facts.get("commit"), "candidate.runtime_evidence.comfyui.commit", issues, 40)
        declared_manifest_commit = manifest.get("comfyui", {}).get("commit")
        if isinstance(declared_manifest_commit, str) and comfy_facts.get("commit") != declared_manifest_commit:
            issues.append("candidate.runtime_evidence.comfyui.commit differs from manifest.comfyui.commit")
        if comfy_facts.get("import_ok") is not True or comfy_facts.get("import_error") is not None:
            issues.append("candidate.runtime_evidence.comfyui import did not succeed")
    return issues


def validate_source_manifest(
    manifest: Mapping[str, Any],
    *,
    workflow_path: Path,
    actual_workflow_sha256: str,
    config: P1RuntimeConfig,
) -> list[str]:
    """Validate source pins while explicitly allowing DRAFT.

    DRAFT is the only status that may be admitted here; BLOCKED is never a
    bootstrap source. A candidate block belongs to a run artifact, not source.
    """

    issues = _manifest_common_issues(
        manifest,
        workflow_path=workflow_path,
        actual_workflow_sha256=actual_workflow_sha256,
        config=config,
    )
    status = manifest.get("status") if isinstance(manifest, Mapping) else None
    if isinstance(status, str) and status in {"DRAFT", "VERIFIED"}:
        pass
    elif isinstance(status, str) and status == "BLOCKED":
        issues.append("manifest.status is BLOCKED for source bootstrap")
    elif isinstance(status, str):
        issues.append("manifest.status must be DRAFT or VERIFIED for source bootstrap")
    if isinstance(manifest, Mapping) and "candidate" in manifest:
        issues.append("source manifest must not contain a candidate block")
    return issues


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    workflow_path: Path,
    actual_workflow_sha256: str,
    config: P1RuntimeConfig,
    manifest_path: Path | None = None,
) -> list[str]:
    """Independently validate a generation manifest; never trust status alone."""

    issues = _manifest_common_issues(
        manifest,
        workflow_path=workflow_path,
        actual_workflow_sha256=actual_workflow_sha256,
        config=config,
    )
    status = manifest.get("status") if isinstance(manifest, Mapping) else None
    if isinstance(status, str) and status == "VERIFIED":
        pass
    elif isinstance(status, str) and status in {"DRAFT", "BLOCKED"}:
        issues.append("manifest.status is not VERIFIED")
    else:
        issues.append("manifest.status must be DRAFT, VERIFIED or BLOCKED")
    if isinstance(manifest, Mapping) and "candidate" in manifest:
        issues.extend(
            _candidate_issues(
                manifest,
                config=config,
                manifest_path=manifest_path,
            )
        )
    return issues


def validate_candidate_manifest(
    manifest: Mapping[str, Any],
    *,
    workflow_path: Path,
    actual_workflow_sha256: str,
    config: P1RuntimeConfig,
    manifest_path: Path | None = None,
) -> list[str]:
    """Shared explicit-candidate validation for app/operator reuse.

    A manifest without ``candidate`` is never treated as a live candidate.
    """

    issues = _manifest_common_issues(
        manifest,
        workflow_path=workflow_path,
        actual_workflow_sha256=actual_workflow_sha256,
        config=config,
    )
    status = manifest.get("status") if isinstance(manifest, Mapping) else None
    if status != "VERIFIED":
        issues.append("manifest.status is not VERIFIED")
    if not isinstance(manifest, Mapping) or "candidate" not in manifest:
        issues.append("manifest.candidate is required for a live candidate")
    else:
        issues.extend(
            _candidate_issues(
                manifest,
                config=config,
                manifest_path=manifest_path,
            )
        )
    return issues


@dataclass(frozen=True)
class P1Readiness:
    live_ready: bool
    issues: tuple[str, ...]
    config: P1RuntimeConfig
    workflow: Mapping[str, Any]
    workflow_sha256: str
    manifest: Mapping[str, Any]
    environment: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "anime-video-studio/p1-readiness/v1",
            "live_ready": self.live_ready,
            "issues": list(self.issues),
            "config": {
                "base_url": self.config.base_url,
                "loopback": self.config.loopback,
                "exclusive_worker": self.config.exclusive_worker,
                "token_env": self.config.token_env,
                "workflow_path": str(self.config.workflow_path),
                "manifest_path": str(self.config.manifest_path),
                "max_upload_bytes": self.config.max_upload_bytes,
                "max_artifact_bytes": self.config.max_artifact_bytes,
                "upload_timeout_seconds": self.config.upload_timeout_seconds,
                "artifact_timeout_seconds": self.config.artifact_timeout_seconds,
                "width": self.config.width,
                "height": self.config.height,
                "frames": self.config.frames,
                "fps": self.config.fps,
            },
            "workflow": {
                "path": str(self.config.workflow_path),
                "sha256": self.workflow_sha256,
                "template_not_executable": TEMPLATE_MARKER
                in json.dumps(self.workflow, ensure_ascii=False),
            },
            "manifest": {
                "manifest_id": self.manifest.get("manifest_id"),
                "status": self.manifest.get("status"),
            },
            "environment": dict(self.environment),
            "note": (
                "live_ready=true is only a structural gate; it is not GPU, "
                "license, price or Pod approval."
            ),
        }


def _environment_check(config: P1RuntimeConfig) -> dict[str, Any]:
    import shutil
    import sys

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    return {
        "python": sys.version.split()[0],
        "ffmpeg": ffmpeg or "NOT_FOUND",
        "ffprobe": ffprobe or "NOT_FOUND",
        "ffmpeg_note": "ffprobe/ffmpeg decode is a live check; NOT_RUN offline",
        "input_image": (
            str(config.input_image_path) if config.input_image_path else "NOT_CONFIGURED"
        ),
        "token_presence": "NOT_READ_OFFLINE",
        "docker": shutil.which("docker") or "NOT_FOUND",
        "docker_note": "Docker workloads are not started by p1-check",
    }


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    try:
        with open(Path(path), "rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise P1ConfigError(f"Cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def execution_fingerprint(
    config: P1RuntimeConfig,
    *,
    workflow_sha256: str,
    manifest_sha256: str,
    input_image_sha256: str | None,
    input_image_size: int | None,
) -> str:
    """Canonical execution identity; never includes a token value."""

    binding = json.dumps(
        dict(config.binding_raw),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    payload = {
        "schema": "anime-video-studio/p1-execution/v1",
        "endpoint": config.base_url,
        "workflow_sha256": workflow_sha256,
        "manifest_sha256": manifest_sha256,
        "binding": binding,
        "input_image": {
            "sha256": input_image_sha256 or "MISSING",
            "size": input_image_size,
            "suffix": (
                config.input_image_path.suffix.lower()
                if config.input_image_path is not None
                else None
            ),
        },
        "semantic": {
            "width": config.width,
            "height": config.height,
            "frames": config.frames,
            "fps": config.fps,
        },
        "exclusive_worker": config.exclusive_worker,
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_config(path: Path | str, *, base_dir: Path = ROOT) -> P1RuntimeConfig:
    config_path = Path(path)
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise P1ConfigError(f"Cannot load P1 config {config_path}: {exc}") from exc
    return P1RuntimeConfig.from_dict(data, base_dir=base_dir)


def load_manifest(path: Path | str) -> dict[str, Any]:
    manifest_path = Path(path)
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise P1ManifestError(
            f"Cannot load P1 manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise P1ManifestError("P1 manifest must be a JSON object")
    return data


def check_readiness(
    config_path: Path | str, *, base_dir: Path = ROOT
) -> P1Readiness:
    config = load_config(config_path, base_dir=base_dir)
    if not config.workflow_path.is_file():
        raise P1ConfigError(f"workflow file does not exist: {config.workflow_path}")
    if not config.manifest_path.is_file():
        raise P1ConfigError(f"manifest file does not exist: {config.manifest_path}")
    workflow, actual_sha = load_workflow(config.workflow_path)
    manifest = load_manifest(config.manifest_path)
    issues = validate_manifest(
        manifest,
        workflow_path=config.workflow_path,
        actual_workflow_sha256=actual_sha,
        config=config,
        manifest_path=config.manifest_path,
    )
    environment = _environment_check(config)
    return P1Readiness(
        live_ready=not issues,
        issues=tuple(issues),
        config=config,
        workflow=workflow,
        workflow_sha256=actual_sha,
        manifest=manifest,
        environment=environment,
    )


__all__ = [
    "CANDIDATE_SCHEMA",
    "CONFIG_SCHEMA",
    "MANIFEST_SCHEMA",
    "P1Readiness",
    "P1RuntimeConfig",
    "RUNTIME_EVIDENCE_SCHEMA",
    "SOURCE_APPROVAL_SCHEMA",
    "SOURCE_BUNDLE_ROLES",
    "check_readiness",
    "execution_fingerprint",
    "load_config",
    "load_manifest",
    "sha256_file",
    "validate_candidate_manifest",
    "validate_manifest",
    "validate_source_manifest",
    "workflow_sha256",
]
