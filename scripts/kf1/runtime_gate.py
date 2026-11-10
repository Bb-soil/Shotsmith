"""Actual runtime ``/object_info`` gates for KF1.

Two different checks exist and must not be conflated:

* installation/readiness capability: classes, schema names, link types, static
  enums, numeric ranges and installed model file names from the pinned profile;
* bound job: the exact bound graph, including every literal/scalar/range/link
  and the trusted upload receipt for each ``LoadImage.image`` upload.

A clean worker's ``LoadImage.image`` enum is only the files currently present.
It may be empty or may omit a just-uploaded file; the ``image_upload: True``
capability plus an exact upload receipt (name/type/subfolder/path) authorizes
the job.  A template sample filename is never treated as an installed file.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from scripts.i1 import probe as i1_probe
from scripts.p1 import preflight

from .errors import KF1Error

_UPLOAD_TYPES = "input"


def _declared_inputs(spec: Mapping[str, Any]) -> set[str]:
    return set(preflight._input_names(spec))


def _required_inputs(spec: Mapping[str, Any]) -> set[str]:
    raw = spec.get("input")
    if not isinstance(raw, Mapping):
        return set()
    required = raw.get("required")
    if isinstance(required, Mapping):
        return {str(name) for name in required}
    if isinstance(required, list):
        return {str(name) for name in required if isinstance(name, str)}
    return set()


def _raw_input_type(spec: Mapping[str, Any], name: str) -> Any:
    kind, _enum, _options = i1_probe._input_definition(spec, name)
    return kind


def _check_scalar(value: Any, kind: Any, *, class_name: str, input_name: str) -> None:
    if kind == "INT":
        if isinstance(value, bool) or not isinstance(value, int):
            raise KF1Error(f"bound graph {class_name}.{input_name} must be INT")
    elif kind == "FLOAT":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise KF1Error(f"bound graph {class_name}.{input_name} must be FLOAT")
    elif kind == "STRING":
        if not isinstance(value, str):
            raise KF1Error(f"bound graph {class_name}.{input_name} must be STRING")
    elif kind == "BOOLEAN":
        if not isinstance(value, bool):
            raise KF1Error(f"bound graph {class_name}.{input_name} must be BOOLEAN")
    elif kind in {"COMBO", "COMFY_DYNAMICCOMBO_V3", None}:
        if not isinstance(value, (str, int, float, bool)):
            raise KF1Error(f"bound graph {class_name}.{input_name} has invalid literal")
    else:
        raise KF1Error(
            f"bound graph {class_name}.{input_name} must be a link, not literal {kind!r}"
        )


def _check_range(
    value: Any,
    options: Mapping[str, Any],
    *,
    class_name: str,
    input_name: str,
) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    checked = False
    minimum = options.get("min")
    maximum = options.get("max")
    if isinstance(minimum, (int, float)):
        checked = True
        if value < minimum:
            raise KF1Error(
                f"bound graph {class_name}.{input_name}={value} below min {minimum}"
            )
    if isinstance(maximum, (int, float)):
        checked = True
        if value > maximum:
            raise KF1Error(
                f"bound graph {class_name}.{input_name}={value} above max {maximum}"
            )
    return checked


def _expected_upload_path(name: str, subfolder: str) -> str:
    if subfolder:
        return f"{_UPLOAD_TYPES}/{subfolder}/{name}"
    return f"{_UPLOAD_TYPES}/{name}"


def validate_upload_receipt(
    receipt: Mapping[str, Any] | None,
    *,
    expected_name: str,
    expected_subfolder: str = "",
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate an exact server upload receipt for a bound input literal."""

    if not isinstance(receipt, Mapping):
        raise KF1Error(f"нет trusted upload receipt для {expected_name}")
    name = receipt.get("name")
    if name != expected_name:
        raise KF1Error("upload receipt name не совпадает с bound literal")
    if receipt.get("requested_name") not in (None, expected_name):
        raise KF1Error("upload receipt requested_name не совпадает")
    if receipt.get("type") != _UPLOAD_TYPES:
        raise KF1Error("upload receipt type должен быть input")
    subfolder = receipt.get("subfolder", "")
    if not isinstance(subfolder, str) or subfolder != expected_subfolder:
        raise KF1Error("upload receipt subfolder не совпадает с expected target")
    expected_path = _expected_upload_path(str(name), expected_subfolder)
    if receipt.get("path") != expected_path:
        raise KF1Error(
            f"upload receipt path {receipt.get('path')!r} != {expected_path!r}"
        )
    if expected_sha256 is not None:
        if str(receipt.get("source_sha256") or "") != expected_sha256:
            raise KF1Error("upload receipt source hash не совпадает с upload")
    return dict(receipt)


def validate_bound_workflow(
    workflow: Mapping[str, Any],
    object_info: Mapping[str, Any],
    *,
    upload_receipts: Mapping[str, Mapping[str, Any]] | None = None,
    expected_output: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the exact bound graph against fresh actual object_info."""

    if not isinstance(workflow, Mapping) or not workflow:
        raise KF1Error("bound graph пуст/не объект")
    if not isinstance(object_info, Mapping) or not object_info:
        raise KF1Error("object_info пуст; bound graph gate невозможен")
    receipts = dict(upload_receipts or {})
    inputs_checked = 0
    links_checked = 0
    ranges_checked = 0
    job_binding_required: list[dict[str, Any]] = []
    for node_id, node in workflow.items():
        if not isinstance(node, Mapping) or not isinstance(node.get("inputs"), Mapping):
            raise KF1Error(f"bound node {node_id} некорректен")
        class_name = str(node.get("class_type"))
        spec = object_info.get(class_name)
        if not isinstance(spec, Mapping):
            raise KF1Error(f"object_info не содержит bound class {class_name}")
        declared = _declared_inputs(spec)
        if not declared:
            raise KF1Error(f"object_info class {class_name} без input schema")
        inputs = node["inputs"]
        for input_name, value in inputs.items():
            name = str(input_name)
            if name not in declared:
                prefix = name.split(".", 1)[0]
                if (
                    "." in name
                    and prefix in declared
                    and _raw_input_type(spec, prefix) == "COMFY_DYNAMICCOMBO_V3"
                ):
                    if not isinstance(value, (str, int, float, bool)):
                        raise KF1Error(
                            f"bound graph {class_name}.{name} dynamic literal некорректен"
                        )
                    inputs_checked += 1
                    continue
                raise KF1Error(
                    f"bound graph {class_name}.{name} отсутствует в actual object_info"
                )
            kind, enum, options = i1_probe._input_definition(spec, name)
            if isinstance(value, list):
                if (
                    len(value) != 2
                    or not isinstance(value[0], str)
                    or isinstance(value[1], bool)
                    or not isinstance(value[1], int)
                ):
                    raise KF1Error(f"bound graph {class_name}.{name} link некорректен")
                source = workflow.get(value[0])
                if not isinstance(source, Mapping):
                    raise KF1Error(f"bound graph {class_name}.{name} source node отсутствует")
                source_class = str(source.get("class_type"))
                source_spec = object_info.get(source_class)
                if not isinstance(source_spec, Mapping):
                    raise KF1Error(f"bound graph source class {source_class} отсутствует")
                source_type = i1_probe._output_type(source_spec, int(value[1]))
                if (
                    not source_type
                    or not kind
                    or (source_type != kind and str(source_type) != "*" and str(kind) != "*")
                ):
                    raise KF1Error(
                        f"bound graph link {class_name}.{name}: "
                        f"{source_type!r} != {kind!r}"
                    )
                links_checked += 1
                continue
            inputs_checked += 1
            if enum is not None:
                if value not in enum:
                    if (
                        class_name == "LoadImage"
                        and name == "image"
                        and options.get("image_upload") is True
                    ):
                        receipt = receipts.get(str(value))
                        validate_upload_receipt(receipt, expected_name=str(value))
                        job_binding_required.append(
                            {
                                "node_id": str(node_id),
                                "class": class_name,
                                "input": name,
                                "value": value,
                                "status": "JOB_BINDING_REQUIRED",
                                "reason": "dynamic_image_upload",
                            }
                        )
                        continue
                    raise KF1Error(
                        f"bound graph enum mismatch {class_name}.{name}={value!r}"
                    )
                if class_name == "LoadImage" and name == "image":
                    # Even when the file enum already lists our upload, require
                    # the exact trusted receipt rather than trusting a name.
                    validate_upload_receipt(
                        receipts.get(str(value)), expected_name=str(value)
                    )
                continue
            _check_scalar(value, kind, class_name=class_name, input_name=name)
            if _check_range(value, options, class_name=class_name, input_name=name):
                ranges_checked += 1
        for required in _required_inputs(spec):
            if required not in inputs:
                # Dotted dynamic children are represented by their parent key in
                # object_info; a genuinely missing top-level required input is a
                # hard failure.
                raise KF1Error(
                    f"bound graph {class_name} missing required input {required!r}"
                )
    if expected_output is not None:
        output_node = str(expected_output.get("node_id"))
        node = workflow.get(output_node)
        if not isinstance(node, Mapping):
            raise KF1Error("bound graph expected output node отсутствует")
        class_name = str(node.get("class_type"))
        spec = object_info.get(class_name)
        if not isinstance(spec, Mapping):
            raise KF1Error("object_info не содержит expected output class")
        declared = _declared_inputs(spec)
        for key in expected_output.get("keys") or []:
            if key not in declared:
                raise KF1Error(f"output class {class_name} не содержит key {key!r}")
    return {
        "status": "BOUND_GRAPH_OK",
        "inputs_checked": inputs_checked,
        "links_checked": links_checked,
        "ranges_checked": ranges_checked,
        "job_binding_required": job_binding_required,
    }


def validate_runtime_object_info(
    profile: Any, object_info: Mapping[str, Any]
) -> dict[str, Any]:
    """Installation/readiness capability gate for every pinned workflow."""

    if not isinstance(object_info, Mapping) or not object_info:
        raise KF1Error("object_info пуст; runtime schema gate невозможен")
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", delete=False, newline="\n"
    )
    path = Path(handle.name)
    try:
        with handle:
            handle.write(json.dumps(dict(object_info), sort_keys=True))
        results = {}
        all_binding_required: list[dict[str, Any]] = []
        deadline = time.time() + 60
        for workflow_id, meta in profile.workflows.items():
            output = dict(meta["output"])
            facts = i1_probe._object_info_facts(
                path,
                required_classes=list(meta["required_classes"]),
                required_inputs=dict(meta["required_inputs"]),
                graph_inputs=list(meta["graph_inputs"]),
                output_specs=[{"workflow_id": workflow_id, **output}],
                deadline=deadline,
            )
            if any(check.get("present") is not True for check in facts["input_checks"]):
                raise KF1Error(
                    f"{workflow_id}: actual object_info не содержит required inputs"
                )
            if any(
                check.get("compatible") is not True for check in facts["link_checks"]
            ):
                raise KF1Error(
                    f"{workflow_id}: actual object_info links несовместимы"
                )
            if any(check.get("present") is not True for check in facts["output_checks"]):
                raise KF1Error(
                    f"{workflow_id}: actual object_info output schema не подтверждён"
                )
            binding_required = list(facts.get("job_binding_required") or [])
            for item in binding_required:
                all_binding_required.append({"workflow_id": workflow_id, **item})
            results[workflow_id] = {
                "classes": facts["classes"],
                "inputs": len(facts["input_checks"]),
                "links": len(facts["link_checks"]),
                "output": output["class"],
                "job_binding_required": binding_required,
            }
        return {
            "status": "OBJECT_INFO_OK",
            "workflows": results,
            "job_binding_required": all_binding_required,
        }
    except i1_probe.ProbeError as exc:
        raise KF1Error(str(exc)) from exc
    finally:
        path.unlink(missing_ok=True)


__all__ = [
    "validate_bound_workflow",
    "validate_runtime_object_info",
    "validate_upload_receipt",
]
