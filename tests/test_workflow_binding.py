from __future__ import annotations

import json
from pathlib import Path

import pytest

from anime_studio.errors import WorkflowBindingError
from anime_studio.workflow import (
    TEMPLATE_MARKER,
    WorkflowBinding,
    load_workflow,
    safe_relative_subfolder,
    safe_remote_filename,
)
from tests.p1_helpers import binding_dict, workflow_nodes, write_workflow


ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_workflow_is_explicitly_a_template() -> None:
    workflow, digest = load_workflow(ROOT / "workflows" / "P1" / "p1-i2v.api.template.json")
    assert len(digest) == 64
    assert TEMPLATE_MARKER in json.dumps(workflow)
    assert workflow["_meta"]["status"] == TEMPLATE_MARKER


def test_binding_injects_only_declared_scalars_and_drops_metadata() -> None:
    binding = WorkflowBinding.from_dict(binding_dict())
    workflow = workflow_nodes()
    bound = binding.bind(
        workflow,
        prompt="new prompt",
        seed=9,
        image_name="uploaded.png",
        width=704,
        height=1280,
        frames=81,
        fps=24,
    )
    assert "_meta" not in bound
    assert bound["2"]["inputs"]["text"] == "new prompt"
    assert bound["6"]["inputs"]["seed"] == 9
    assert bound["1"]["inputs"]["image"] == "uploaded.png"
    assert bound["5"]["inputs"]["length"] == 81
    assert bound["5"]["inputs"]["fps"] == 24


def test_binding_rejects_missing_node_wrong_type_and_missing_input() -> None:
    binding = WorkflowBinding.from_dict(binding_dict())

    missing = workflow_nodes()
    del missing["7"]
    assert any("output node 7" in issue for issue in binding.validate(missing))

    wrong = workflow_nodes()
    wrong["7"]["class_type"] = "SomethingElse"
    assert any("class_type" in issue for issue in binding.validate(wrong))

    no_input = workflow_nodes()
    del no_input["5"]["inputs"]["length"]
    assert any("missing input 'length'" in issue for issue in binding.validate(no_input))

    with pytest.raises(WorkflowBindingError):
        binding.bind(
            missing,
            prompt="p",
            seed=1,
            image_name="x.png",
            width=704,
            height=1280,
            frames=81,
            fps=24,
        )


@pytest.mark.parametrize("value", ["../x.png", "a\\b.png", "C:\\x.png", "x\x00.png"])
def test_unsafe_image_name_is_rejected(value: str) -> None:
    binding = WorkflowBinding.from_dict(binding_dict())
    with pytest.raises(WorkflowBindingError):
        binding.bind(
            workflow_nodes(),
            prompt="p",
            seed=1,
            image_name=value,
            width=704,
            height=1280,
            frames=81,
            fps=24,
        )


def test_subfolder_validation() -> None:
    assert safe_relative_subfolder("") == ""
    assert safe_relative_subfolder("nested/path") == "nested/path"
    with pytest.raises(WorkflowBindingError):
        safe_relative_subfolder("../escape")
    with pytest.raises(WorkflowBindingError):
        safe_relative_subfolder("a\\b")


def test_remote_filename_validation() -> None:
    assert safe_remote_filename("result.mp4") == "result.mp4"
    for value in ["", "..", "a/b.mp4", "a\\b.mp4", "C:out.mp4", "CON.mp4"]:
        with pytest.raises(WorkflowBindingError):
            safe_remote_filename(value)


def test_template_written_by_helper_is_template(tmp_path) -> None:
    path = tmp_path / "workflow.json"
    write_workflow(path, template=True)
    workflow, _ = load_workflow(path)
    assert TEMPLATE_MARKER in json.dumps(workflow)

def test_binding_rejects_missing_edge_target() -> None:
    binding = WorkflowBinding.from_dict(binding_dict())
    broken = workflow_nodes()
    broken["6"]["inputs"]["model"] = ["does-not-exist", 0]
    issues = binding.validate(broken)
    assert any("references missing node" in issue for issue in issues)


def test_duplicate_node_input_mapping_is_rejected() -> None:
    duplicate_width_height = binding_dict()
    duplicate_width_height["height"] = dict(duplicate_width_height["width"])
    with pytest.raises(WorkflowBindingError):
        WorkflowBinding.from_dict(duplicate_width_height)

    duplicate_prompt_image = binding_dict()
    duplicate_prompt_image["image"] = dict(duplicate_prompt_image["prompt"])
    with pytest.raises(WorkflowBindingError):
        WorkflowBinding.from_dict(duplicate_prompt_image)


@pytest.mark.parametrize("edge_index", [-1, 1.5, True])
def test_negative_or_fractional_edge_index_is_rejected(edge_index) -> None:
    binding = WorkflowBinding.from_dict(binding_dict())
    broken = workflow_nodes()
    broken["6"]["inputs"]["model"] = ["4", edge_index]
    issues = binding.validate(broken)
    assert any("malformed edge" in issue for issue in issues)
