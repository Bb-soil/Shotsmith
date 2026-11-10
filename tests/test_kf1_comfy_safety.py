from __future__ import annotations

import pytest

from scripts.kf1.comfy import (
    ComfyClient,
    ComfyError,
    HistoryContractError,
    output_from_history,
    validate_base_url,
)
from tests.kf1_fake_comfy import FakeComfy


def test_only_loopback_base_url() -> None:
    assert validate_base_url("http://127.0.0.1:8188")
    for value in (
        "http://example.com:8188",
        "https://user:pass@127.0.0.1:8188",
        "http://127.0.0.1:8188/?token=x",
    ):
        with pytest.raises(ComfyError):
            validate_base_url(value)


def _entry(
    filename: str = "ok.png",
    *,
    subfolder: str = "",
    artifact_type: str = "output",
    prompt_id: str = "prompt-1",
    completed: bool = True,
    extra: dict | None = None,
) -> dict:
    return {
        "prompt": [
            0,
            prompt_id,
            {"70": {"class_type": "SaveImage"}},
            extra or {"correlation_key": "c", "nonce": "n"},
            [],
        ],
        "status": {
            "status_str": "success" if completed else "running",
            "completed": completed,
            "messages": [],
        },
        "outputs": {
            "70": {
                "images": [
                    {"filename": filename, "subfolder": subfolder, "type": artifact_type}
                ]
            }
        },
    }


def test_output_from_history_rejects_unsafe_names_and_non_png() -> None:
    with pytest.raises(Exception):
        output_from_history(_entry("../escape.png"), node_id="70", keys=["images"])
    assert output_from_history(_entry(), node_id="70", keys=["images"])["filename"] == "ok.png"
    with pytest.raises(Exception):
        output_from_history(_entry(artifact_type="temp"), node_id="70", keys=["images"])
    with pytest.raises(Exception):
        output_from_history(_entry(filename="ok.jpg"), node_id="70", keys=["images"])
    with pytest.raises(Exception):
        output_from_history(_entry(completed=False), node_id="70", keys=["images"])
    with pytest.raises(Exception):
        output_from_history(
            _entry(filename="other_00001_.png"),
            node_id="70",
            keys=["images"],
            expected_output={"subfolder": "", "filename_prefix": "expected"},
        )


def test_requested_prompt_id_must_be_exact_history_key() -> None:
    entry = _entry(prompt_id="other")
    with pytest.raises(HistoryContractError):
        # exact key is enforced by the client; the helper also rejects prompt-id
        # mismatch when expected is supplied by the caller.
        output_from_history(
            entry,
            node_id="70",
            keys=["images"],
            expected_prompt_id="prompt-1",
        )


def test_view_rejects_redirect() -> None:
    fake = FakeComfy().start()
    fake.view_redirect = True
    try:
        client = ComfyClient(fake.base_url)
        with pytest.raises(ComfyError):
            client.view({"filename": "x.png", "subfolder": "", "type": "output"})
        client.close()
    finally:
        fake.stop()
