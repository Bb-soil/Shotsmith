from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from anime_studio.artifacts import HttpArtifactStore
from anime_studio.comfy_http import ComfyHttpTransport
from anime_studio.p1_config import load_config, load_manifest
from anime_studio.workflow import load_workflow
from tests.comfy_stub import MP4_FIXTURE, run_stub


ROOT = Path(__file__).resolve().parents[1]
DRAFT_CONFIG = ROOT / "config" / "p1.wan22-5b.draft.json"
DRAFT_WORKFLOW = ROOT / "workflows" / "P1" / "wan22-5b-shot01.api.draft.json"
DRAFT_MANIFEST = ROOT / "manifests" / "P1.wan22-5b.draft.json"
SHOT_JOB = ROOT / "config" / "p1.shot01.job.json"


def _prompt_and_seed() -> tuple[str, int]:
    job = json.loads(SHOT_JOB.read_text(encoding="utf-8"))
    return job["prompt"], job["seed"]


def test_prepared_wan_graph_binds_twelve_nodes_and_flat_savevideo_fields() -> None:
    config = load_config(DRAFT_CONFIG)
    workflow, workflow_sha = load_workflow(DRAFT_WORKFLOW)
    manifest = load_manifest(DRAFT_MANIFEST)
    prompt, seed = _prompt_and_seed()

    assert workflow_sha == config.workflow_sha256
    assert manifest["workflow"]["sha256"] == workflow_sha
    assert config.binding.validate(workflow) == []
    assert config.binding.output.keys == ("images",)

    bound = config.binding.bind(
        workflow,
        prompt=prompt,
        seed=seed,
        image_name="shot-01-start-v2.png",
        width=config.width,
        height=config.height,
        frames=config.frames,
        fps=config.fps,
    )

    assert len(bound) == 12
    assert "_meta" not in bound
    assert bound["55"]["class_type"] == "Wan22ImageToVideoLatent"
    assert bound["55"]["inputs"]["start_image"] == ["56", 0]
    assert bound["55"]["inputs"]["width"] == 704
    assert bound["55"]["inputs"]["height"] == 1280
    assert bound["55"]["inputs"]["length"] == 121
    assert bound["56"]["class_type"] == "LoadImage"
    assert bound["56"]["inputs"]["image"] == "shot-01-start-v2.png"
    assert bound["38"]["class_type"] == "CLIPLoader"
    assert bound["38"]["inputs"]["type"] == "wan"
    assert bound["38"]["inputs"]["device"] == "cpu"
    assert bound["57"]["class_type"] == "CreateVideo"
    assert bound["57"]["inputs"]["fps"] == 24
    assert bound["3"]["class_type"] == "KSampler"
    assert bound["3"]["inputs"]["seed"] == 424242
    assert bound["3"]["inputs"]["steps"] == 20
    assert bound["3"]["inputs"]["cfg"] == 5
    assert bound["3"]["inputs"]["sampler_name"] == "uni_pc"
    assert bound["3"]["inputs"]["scheduler"] == "simple"
    assert bound["48"]["inputs"]["shift"] == 8
    assert bound["58"]["class_type"] == "SaveVideo"
    assert bound["58"]["inputs"]["format"] == "mp4"
    assert bound["58"]["inputs"]["format.codec"] == "h264"
    assert bound["58"]["inputs"]["format.codec.encoding"] == "auto"


def test_native_savevideo_images_history_returns_mp4_artifact(tmp_path) -> None:
    prompt, seed = _prompt_and_seed()
    with run_stub() as (base_url, state):
        config = replace(load_config(DRAFT_CONFIG), base_url=base_url)
        workflow, _ = load_workflow(DRAFT_WORKFLOW)
        transport = ComfyHttpTransport(
            config,
            workflow,
            prompt_lost_grace_seconds=0.0,
        )
        remote = transport.submit(
            "corr-native-savevideo",
            {"prompt": prompt, "seed": seed},
        )

        state.queue_pending = []
        state.queue_running = []
        state.history[state.prompt_id] = {
            "status": {"status_str": "success", "completed": True, "messages": []},
            "outputs": {
                "58": {
                    "images": [
                        {
                            "filename": "shot-01_00001_.mp4",
                            "subfolder": "anime-studio/P1",
                            "type": "output",
                        }
                    ]
                }
            },
        }

        succeeded = transport.get(remote.prompt_id)
        assert succeeded is not None
        assert succeeded.state == "succeeded"
        assert succeeded.artifact is not None
        assert succeeded.artifact["filename"] == "shot-01_00001_.mp4"
        assert succeeded.artifact["subfolder"] == "anime-studio/P1"
        assert succeeded.artifact["media_type"] == "video/mp4"

        store = HttpArtifactStore(
            tmp_path / "artifacts",
            base_url=base_url,
            timeout_seconds=2.0,
            max_bytes=1024 * 1024,
        )
        record = store.fetch_result(succeeded.artifact, "native-job-1")
        assert record["mock"] is False
        assert record["media_type"] == "video/mp4"
        assert record["relative_path"] == "native-job-1/result.mp4"
        assert Path(record["path"]).is_file()
        assert Path(record["path"]).stat().st_size == len(MP4_FIXTURE)
        assert state.prompt_requests == 1
        assert state.upload_requests == 1
