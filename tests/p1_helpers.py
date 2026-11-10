from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anime_studio.artifacts import CompositeArtifactStore, HttpArtifactStore, LocalArtifactStore
from anime_studio.comfy_http import ComfyHttpTransport
from anime_studio.db import Database
from anime_studio.orchestrator import Orchestrator
from anime_studio.p1_config import P1RuntimeConfig, load_config
from anime_studio.worker import Worker


PNG_FIXTURE = b"\x89PNG\r\n\x1a\nfixture-not-a-real-model-input"


def workflow_nodes(*, template: bool = False) -> dict[str, Any]:
    marker = "TEMPLATE_NOT_EXECUTABLE" if template else "fixture-model.safetensors"
    payload: dict[str, Any] = {
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": marker + ".png"},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "old prompt", "clip": ["4", 0]},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "negative", "clip": ["4", 0]},
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": marker},
        },
        "5": {
            "class_type": "WanImageToVideo",
            "inputs": {
                "width": 704,
                "height": 1280,
                "length": 81,
                "fps": 24,
                "image": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "model": ["4", 0],
            },
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 1,
                "model": ["5", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["5", 0],
            },
        },
        "7": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["6", 0],
                "frame_rate": 24,
                "filename_prefix": "fixture/p1",
            },
        },
    }
    if template:
        payload["_meta"] = {"status": "TEMPLATE_NOT_EXECUTABLE"}
    return payload


def binding_dict() -> dict[str, Any]:
    return {
        "node_types": {
            "1": "LoadImage",
            "2": "CLIPTextEncode",
            "3": "CLIPTextEncode",
            "5": "WanImageToVideo",
            "6": "KSampler",
            "7": "VHS_VideoCombine",
        },
        "prompt": {"node": "2", "input": "text"},
        "seed": {"node": "6", "input": "seed"},
        "image": {"node": "1", "input": "image"},
        "width": {"node": "5", "input": "width"},
        "height": {"node": "5", "input": "height"},
        "frames": {"node": "5", "input": "length"},
        "fps": {"node": "5", "input": "fps"},
        "output": {"node": "7", "keys": ["gifs"]},
    }


def write_workflow(path: Path, *, template: bool = False) -> dict[str, Any]:
    data = workflow_nodes(template=template)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def write_config(
    tmp_path: Path,
    *,
    base_url: str,
    workflow_path: Path,
    manifest_path: Path,
    input_image_path: Path | None = None,
    workflow_sha256: str | None = None,
    exclusive_worker: bool = True,
    upload_timeout_seconds: float = 2.0,
    artifact_timeout_seconds: float = 2.0,
    read_timeout_seconds: float = 2.0,
) -> Path:
    if input_image_path is None:
        input_image_path = tmp_path / "input.png"
        input_image_path.write_bytes(PNG_FIXTURE)
    config = {
        "schema": "anime-video-studio/p1-config/v1",
        "runtime": {
            "base_url": base_url,
            "token_env": "ANIME_STUDIO_COMFY_TOKEN",
            "exclusive_worker": exclusive_worker,
            "client_id": "test-client",
            "connect_timeout_seconds": 2.0,
            "read_timeout_seconds": read_timeout_seconds,
            "upload_timeout_seconds": upload_timeout_seconds,
            "artifact_timeout_seconds": artifact_timeout_seconds,
            "max_upload_bytes": 1024 * 1024,
            "max_artifact_bytes": 1024 * 1024,
            "width": 704,
            "height": 1280,
            "frames": 81,
            "fps": 24,
            "input_image_path": str(input_image_path),
            "input_image_suffixes": [".png"],
            "workflow_path": str(workflow_path),
            "workflow_sha256": workflow_sha256,
            "manifest_path": str(manifest_path),
        },
        "binding": binding_dict(),
    }
    path = tmp_path / "p1-config.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def make_manifest(
    *,
    workflow_sha256: str,
    status: str = "VERIFIED",
    verified_fields: bool = True,
) -> dict[str, Any]:
    bad = "NOT_VERIFIED"
    zero64 = "0" * 64
    manifest: dict[str, Any] = {
        "schema": "anime-video-studio/p1-manifest/v1",
        "manifest_id": "P1-TEST",
        "status": status,
        "container": {
            "image": "registry.invalid/p1@sha256:" + ("a" * 64) if verified_fields else bad,
            "digest": "sha256:" + ("a" * 64) if verified_fields else bad,
        },
        "comfyui": {"commit": "b" * 40 if verified_fields else bad},
        "custom_nodes": [
            {
                "name": "fixture-node",
                "commit": "c" * 40 if verified_fields else bad,
                "source_url": "https://example.invalid/node",
            }
        ],
        "workflow": {
            "path": "workflow.json",
            "sha256": workflow_sha256 if verified_fields else bad,
        },
        "models": [
            {
                "id": "fixture-model",
                "role": "i2v",
                "source_url": "https://example.invalid/model",
                "revision": "d" * 40 if verified_fields else bad,
                "filename": "fixture.safetensors" if verified_fields else bad,
                "sha256": ("e" * 64 if verified_fields else bad),
                "license_url": "https://example.invalid/license",
                "license_reviewed_at": "2026-10-20" if verified_fields else None,
                "hosted_use": "allowed" if verified_fields else "NOT_VERIFIED",
                "commercial_use_status": "allowed" if verified_fields else "unknown",
            }
        ],
        "runtime": {
            "gpu": "fixture-gpu",
            "vram_gb": 24 if verified_fields else None,
            "ram_gb": 32 if verified_fields else None,
            "disk_gb": 100 if verified_fields else None,
            "estimated": True,
        },
        "checked_at": "2026-10-20" if verified_fields else None,
    }
    if not verified_fields:
        manifest["container"]["digest"] = bad
        manifest["workflow"]["sha256"] = bad
    return manifest


def write_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def config_from_path(path: Path) -> P1RuntimeConfig:
    return load_config(path, base_dir=path.parent)


def make_transport(
    tmp_path: Path,
    *,
    base_url: str,
    exclusive_worker: bool = True,
    prompt_lost_grace_seconds: float = 0.0,
    upload_timeout_seconds: float = 2.0,
    artifact_timeout_seconds: float = 2.0,
    read_timeout_seconds: float = 2.0,
) -> ComfyHttpTransport:
    from anime_studio.workflow import load_workflow, workflow_sha256

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
        exclusive_worker=exclusive_worker,
        upload_timeout_seconds=upload_timeout_seconds,
        artifact_timeout_seconds=artifact_timeout_seconds,
        read_timeout_seconds=read_timeout_seconds,
    )
    config = config_from_path(config_path)
    workflow, _ = load_workflow(workflow_path)
    return ComfyHttpTransport(
        config,
        workflow,
        prompt_lost_grace_seconds=prompt_lost_grace_seconds,
    )


@dataclass
class RealStack:
    config: P1RuntimeConfig
    transport: ComfyHttpTransport
    db: Database
    artifacts: CompositeArtifactStore
    orchestrator: Orchestrator
    worker: Worker


def make_real_stack(
    tmp_path: Path,
    *,
    base_url: str,
    exclusive_worker: bool = True,
    prompt_lost_grace_seconds: float = 0.0,
) -> RealStack:
    from anime_studio.workflow import load_workflow, workflow_sha256

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
        exclusive_worker=exclusive_worker,
    )
    config = config_from_path(config_path)
    workflow, _ = load_workflow(workflow_path)
    transport = ComfyHttpTransport(
        config,
        workflow,
        prompt_lost_grace_seconds=prompt_lost_grace_seconds,
    )
    db = Database(tmp_path / "p1.sqlite3")
    db.init()
    local = LocalArtifactStore(tmp_path / "artifacts")
    remote = HttpArtifactStore(
        tmp_path / "artifacts",
        base_url=base_url,
        timeout_seconds=2.0,
        max_bytes=1024 * 1024,
    )
    artifacts = CompositeArtifactStore(local, remote)
    orchestrator = Orchestrator(db, transport, artifacts, lease_seconds=0.5)
    worker = Worker(orchestrator, poll_interval=0.01, lease_seconds=0.5)
    return RealStack(config, transport, db, artifacts, orchestrator, worker)


__all__ = [
    "PNG_FIXTURE",
    "RealStack",
    "binding_dict",
    "config_from_path",
    "make_manifest",
    "make_real_stack",
    "workflow_nodes",
    "write_config",
    "write_manifest",
    "write_workflow",
]
