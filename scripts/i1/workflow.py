"""Workflow registration with honest API/UI status labels."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from scripts.p1.remote import RemoteRunner

from . import remote as i1_remote
from .errors import RemoteFailure
from .profile import Profile, verify_profile_sources
from .util import sha256_file, validate_workspace

USER_WORKFLOWS = ("ComfyUI", "user", "default", "workflows")


def api_remote_path(root: str, profile: Profile) -> str:
    return str(PurePosixPath(root) / "workflow" / f"{profile.profile_id}.api.json")


def api_remote_path_for(root: str, profile: Profile, workflow_id: str) -> str:
    if profile.data.get("schema") == "anime-video-studio/i1-profile/v2":
        return str(
            PurePosixPath(root)
            / "workflow"
            / profile.profile_id
            / f"{workflow_id}.api.json"
        )
    return api_remote_path(root, profile)


def ui_remote_path(workspace: str, profile: Profile) -> str:
    base = PurePosixPath(validate_workspace(workspace)).joinpath(*USER_WORKFLOWS)
    return str(base / f"anime-studio-{profile.profile_id}-ui.json")


def register(
    runner: RemoteRunner,
    identity: Any,
    *,
    root: str,
    workspace: str,
    profile: Profile,
    repo_root: Path | str,
    timeout: float = 60.0,
) -> dict[str, Any]:
    verify_profile_sources(profile, repo_root)
    repo = Path(repo_root)
    workflow_dir = str(PurePosixPath(root) / "workflow")
    i1_remote.mkdir(
        runner,
        identity,
        workflow_dir,
        str(PurePosixPath(validate_workspace(workspace)).joinpath(*USER_WORKFLOWS)),
        timeout=timeout,
    )
    staged_workflows = []
    for workflow_id, meta in profile.workflow_items():
        api_local = repo / meta["api_path"]
        api_remote = api_remote_path_for(root, profile, workflow_id)
        i1_remote.mkdir(
            runner,
            identity,
            str(PurePosixPath(api_remote).parent),
            timeout=timeout,
        )
        i1_remote.stage_file(
            runner,
            identity,
            api_remote,
            api_local,
            root=root,
            timeout=timeout,
        )
        api_payload = i1_remote.read_json(
            runner, identity, root=root, remote_path=api_remote, timeout=timeout
        )
        if sha256_file(api_local) != meta["api_sha256"]:
            raise RemoteFailure("local API workflow hash changed before staging")
        if not isinstance(api_payload, dict):
            raise RemoteFailure("remote API workflow is not an object")
        staged_workflows.append(
            {
                "workflow_id": workflow_id,
                "api_path": api_remote,
                "api_sha256": meta["api_sha256"],
                "media_type": (meta.get("output") or {}).get("media_type"),
            }
        )
    primary = profile.workflow
    primary_staged = next(
        item for item in staged_workflows if item["workflow_id"] == profile.primary_workflow_id
    )
    ui_remote = None
    ui_local = None
    ui_status = "API_FILE_ONLY"
    if profile.data.get("schema") != "anime-video-studio/i1-profile/v2" and primary.get("ui_path"):
        ui_local = repo / primary["ui_path"]
        ui_remote = ui_remote_path(workspace, profile)
        i1_remote.stage_file(
            runner,
            identity,
            ui_remote,
            ui_local,
            root=str(PurePosixPath(validate_workspace(workspace)) / "ComfyUI"),
            timeout=timeout,
        )
        ui_payload = i1_remote.read_json(
            runner,
            identity,
            root=str(PurePosixPath(validate_workspace(workspace)) / "ComfyUI"),
            remote_path=ui_remote,
            timeout=timeout,
            helper_root=root,
        )
        if not isinstance(ui_payload, dict):
            raise RemoteFailure("remote UI workflow is not an object")
        if sha256_file(ui_local) != primary["ui_sha256"]:
            raise RemoteFailure("local UI workflow hash changed before staging")
        ui_status = str(primary.get("ui_status") or "UI_GRAPH_NOT_VERIFIED")
    return {
        "api_path": primary_staged["api_path"],
        "api_sha256": primary_staged["api_sha256"],
        "workflows": staged_workflows,
        "ui_path": ui_remote,
        "ui_sha256": primary.get("ui_sha256"),
        "ui_status": ui_status,
        "ui_filename": PurePosixPath(ui_remote).name if ui_remote else None,
        "ui_graph_rendered": False,
    }


def _default_userdata_fetch(url: str, timeout: float) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read(256 * 1024)


def list_userdata(
    *,
    local_port: int,
    filename: str | None,
    timeout: float = 5.0,
    fetch: Callable[[str, float], Any] | None = None,
) -> str:
    """Return UI_LISTED/UI_FILE_ONLY/UI_LISTING_UNAVAILABLE.

    The endpoint is a ComfyUI convention, not a proof that the canvas rendered.
    A 404/HTML/JSON failure is reported as unavailable, never as READY.
    """

    if not filename:
        return "API_FILE_ONLY"
    url = f"http://127.0.0.1:{int(local_port)}/userdata?dir=workflows&recurse=true"
    try:
        raw = (fetch or _default_userdata_fetch)(url, timeout)
        data = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else str(raw))
        if isinstance(data, list):
            names = {
                str(item.get("name") or item.get("path") or "")
                for item in data
                if isinstance(item, Mapping)
            }
            if filename in names or any(name.endswith("/" + filename) for name in names):
                return "UI_LISTED"
            return "UI_FILE_ONLY"
        if isinstance(data, Mapping):
            text = json.dumps(data, ensure_ascii=False)
            return "UI_LISTED" if filename in text else "UI_FILE_ONLY"
        return "UI_FILE_ONLY"
    except (OSError, urllib.error.URLError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return "UI_LISTING_UNAVAILABLE"


__all__ = [
    "USER_WORKFLOWS",
    "api_remote_path",
    "list_userdata",
    "register",
    "ui_remote_path",
]
