from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts.i1 import workflow
from scripts.i1.profile import load_profile

ROOT = Path(__file__).resolve().parents[1]


class FakeRemote:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.puts = []

    def _local(self, remote: str) -> Path:
        return self.workspace / remote[len("/workspace/") :]

    def put_file(self, identity, remote_path, local_path, **kwargs):
        self.puts.append(remote_path)
        target = self._local(remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(local_path).read_bytes())

    def read_remote_exact_json(self, identity, *, helper_path, remote_path, root, max_bytes, timeout):
        return json.loads(self._local(remote_path).read_text(encoding="utf-8"))

    def run_remote(self, identity, argv, timeout=None, **kwargs):
        return SimpleNamespace(returncode=0, stdout="", stderr="", timed_out=False)


def test_register_keeps_api_separate_and_marks_ui_not_rendered(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / ".avs-i1").mkdir(parents=True)
    profile = load_profile(ROOT / "config" / "i1" / "wan22-14b-ab.profile.json")
    runner = FakeRemote(workspace)
    result = workflow.register(
        runner,
        object(),
        root="/workspace/.avs-i1",
        workspace="/workspace",
        profile=profile,
        repo_root=ROOT,
    )
    assert result["ui_status"] == "DERIVED_FROM_API_NOT_LIVE_VERIFIED"
    assert result["ui_graph_rendered"] is False
    api_remote = workspace / ".avs-i1" / "workflow" / f"{profile.profile_id}.api.json"
    ui_remote = workspace / "ComfyUI" / "user" / "default" / "workflows" / result["ui_filename"]
    assert api_remote.is_file()
    assert ui_remote.is_file()
    assert json.loads(api_remote.read_text(encoding="utf-8"))
    assert json.loads(ui_remote.read_text(encoding="utf-8"))


def test_userdata_listing_never_claims_rendered_graph() -> None:
    def fetch(url, timeout):
        return json.dumps([{"name": "anime-studio-x-ui.json"}]).encode()

    assert (
        workflow.list_userdata(local_port=8189, filename="anime-studio-x-ui.json", fetch=fetch)
        == "UI_LISTED"
    )
    assert (
        workflow.list_userdata(local_port=8189, filename="other.json", fetch=fetch)
        == "UI_FILE_ONLY"
    )
    assert (
        workflow.list_userdata(local_port=8189, filename="x.json", fetch=lambda *_: b"not-json")
        == "UI_LISTING_UNAVAILABLE"
    )
