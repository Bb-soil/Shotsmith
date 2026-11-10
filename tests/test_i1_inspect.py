from __future__ import annotations

import json
from types import SimpleNamespace

from scripts.i1 import stock_nodes
from scripts.i1.inspect import inspect_remote
from scripts.i1.profile import load_profile

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
PINNED = "7a28332047b6af638dc6c7f72ef3481a718769fd"


class Runner:
    def __init__(self) -> None:
        self.commands = []
        self.stock_returncode = 0
        self.stock_nodes = {
            "schema": stock_nodes.SCHEMA,
            "ok": True,
            "head": PINNED,
            "expected": PINNED,
            "object_format": "sha1",
            "tracked": [
                "custom_nodes/example_node.py.example",
                "custom_nodes/websocket_image_save.py",
            ],
            "tracked_total": 2,
            "cache_unverified": [],
            "cache_dirs": [],
            "violations": [],
            "truncated": False,
        }

    def run_remote(self, identity, argv, timeout=None):
        self.commands.append(tuple(argv))
        if len(argv) >= 4 and tuple(argv[:3]) == ("python3", "-B", "-c"):
            if "anime-video-studio/i1-stock-nodes/v1" in str(argv[3]):
                return SimpleNamespace(
                    returncode=self.stock_returncode,
                    stdout=json.dumps(self.stock_nodes),
                    stderr="",
                    timed_out=False,
                )
        if len(argv) > 2 and "torch" in str(argv[2]):
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"version": "2.9.1", "cuda": "12.8", "available": True}),
                stderr="",
                timed_out=False,
            )
        if len(argv) > 2 and "expanduser" in str(argv[2]):
            return SimpleNamespace(
                returncode=0, stdout="/root\n", stderr="", timed_out=False
            )
        if len(argv) > 2 and "MemTotal" in str(argv[2]):
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"MemTotal": 64 * 1024**3, "MemAvailable": 60 * 1024**3}),
                stderr="",
                timed_out=False,
            )
        if argv[0] == "uname" and argv[1] == "-s":
            return SimpleNamespace(returncode=0, stdout="Linux\n", stderr="", timed_out=False)
        if argv[0] == "uname" and argv[1] == "-m":
            return SimpleNamespace(returncode=0, stdout="x86_64\n", stderr="", timed_out=False)
        if argv[0] == "id":
            return SimpleNamespace(returncode=0, stdout="1000\n", stderr="", timed_out=False)
        if argv[0] in {"bash", "git", "python3"} and argv[1] in {"--version", "-c"}:
            if argv[0] == "python3" and len(argv) > 2 and "MemTotal" in argv[2]:
                return SimpleNamespace(returncode=0, stdout=json.dumps({"MemTotal": 64 * 1024**3, "MemAvailable": 60 * 1024**3}), stderr="", timed_out=False)
            if len(argv) > 2 and (argv[0] == "python3" or "p1-venv" in argv[0]) and "torch" in argv[2]:
                return SimpleNamespace(returncode=0, stdout=json.dumps({"version": "2.9.1", "cuda": "12.8", "available": True}), stderr="", timed_out=False)
            return SimpleNamespace(returncode=0, stdout="ok\n", stderr="", timed_out=False)
        if argv[0] == "test":
            return SimpleNamespace(returncode=0, stdout="", stderr="", timed_out=False)
        if argv[0] == "df":
            return SimpleNamespace(returncode=0, stdout="Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/x 100000000 1 90000000 1% /workspace\n", stderr="", timed_out=False)
        if argv[0] == "nvidia-smi":
            return SimpleNamespace(returncode=0, stdout="Fake GPU, 24564, 999\n", stderr="", timed_out=False)
        if argv[0] == "git" and "-C" in argv:
            if "rev-parse" in argv:
                return SimpleNamespace(returncode=0, stdout="7a28332047b6af638dc6c7f72ef3481a718769fd\n", stderr="", timed_out=False)
            return SimpleNamespace(returncode=0, stdout="", stderr="", timed_out=False)
        if argv[0] == "find":
            return SimpleNamespace(returncode=0, stdout="", stderr="", timed_out=False)
        if argv[0] == "stat":
            return SimpleNamespace(returncode=0, stdout="1\n", stderr="", timed_out=False)
        return SimpleNamespace(returncode=0, stdout="", stderr="", timed_out=False)


def test_inspect_is_read_only_and_reports_facts() -> None:
    runner = Runner()
    profile = load_profile(ROOT / "config" / "i1" / "wan22-14b-ab.profile.json")
    result = inspect_remote(runner, object(), workspace="/workspace", profile=profile)
    assert result["compatible"] is True
    assert result["facts"]["os"] == "Linux"
    assert result["facts"]["arch"] == "x86_64"
    assert result["facts"]["gpu"]
    assert result["facts"]["custom_nodes"]["ok"] is True
    assert result["facts"]["custom_nodes"]["tracked"] == [
        "custom_nodes/example_node.py.example",
        "custom_nodes/websocket_image_save.py",
    ]
    assert all(command[0] not in {"mkdir", "put", "chmod", "rm", "pip"} for command in runner.commands)
    assert not any("--execute" in " ".join(command) for command in runner.commands)
    assert not any(command[0] == "find" for command in runner.commands)


def _inspect_with(runner: Runner) -> dict:
    profile = load_profile(ROOT / "config" / "i1" / "wan22-14b-ab.profile.json")
    return inspect_remote(runner, object(), workspace="/workspace", profile=profile)


def test_inspect_blocks_untracked_node() -> None:
    runner = Runner()
    runner.stock_returncode = 1
    runner.stock_nodes = {
        **runner.stock_nodes,
        "ok": False,
        "violations": [
            {
                "code": "untracked",
                "path": "custom_nodes/evil.py",
                "message": "untracked entry under custom_nodes",
            }
        ],
    }
    result = _inspect_with(runner)
    assert result["compatible"] is False
    assert any(item["code"] == "custom_nodes" for item in result["blockers"])
    assert "evil.py" in result["blockers"][0]["message"]


def test_inspect_blocks_changed_stock_node() -> None:
    runner = Runner()
    runner.stock_returncode = 1
    runner.stock_nodes = {
        **runner.stock_nodes,
        "ok": False,
        "violations": [
            {
                "code": "content",
                "path": "custom_nodes/websocket_image_save.py",
                "message": "tracked content differs from the pinned blob",
            }
        ],
    }
    result = _inspect_with(runner)
    assert result["compatible"] is False
    assert any(item["code"] == "custom_nodes" for item in result["blockers"])
    assert "content" in result["blockers"][0]["message"]


def test_inspect_blocks_symlinked_stock_node() -> None:
    runner = Runner()
    runner.stock_returncode = 1
    runner.stock_nodes = {
        **runner.stock_nodes,
        "ok": False,
        "violations": [
            {
                "code": "symlink",
                "path": "custom_nodes/websocket_image_save.py",
                "message": "tracked path contains a symlink",
            }
        ],
    }
    result = _inspect_with(runner)
    assert result["compatible"] is False
    assert any(item["code"] == "custom_nodes" for item in result["blockers"])


def test_inspect_reports_runtime_cache_as_warning_not_blocker() -> None:
    runner = Runner()
    runner.stock_nodes = {
        **runner.stock_nodes,
        "cache_unverified": [
            "custom_nodes/__pycache__/websocket_image_save.cpython-312.pyc"
        ],
        "cache_dirs": ["custom_nodes/__pycache__"],
    }
    result = _inspect_with(runner)
    assert result["compatible"] is True
    assert not any(item["code"] == "custom_nodes" for item in result["blockers"])
    assert any(item["code"] == "custom_nodes_cache" for item in result["warnings"])
    assert result["facts"]["custom_nodes"]["cache_unverified"]


def test_inspect_fails_closed_on_invalid_validator_output() -> None:
    class BadOutputRunner(Runner):
        def run_remote(self, identity, argv, timeout=None):
            if len(argv) >= 4 and tuple(argv[:3]) == ("python3", "-B", "-c"):
                if "anime-video-studio/i1-stock-nodes/v1" in str(argv[3]):
                    self.commands.append(tuple(argv))
                    return SimpleNamespace(
                        returncode=0, stdout="not-json\n", stderr="", timed_out=False
                    )
            return super().run_remote(identity, argv, timeout=timeout)

    result = _inspect_with(BadOutputRunner())
    assert result["compatible"] is False
    assert any(item["code"] == "custom_nodes" for item in result["blockers"])


def test_inspect_fails_closed_when_validator_reports_inconsistent_success() -> None:
    runner = Runner()
    runner.stock_returncode = 1
    runner.stock_nodes = {
        **runner.stock_nodes,
        "ok": True,
        "violations": [
            {"code": "validator_error", "path": "", "message": "inconsistent result"}
        ],
    }
    result = _inspect_with(runner)
    assert result["compatible"] is False
    assert any(item["code"] == "custom_nodes" for item in result["blockers"])


def test_reviewer_stock_scenario_now_has_no_custom_nodes_blocker() -> None:
    class StockRunner(Runner):
        def run_remote(self, identity, argv, timeout=None):
            if argv[0] == "find":
                self.commands.append(tuple(argv))
                return SimpleNamespace(
                    returncode=0,
                    stdout="/workspace/ComfyUI/custom_nodes/websocket_image_save.py\n",
                    stderr="",
                    timed_out=False,
                )
            return super().run_remote(identity, argv, timeout=timeout)

    result = _inspect_with(StockRunner())
    assert result["compatible"] is True
    assert not any(item["code"] == "custom_nodes" for item in result["blockers"])


def test_inspect_non_git_comfy_dir_blocks_without_unbound_error() -> None:
    class NoGitRunner(Runner):
        def run_remote(self, identity, argv, timeout=None):
            if argv[0] == "test" and len(argv) >= 3 and argv[-1].endswith("/.git"):
                self.commands.append(tuple(argv))
                return SimpleNamespace(
                    returncode=1, stdout="", stderr="", timed_out=False
                )
            return super().run_remote(identity, argv, timeout=timeout)

    result = _inspect_with(NoGitRunner())
    assert result["compatible"] is False
    assert any(item["code"] == "comfy_git" for item in result["blockers"])
    assert result["facts"]["custom_nodes"] is None


class BadContainmentRunner(Runner):
    def run_remote(self, identity, argv, timeout=None):
        if argv[0] == "python3" and len(argv) > 2 and "foreign_venv" in argv[2]:
            self.commands.append(tuple(argv))
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {"symlinks": ["/workspace/p1-venv"], "outside": [], "foreign_venv": False}
                ),
                stderr="",
                timed_out=False,
            )
        return super().run_remote(identity, argv, timeout=timeout)


def test_inspect_blocks_foreign_symlink_before_setup() -> None:
    profile = load_profile(ROOT / "config" / "i1" / "wan22-14b-ab.profile.json")
    result = inspect_remote(
        BadContainmentRunner(), object(), workspace="/workspace", profile=profile
    )
    assert result["compatible"] is False
    assert any(item["code"] == "symlink" for item in result["blockers"])
