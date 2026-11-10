from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts.p1 import v3_download_worker, v3_lifecycle, v3_remote_launcher

ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "remote-root"
    root.mkdir()
    for name in ("v3_remote_launcher.py", "v3_download_worker.py", "setup_plan.py"):
        (root / name).write_bytes((ROOT / "scripts" / "p1" / name).read_bytes())
    return root


def _manifest_file(root: Path, tag: str = "m") -> dict:
    models = []
    for index in range(3):
        data = f"{tag}-{index}-payload".encode()
        target = f"models/{tag}{index}.bin"
        path = root / "ComfyUI" / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        models.append(
            {
                "id": f"{tag}{index}",
                "download_url": f"https://example.invalid/{tag}{index}.bin",
                "target": target,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"models": models}), encoding="utf-8")
    return {"models": models}


def _fake_worker(root: Path) -> Path:
    worker = root / "v3_download_worker.py"
    worker.write_text(
        "import json,sys,time,pathlib,os\n"
        "payload=json.loads(pathlib.Path(sys.argv[1]).read_text())\n"
        "def write(path,payload):\n"
        "    pathlib.Path(path).write_text(json.dumps(payload))\n"
        "base={'run_id':payload['run_id'],'nonce':payload['nonce'],'status':'RUNNING'}\n"
        "write(payload['status_path'],base)\n"
        "write(payload['progress_path'],{**base,'files_done':0,'files_total':3,'bytes_done':0,'bytes_total':10})\n"
        "write(payload['heartbeat_path'],{**base,'pid':os.getpid()})\n"
        "time.sleep(3.0)\n"
        "done={**base,'status':'COMPLETE'}\n"
        "write(payload['status_path'],done)\n"
        "write(payload['progress_path'],{**done,'files_done':3,'files_total':3,'bytes_done':10,'bytes_total':10})\n"
        "write(payload['heartbeat_path'],{**done,'pid':os.getpid()})\n",
        encoding="utf-8",
    )
    return worker


def _payload(root: Path, *, deadline: float | None = None) -> tuple[Path, dict]:
    manifest = _manifest_file(root)
    payload = {
        "schema": "anime-video-studio/p1-v3-download-job/v1",
        "run_id": "run-remote-1",
        "nonce": "nonce-remote-1",
        "workspace": str(root),
        "root": str(root / "ComfyUI"),
        "deadline_epoch": deadline or (time.time() + 120),
        "manifest_path": str(root / "manifest.json"),
        "manifest_sha256": _sha(root / "manifest.json"),
        "required_bytes": sum(item["size_bytes"] for item in manifest["models"]),
        "status_path": str(root / "status.json"),
        "progress_path": str(root / "progress.json"),
        "heartbeat_path": str(root / "heartbeat.json"),
    }
    path = root / "job.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def _launch_manifest(root: Path, worker: Path, payload_path: Path, payload: dict) -> Path:
    manifest = {
        "schema": "anime-video-studio/p1-v3-launch/v1",
        "kind": "download_worker",
        "run_id": payload["run_id"],
        "nonce": payload["nonce"],
        "root": str(root),
        "worker": str(worker),
        "worker_sha256": _sha(worker),
        "payload": str(payload_path),
        "payload_sha256": _sha(payload_path),
        "deadline_epoch": payload["deadline_epoch"],
        "status": payload["status_path"],
        "progress": payload["progress_path"],
        "heartbeat": payload["heartbeat_path"],
        "log": str(root / "download.log"),
        "owner": str(root / "worker-owner.json"),
        "ack": str(root / "worker-ack.json"),
    }
    path = root / "launch.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _run_launcher(manifest: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(manifest.parent / "v3_remote_launcher.py"), "--launch", str(manifest), *extra],
        cwd=str(manifest.parent),
        capture_output=True,
        text=True,
        timeout=15,
    )


def _kill(pid: int) -> None:
    if not isinstance(pid, int) or pid <= 0:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
        else:
            os.kill(pid, 9)
    except OSError:
        pass


def test_launcher_start_attach_and_child_survives_parent(tmp_path: Path) -> None:
    root = _root(tmp_path)
    worker = _fake_worker(root)
    payload_path, payload = _payload(root)
    manifest = _launch_manifest(root, worker, payload_path, payload)
    first = None
    pid = 0
    try:
        first = _run_launcher(manifest)
        assert first.returncode == 0, first.stderr
        started = json.loads(first.stdout)
        assert started["status"] == "STARTED"
        pid = started["pid"]
        owner = json.loads((root / "worker-owner.json").read_text())
        assert owner["pid"] == pid
        assert owner["command_sha256"] == started["command_sha256"]
        second = _run_launcher(manifest)
        assert second.returncode == 0, second.stderr
        attached = json.loads(second.stdout)
        assert attached["status"] == "ATTACHED"
        assert attached["pid"] == pid
        assert json.loads((root / "worker-owner.json").read_text())["pid"] == pid
    finally:
        _kill(pid)


def test_launcher_refuses_foreign_owner_and_never_replaces(tmp_path: Path) -> None:
    root = _root(tmp_path)
    worker = _fake_worker(root)
    payload_path, payload = _payload(root)
    manifest = _launch_manifest(root, worker, payload_path, payload)
    foreign = {
        "schema": "anime-video-studio/p1-v3-process-owner/v1",
        "kind": "download_worker",
        "run_id": "other-run",
        "nonce": "other-nonce",
        "command_sha256": "f" * 64,
        "pid": 999999,
    }
    (root / "worker-owner.json").write_text(json.dumps(foreign), encoding="utf-8")
    result = _run_launcher(manifest)
    assert result.returncode != 0
    assert json.loads((root / "worker-owner.json").read_text()) == foreign
    assert not (root / "worker-ack.json").exists()


def test_launcher_rejects_symlink_or_nonregular_paths_before_spawn(tmp_path: Path) -> None:
    root = _root(tmp_path)
    worker = _fake_worker(root)
    payload_path, payload = _payload(root)
    manifest = _launch_manifest(root, worker, payload_path, payload)
    outside = tmp_path / "sentinel.json"
    outside.write_text("sentinel-bytes", encoding="utf-8")
    try:
        os.symlink(outside, root / "download.log")
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this host")
    result = _run_launcher(manifest)
    assert result.returncode != 0
    assert outside.read_text(encoding="utf-8") == "sentinel-bytes"
    assert not (root / "worker-owner.json").exists()

    (root / "download.log").unlink()
    worker.unlink()
    worker.write_text("not the real worker", encoding="utf-8")
    payload["deadline_epoch"] = time.time() - 1
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = _launch_manifest(root, worker, payload_path, payload)
    result = _run_launcher(manifest)
    assert result.returncode != 0
    assert not (root / "worker-owner.json").exists()


def test_launcher_expired_deadline_never_spawns(tmp_path: Path) -> None:
    root = _root(tmp_path)
    worker = _fake_worker(root)
    payload_path, payload = _payload(root, deadline=time.time() - 1)
    manifest = _launch_manifest(root, worker, payload_path, payload)
    result = _run_launcher(manifest)
    assert result.returncode != 0
    assert not (root / "worker-owner.json").exists()


class _LocalRunner:
    """Minimal local command transport for the production async protocol."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.commands: list[tuple[str, ...]] = []
        self.attached: set[str] = set()

    def put_file(self, identity, remote_path, local_path, **kwargs):
        destination = self.root / Path(remote_path).name
        destination.write_bytes(Path(local_path).read_bytes())

    def run_remote(self, identity, args, timeout=None, **kwargs):
        from types import SimpleNamespace

        self.commands.append(tuple(args))
        if args[0] == "mkdir":
            return SimpleNamespace(returncode=0, stdout="", timed_out=False)
        if args[0] == "df":
            return SimpleNamespace(
                returncode=0,
                stdout="Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/x 100 1 99 1% /workspace\n",
                timed_out=False,
            )
        if args[0] == "python3" and "v3_remote_launcher.py" in args[1]:
            manifest = json.loads((self.root / "launch.json").read_text())
            owner = {
                "schema": "anime-video-studio/p1-v3-process-owner/v1",
                "kind": "download_worker",
                "run_id": manifest["run_id"],
                "nonce": manifest["nonce"],
                "command_sha256": v3_remote_launcher.manifest_hash(manifest),
                "pid": 4242,
            }
            (self.root / "worker-owner.json").write_text(json.dumps(owner), encoding="utf-8")
            (self.root / "worker-ack.json").write_text(json.dumps(owner), encoding="utf-8")
            (self.root / "status.json").write_text(
                json.dumps({"run_id": manifest["run_id"], "nonce": manifest["nonce"], "status": "COMPLETE"}),
                encoding="utf-8",
            )
            (self.root / "progress.json").write_text(
                json.dumps(
                    {
                        "run_id": manifest["run_id"],
                        "nonce": manifest["nonce"],
                        "status": "COMPLETE",
                        "files_done": 3,
                        "files_total": 3,
                        "bytes_done": 10,
                        "bytes_total": 10,
                    }
                ),
                encoding="utf-8",
            )
            (self.root / "heartbeat.json").write_text(
                json.dumps(
                    {
                        "run_id": manifest["run_id"],
                        "nonce": manifest["nonce"],
                        "status": "COMPLETE",
                        "heartbeat_at": time.time(),
                    }
                ),
                encoding="utf-8",
            )
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "schema": "anime-video-studio/p1-v3-launch-result/v1",
                        "kind": "download_worker",
                        "run_id": manifest["run_id"],
                        "nonce": manifest["nonce"],
                        "status": "STARTED",
                        "command_sha256": v3_remote_launcher.manifest_hash(manifest),
                        "pid": 4242,
                    }
                ),
                timed_out=False,
            )
        return SimpleNamespace(returncode=0, stdout="", timed_out=False)

    def read_remote_exact_json(self, identity, *, helper_path, remote_path, root, max_bytes, timeout, **kwargs):
        name = Path(remote_path).name
        return json.loads((self.root / name).read_text())


def test_async_protocol_attach_complete_does_not_start_second_worker(tmp_path: Path, monkeypatch) -> None:
    root = _root(tmp_path)
    # Stage by hand: launch.json contains local paths, but the local runner maps
    # every staged file into the fake remote root by basename.
    payload_path, payload = _payload(root)
    local_worker = root / "v3_download_worker.py"
    local_manifest = root / "manifest.json"
    runner = _LocalRunner(root)
    identity = type("Identity", (), {})()
    protocol = v3_lifecycle.RemoteDownloadWorker(
        runner=runner,
        identity=identity,
        remote_root="/workspace/p1-run",
        run_id=payload["run_id"],
        nonce=payload["nonce"],
        local_worker=local_worker,
        local_setup_plan=ROOT / "scripts" / "p1" / "setup_plan.py",
        local_manifest=local_manifest,
    )
    staged = protocol.stage(
        payload=payload,
        manifest_sha256=payload["manifest_sha256"],
        required_bytes=payload["required_bytes"],
        timeout=10,
    )
    started = protocol.start(launch_manifest=staged["launch_manifest"], timeout=10)
    assert started["status"] in {"STARTED", "ATTACHED"}
    again = protocol.start(launch_manifest=staged["launch_manifest"], timeout=10)
    assert again["status"] == "ATTACHED_COMPLETE"
    launch_calls = [c for c in runner.commands if "v3_remote_launcher.py" in c[1]]
    assert len(launch_calls) == 1


def test_real_worker_completes_via_launcher(tmp_path: Path) -> None:
    root = _root(tmp_path)
    payload_path, payload = _payload(root)
    worker = root / "v3_download_worker.py"
    manifest = _launch_manifest(root, worker, payload_path, payload)
    result = _run_launcher(manifest)
    assert result.returncode == 0, result.stderr
    pid = json.loads(result.stdout)["pid"]
    try:
        deadline = time.time() + 15
        status = {}
        while time.time() < deadline:
            try:
                status = json.loads((root / "status.json").read_text()) if (root / "status.json").is_file() else {}
            except (OSError, json.JSONDecodeError):
                time.sleep(0.05)
                continue
            if status.get("status") in {"COMPLETE", "FAILED"}:
                break
            time.sleep(0.1)

        assert status.get("status") == "COMPLETE"
        progress = {}
        heartbeat = {}
        for _ in range(100):
            try:
                progress = json.loads((root / "progress.json").read_text())
                heartbeat = json.loads((root / "heartbeat.json").read_text())
                break
            except (OSError, json.JSONDecodeError):
                time.sleep(0.05)
        assert progress["files_done"] == 3
        assert progress["bytes_done"] == progress["bytes_total"]
        assert heartbeat["run_id"] == payload["run_id"]

        deadline = time.time() + 5
        while time.time() < deadline and (root / ".p1-download.lock").exists():
            time.sleep(0.05)
        assert not (root / ".p1-download.lock").exists()
    finally:
        _kill(pid)

def _failing_worker(root: Path) -> Path:
    worker = root / "v3_download_worker.py"
    worker.write_text(
        "import json,sys,time,pathlib,os\n"
        "payload=json.loads(pathlib.Path(sys.argv[1]).read_text())\n"
        "base={'run_id':payload['run_id'],'nonce':payload['nonce'],'status':'FAILED','last_error':'synthetic'}\n"
        "pathlib.Path(payload['status_path']).write_text(json.dumps(base))\n"
        "pathlib.Path(payload['progress_path']).write_text(json.dumps({**base,'files_done':0,'files_total':3,'bytes_done':0,'bytes_total':10}))\n"
        "pathlib.Path(payload['heartbeat_path']).write_text(json.dumps({**base,'pid':os.getpid()}))\n",
        encoding="utf-8",
    )
    return worker


def test_known_dead_requires_explicit_resume_and_archives_owner(tmp_path: Path) -> None:
    root = _root(tmp_path)
    worker = _failing_worker(root)
    payload_path, payload = _payload(root)
    manifest = _launch_manifest(root, worker, payload_path, payload)
    first = _run_launcher(manifest)
    assert first.returncode == 0, first.stderr
    first_pid = json.loads(first.stdout)["pid"]
    deadline = time.time() + 10
    while time.time() < deadline:
        if not v3_remote_launcher._pid_alive(first_pid):
            break
        time.sleep(0.05)
    assert not v3_remote_launcher._pid_alive(first_pid)

    refused = _run_launcher(manifest)
    assert refused.returncode != 0
    assert "dead-resume" in refused.stderr
    assert json.loads((root / "worker-owner.json").read_text())["pid"] == first_pid

    resumed = _run_launcher(manifest, "--resume-dead")
    assert resumed.returncode == 0, resumed.stderr
    new_owner = json.loads((root / "worker-owner.json").read_text())
    assert new_owner["pid"] != first_pid
    assert list(root.glob("worker-owner.json.dead-*"))
    _kill(new_owner["pid"])


def _resume_once_worker(root: Path) -> Path:
    """Fail on the first attempt, stay alive on an explicit dead resume."""

    worker = root / "v3_download_worker.py"
    worker.write_text(
        "import json,os,sys,time,pathlib\n"
        "payload=json.loads(pathlib.Path(sys.argv[1]).read_text())\n"
        "marker=pathlib.Path(payload['workspace'])/'attempts.txt'\n"
        "count=int(marker.read_text()) if marker.exists() else 0\n"
        "marker.write_text(str(count+1))\n"
        "base={'run_id':payload['run_id'],'nonce':payload['nonce']}\n"
        "if not os.environ.get('P1_RESUME_DEAD'):\n"
        "    failed={**base,'status':'FAILED','last_error':'synthetic-first'}\n"
        "    pathlib.Path(payload['status_path']).write_text(json.dumps(failed))\n"
        "    pathlib.Path(payload['progress_path']).write_text(json.dumps({**failed,'files_done':0,'files_total':3,'bytes_done':0,'bytes_total':10}))\n"
        "    pathlib.Path(payload['heartbeat_path']).write_text(json.dumps({**failed,'pid':os.getpid()}))\n"
        "    raise SystemExit(0)\n"
        "running={**base,'status':'RUNNING'}\n"
        "pathlib.Path(payload['status_path']).write_text(json.dumps(running))\n"
        "pathlib.Path(payload['progress_path']).write_text(json.dumps({**running,'files_done':0,'files_total':3,'bytes_done':0,'bytes_total':10}))\n"
        "pathlib.Path(payload['heartbeat_path']).write_text(json.dumps({**running,'pid':os.getpid()}))\n"
        "time.sleep(8.0)\n"
        "done={**base,'status':'COMPLETE'}\n"
        "pathlib.Path(payload['status_path']).write_text(json.dumps(done))\n",
        encoding="utf-8",
    )
    return worker


def test_concurrent_dead_resume_spawns_exactly_one_owner(tmp_path: Path) -> None:
    root = _root(tmp_path)
    worker = _resume_once_worker(root)
    payload_path, payload = _payload(root)
    manifest = _launch_manifest(root, worker, payload_path, payload)
    first = _run_launcher(manifest)
    assert first.returncode == 0, first.stderr
    first_pid = json.loads(first.stdout)["pid"]
    deadline = time.time() + 10
    while time.time() < deadline and v3_remote_launcher._pid_alive(first_pid):
        time.sleep(0.05)
    assert not v3_remote_launcher._pid_alive(first_pid)
    assert json.loads((root / "status.json").read_text())["status"] == "FAILED"
    owner_before = json.loads((root / "worker-owner.json").read_text())
    assert owner_before["pid"] == first_pid

    command = [
        sys.executable,
        str(root / "v3_remote_launcher.py"),
        "--launch",
        str(manifest),
        "--resume-dead",
    ]
    one = subprocess.Popen(
        command, cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    two = subprocess.Popen(
        command, cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    out1, err1 = one.communicate(timeout=30)
    out2, err2 = two.communicate(timeout=30)
    results = []
    for code, out, err in ((one.returncode, out1, err1), (two.returncode, out2, err2)):
        if code == 0:
            results.append(json.loads(out))
        else:
            assert "launch lock" not in err
    try:
        assert results
        assert all(item["status"] in {"STARTED", "ATTACHED"} for item in results)
        owner = json.loads((root / "worker-owner.json").read_text())
        assert owner["pid"] != first_pid
        assert v3_remote_launcher._pid_alive(owner["pid"])
        assert len(list(root.glob("worker-owner.json.dead-*"))) == 1
        deadline = time.time() + 10
        attempts = 0
        while time.time() < deadline:
            marker = root / "attempts.txt"
            attempts = int(marker.read_text()) if marker.is_file() else 0
            if attempts >= 2:
                break
            time.sleep(0.05)
        # One failed initial attempt plus exactly one resumed child.
        assert attempts == 2
        started = [item for item in results if item["status"] == "STARTED"]
        assert len(started) == 1
        assert owner["pid"] == started[0]["pid"]
    finally:
        try:
            owner = json.loads((root / "worker-owner.json").read_text())
            _kill(owner["pid"])
        except (OSError, json.JSONDecodeError):
            pass


def test_unknown_partial_state_without_owner_fails_closed(tmp_path: Path) -> None:
    root = _root(tmp_path)
    worker = _fake_worker(root)
    payload_path, payload = _payload(root)
    manifest = _launch_manifest(root, worker, payload_path, payload)
    (root / "status.json").write_text(
        json.dumps({"run_id": payload["run_id"], "nonce": payload["nonce"], "status": "RUNNING"}),
        encoding="utf-8",
    )
    runner = _LocalRunner(root)
    identity = type("Identity", (), {})()
    protocol = v3_lifecycle.RemoteDownloadWorker(
        runner=runner,
        identity=identity,
        remote_root="/workspace/p1-run",
        run_id=payload["run_id"],
        nonce=payload["nonce"],
        local_worker=worker,
        local_setup_plan=ROOT / "scripts" / "p1" / "setup_plan.py",
        local_manifest=root / "manifest.json",
    )
    staged = protocol.stage(
        payload=payload,
        manifest_sha256=payload["manifest_sha256"],
        required_bytes=payload["required_bytes"],
        timeout=10,
    )
    with pytest.raises(v3_lifecycle.LifecycleError):
        protocol.start(launch_manifest=staged["launch_manifest"], timeout=10)
    launch_calls = [c for c in runner.commands if "v3_remote_launcher.py" in c[1]]
    assert launch_calls == []

def test_dead_resume_does_not_mutate_immutable_job_payload(tmp_path: Path) -> None:
    root = _root(tmp_path)
    payload_path, payload = _payload(root)
    runner = _LocalRunner(root)
    identity = type("Identity", (), {})()
    protocol = v3_lifecycle.RemoteDownloadWorker(
        runner=runner,
        identity=identity,
        remote_root="/workspace/p1-run",
        run_id=payload["run_id"],
        nonce=payload["nonce"],
        local_worker=root / "v3_download_worker.py",
        local_setup_plan=ROOT / "scripts" / "p1" / "setup_plan.py",
        local_manifest=root / "manifest.json",
    )
    protocol.stage(
        payload=payload,
        manifest_sha256=payload["manifest_sha256"],
        required_bytes=payload["required_bytes"],
        timeout=10,
    )
    first = (root / "job.json").read_bytes()
    protocol.stage(
        payload=payload,
        manifest_sha256=payload["manifest_sha256"],
        required_bytes=payload["required_bytes"],
        timeout=10,
        resume_dead=True,
    )
    second = (root / "job.json").read_bytes()
    assert first == second
    assert b"resume_dead" not in first


def test_comfy_launcher_actual_generated_paths_and_fake_api(tmp_path: Path) -> None:
    root = tmp_path / "owned"
    comfy = root / "ComfyUI"
    venv = root / "p1-venv" / "bin"
    comfy.mkdir(parents=True)
    venv.mkdir(parents=True)
    python = venv / "python"
    try:
        os.symlink(sys.executable, python)
    except (OSError, NotImplementedError):
        pytest.skip("Python symlink creation not permitted on this host")

    main = comfy / "main.py"
    main.write_text(
        "import argparse, json\n"
        "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
        "p=argparse.ArgumentParser(); p.add_argument('--listen'); p.add_argument('--port',type=int); a=p.parse_args()\n"
        "class H(BaseHTTPRequestHandler):\n"
        "    def log_message(self,*args): return\n"
        "    def do_GET(self):\n"
        "        data=json.dumps({'system':{'comfyui_version':'fixture'}}).encode()\n"
        "        self.send_response(200); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)\n"
        "HTTPServer((a.listen,a.port),H).serve_forever()\n",
        encoding="utf-8",
    )
    port = _free_port()
    manifest = {
        "schema": "anime-video-studio/p1-v3-launch/v1",
        "kind": "comfy",
        "run_id": "run-comfy",
        "nonce": "nonce-comfy",
        "root": str(root),
        "python": str(python),
        "comfy_dir": str(comfy),
        "main": str(main),
        "port": port,
        "log": str(root / "comfy.log"),
        "owner": str(root / "comfy-owner.json"),
        "ack": str(root / "comfy-ack.json"),
    }
    manifest_path = root / "comfy-launch.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "p1" / "v3_remote_launcher.py"), "--launch", str(manifest_path)],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    pid = json.loads(result.stdout)["pid"]
    try:
        import urllib.request

        deadline = time.time() + 10
        data = None
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=1) as response:
                    data = json.loads(response.read())
                break
            except Exception:
                time.sleep(0.1)
        assert data["system"]["comfyui_version"] == "fixture"
        owner = json.loads((root / "comfy-owner.json").read_text())
        ack = json.loads((root / "comfy-ack.json").read_text())
        assert owner["pid"] == pid
        assert owner["command_sha256"] == ack["command_sha256"]
    finally:
        _kill(pid)


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])

def test_concurrent_launchers_never_spawn_two_registered_children(tmp_path: Path) -> None:
    root = _root(tmp_path)
    worker = _fake_worker(root)
    payload_path, payload = _payload(root)
    manifest = _launch_manifest(root, worker, payload_path, payload)
    command = [sys.executable, str(root / "v3_remote_launcher.py"), "--launch", str(manifest)]
    first = subprocess.Popen(command, cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    second = subprocess.Popen(command, cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out1, err1 = first.communicate(timeout=15)
    out2, err2 = second.communicate(timeout=15)
    results = []
    for code, out, err in ((first.returncode, out1, err1), (second.returncode, out2, err2)):
        if code == 0:
            results.append(json.loads(out))
    try:
        assert results
        assert all(item["status"] in {"STARTED", "ATTACHED"} for item in results)
        owner = json.loads((root / "worker-owner.json").read_text())
        assert owner["pid"] in {item["pid"] for item in results}
        assert len(list(root.glob("worker-owner.json"))) == 1
    finally:
        _kill(json.loads((root / "worker-owner.json").read_text())["pid"])

def test_dead_resume_payload_is_identical_through_receiver(tmp_path: Path) -> None:
    from scripts.p1 import remote as remote_module

    root = _root(tmp_path)
    payload_path, payload = _payload(root)
    receive_root = tmp_path / "receiver-root"
    receive_root.mkdir()

    class ReceiverRunner:
        def put_file(self, identity, remote_path, local_path, **kwargs):
            relative = remote_path.removeprefix("/workspace/")
            dest = receive_root / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = Path(local_path).read_bytes()
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    remote_module._REMOTE_RECEIVER,
                    str(dest),
                    str(receive_root),
                    hashlib.sha256(data).hexdigest(),
                    "same",
                ],
                input=data,
                capture_output=True,
                timeout=10,
            )
            assert result.returncode == 0, result.stderr

        def run_remote(self, identity, args, timeout=None, **kwargs):
            if args[0] == "df":
                return type(
                    "Result",
                    (),
                    {
                        "returncode": 0,
                        "stdout": "Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/x 1000000 1 999999 1% /workspace\n",
                        "timed_out": False,
                    },
                )()
            return type("Result", (), {"returncode": 0, "stdout": "", "timed_out": False})()


        def read_remote_exact_json(self, identity, *, helper_path, remote_path, root, max_bytes, timeout, **kwargs):
            relative = remote_path.removeprefix("/workspace/")
            return json.loads((receive_root / relative).read_text(encoding="utf-8"))

    protocol = v3_lifecycle.RemoteDownloadWorker(
        runner=ReceiverRunner(),
        identity=type("Identity", (), {})(),
        remote_root="/workspace",
        run_id=payload["run_id"],
        nonce=payload["nonce"],
        local_worker=root / "v3_download_worker.py",
        local_setup_plan=ROOT / "scripts" / "p1" / "setup_plan.py",
        local_manifest=root / "manifest.json",
    )
    protocol.stage(
        payload=payload,
        manifest_sha256=payload["manifest_sha256"],
        required_bytes=payload["required_bytes"],
        timeout=10,
    )
    first = (receive_root / "job.json").read_bytes()
    protocol.stage(
        payload=payload,
        manifest_sha256=payload["manifest_sha256"],
        required_bytes=payload["required_bytes"],
        timeout=10,
        resume_dead=True,
    )
    second = (receive_root / "job.json").read_bytes()
    assert first == second
    assert "resume_dead" not in (receive_root / "job.json").read_text()
