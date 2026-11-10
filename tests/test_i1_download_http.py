from __future__ import annotations

import hashlib
import http.server
import json
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from scripts.i1 import profile as profile_mod
from scripts.i1 import worker
from scripts.p1 import setup_plan

ROOT = Path(__file__).resolve().parents[1]
SIZE = 2 * 1024 * 1024


class Clock:
    def __init__(self) -> None:
        self.value = time.time()

    def __call__(self) -> float:
        return self.value


class Fixture:
    def __init__(
        self,
        files: dict[str, bytes],
        *,
        delay: float = 0.0,
        slow: set[str] | None = None,
        broken: set[str] | None = None,
        fail_once: set[str] | None = None,
    ) -> None:
        self.files = files
        self.delay = delay
        self.slow = set(slow or ())
        self.broken = set(broken or ())
        self.fail_once = set(fail_once or ())
        self.failed_once: set[str] = set()
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.two_active = threading.Event()
        self.slow_started = threading.Event()
        self.bad_gate = threading.Event()
        self.idle = threading.Event()
        self.idle.set()

    def enter(self) -> None:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.idle.clear()
            if self.active >= 2:
                self.two_active.set()

    def leave(self) -> None:
        with self.lock:
            self.active -= 1
            if self.active == 0:
                self.idle.set()

    def wait_two_active(self, timeout: float = 5.0) -> bool:
        return self.two_active.wait(timeout)

    def wait_idle(self, timeout: float = 10.0) -> bool:
        return self.idle.wait(timeout)

    def handler(self):
        fixture = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                return

            def do_GET(self) -> None:
                name = self.path.split("?", 1)[0].lstrip("/")
                data = fixture.files.get(name)
                if data is None:
                    self.send_error(404)
                    return
                fixture.enter()
                try:
                    with fixture.lock:
                        first_failure = name in fixture.fail_once and name not in fixture.failed_once
                        if first_failure:
                            fixture.failed_once.add(name)
                    if first_failure:
                        self.send_error(503)
                        return
                    if name in fixture.broken:
                        # The test releases this only after the slow sibling has
                        # a non-zero .part on the client side.
                        fixture.bad_gate.wait(10.0)
                    if name in fixture.broken:
                        data = b"X" * len(data)
                    start = 0
                    status = 200
                    if self.headers.get("Range"):
                        start = int(self.headers["Range"].split("=", 1)[1].split("-", 1)[0] or "0")
                        status = 206
                    self.send_response(status)
                    self.send_header("Content-Length", str(len(data) - start))
                    self.send_header("Accept-Ranges", "bytes")
                    if status == 206:
                        self.send_header("Content-Range", f"bytes {start}-{len(data)-1}/{len(data)}")
                    self.end_headers()
                    first = True
                    for offset in range(start, len(data), 32 * 1024):
                        self.wfile.write(data[offset : offset + 32 * 1024])
                        self.wfile.flush()
                        if first and name in fixture.slow:
                            fixture.slow_started.set()
                        first = False
                        if name in fixture.slow and fixture.delay:
                            time.sleep(fixture.delay)
                except (BrokenPipeError, ConnectionResetError):
                    return
                finally:
                    fixture.leave()

        return Handler


def _serve(fixture: Fixture) -> tuple[http.server.ThreadingHTTPServer, int]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), fixture.handler())
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def _profile_and_job(tmp: Path, port: int) -> tuple[Path, profile_mod.Profile, dict[str, bytes]]:
    base = json.loads(
        (ROOT / "config" / "i1" / "wan22-14b-ab.profile.json").read_text(encoding="utf-8")
    )
    base["disk"] = {"install_overhead_bytes": 0, "reserve_bytes": 0}
    payloads: dict[str, bytes] = {}
    for index, item in enumerate(base["models"]):
        name = f"file{index}.bin"
        seed = (f"http-fixture-{index}-".encode() * (SIZE // 16 + 1))[:SIZE]
        payloads[name] = seed
        item["url"] = f"https://bench.invalid/{name}"
        item["target"] = f"models/http/{name}"
        item["size_bytes"] = len(seed)
        item["sha256"] = hashlib.sha256(seed).hexdigest()
    profile = profile_mod.validate_profile(base)
    workspace = tmp / "workspace"
    owned = workspace / ".avs-i1"
    attempt = owned / "attempts" / "0001"
    comfy = workspace / "ComfyUI"
    attempt.mkdir(parents=True)
    comfy.mkdir(parents=True)
    profile_path = attempt / "profile.json"
    profile_path.write_text(
        json.dumps(profile.data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    target_home = tmp / "home"
    target_home.mkdir()
    target_id = "t" * 64
    job = {
        "schema": "anime-video-studio/i1-download-job/v1",
        "run_id": "http-run",
        "nonce": "http-nonce",
        "workspace": str(workspace),
        "owned_root": str(attempt),
        "attempt_dir": str(attempt),
        "comfy_root": str(comfy),
        "profile_path": str(profile_path),
        "profile_sha256": profile.sha256,
        "target_home": str(target_home),
        "target_id": target_id,
        "target_lock_path": str(
            target_home / ".cache" / "anime-video-studio" / "i1" / "targets" / f"{target_id}.lock"
        ),
        "deadline_epoch": time.time() + 60,
        "log": str(attempt / "download.log"),
        "status_path": str(attempt / "status.json"),
        "progress_path": str(attempt / "progress.json"),
        "heartbeat_path": str(attempt / "heartbeat.json"),
        "required_bytes": profile.disk_required_bytes,
    }
    job_path = attempt / "job.json"
    job_path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return job_path, profile, payloads


def _patch_local_http(monkeypatch: pytest.MonkeyPatch, port: int) -> None:
    def local_open(url: str, headers, timeout: float = 60.0):
        name = url.rsplit("/", 1)[-1]
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/{name}", headers=dict(headers), method="GET"
        )
        return urllib.request.urlopen(request, timeout=float(timeout))

    monkeypatch.setattr(setup_plan, "_urllib_open", local_open)


def _status(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _targets(job: dict, profile: profile_mod.Profile) -> list[Path]:
    return [Path(job["comfy_root"]) / item["target"] for item in profile.models]


def test_real_http_deadline_with_two_active_cancels_and_drains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = Fixture({}, delay=0.03, slow={"file0.bin", "file1.bin"})
    server, port = _serve(fixture)
    try:
        job_path, profile, payloads = _profile_and_job(tmp_path, port)
        fixture.files = {name.rsplit("/", 1)[-1]: data for name, data in payloads.items()}
        _patch_local_http(monkeypatch, port)
        clock = Clock()
        job = json.loads(job_path.read_text(encoding="utf-8"))
        job["deadline_epoch"] = clock.value + 60.0
        job_path.write_text(json.dumps(job), encoding="utf-8")
        outcome: dict[str, BaseException] = {}

        def run_worker() -> None:
            try:
                worker.run(job_path, clock=clock)
            except BaseException as exc:
                outcome["error"] = exc

        thread = threading.Thread(target=run_worker, name="test-http-deadline")
        thread.start()
        assert fixture.wait_two_active(5.0)
        active_targets = [
            Path(job["comfy_root"]) / profile.models[index]["target"]
            for index in (0, 1)
        ]
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if all(
                target.with_name(target.name + ".part").exists()
                and target.with_name(target.name + ".part").stat().st_size > 0
                for target in active_targets
            ):
                break
            time.sleep(0.02)
        assert all(
            target.with_name(target.name + ".part").exists()
            and target.with_name(target.name + ".part").stat().st_size > 0
            for target in active_targets
        )
        # Both streams are active and have partial bytes; now advance the
        # injected clock past the already-registered deadline.  The job
        # deadline itself is not patched.
        clock.value += 120.0
        thread.join(10.0)
        assert thread.is_alive() is False
        assert isinstance(outcome.get("error"), worker.WorkerError)
        status = _status(Path(job["status_path"]))
        assert status["status"] == "FAILED"
        assert "deadline" in str(status.get("last_error")).lower()
        for index, target in enumerate(_targets(job, profile)):
            assert not target.exists()
            part = target.with_name(target.name + ".part")
            if part.exists():
                assert 0 < part.stat().st_size < int(profile.models[index]["size_bytes"])
        assert fixture.wait_idle(5.0)
        assert not [t for t in threading.enumerate() if t.name.startswith("i1-download-")]
        actual_lock = Path(job["target_lock_path"])
        assert not actual_lock.with_name(actual_lock.name + ".owner.json").exists()
        lock = worker.TargetLock(
            Path(job["workspace"]), "after-deadline", "nonce", lock_dir=actual_lock,
            target_id_value=str(job["target_id"]),
        ).acquire()
        lock.release()
    finally:
        server.shutdown()
        server.server_close()


def test_real_http_bad_sha_and_slow_sibling_preserve_correct_part_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = Fixture({}, delay=0.05, slow={"file1.bin"}, broken={"file0.bin"})
    server, port = _serve(fixture)
    try:
        job_path, profile, payloads = _profile_and_job(tmp_path, port)
        fixture.files = {name.rsplit("/", 1)[-1]: data for name, data in payloads.items()}
        _patch_local_http(monkeypatch, port)
        job = json.loads(job_path.read_text(encoding="utf-8"))
        outcome: dict[str, BaseException] = {}

        def run_worker() -> None:
            try:
                worker.run(job_path)
            except BaseException as exc:
                outcome["error"] = exc

        thread = threading.Thread(target=run_worker, name="test-http-badsha")
        thread.start()
        comfy = Path(job["comfy_root"])
        good = comfy / profile.models[1]["target"]
        good_part = good.with_name(good.name + ".part")
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if good_part.exists() and good_part.stat().st_size > 0:
                break
            time.sleep(0.02)
        assert good_part.exists() and good_part.stat().st_size > 0
        fixture.bad_gate.set()
        thread.join(15.0)
        assert thread.is_alive() is False
        assert isinstance(outcome.get("error"), setup_plan.DownloadError)
        status = _status(Path(job["status_path"]))
        assert status["status"] == "FAILED"
        bad = comfy / profile.models[0]["target"]
        assert not bad.exists()
        assert not bad.with_name(bad.name + ".part").exists()
        assert not good.exists()
        assert 0 < good_part.stat().st_size < int(profile.models[1]["size_bytes"])
        assert fixture.wait_idle(10.0)
        assert not Path(job["target_lock_path"]).with_name(
            Path(job["target_lock_path"]).name + ".owner.json"
        ).exists()
        assert not [t for t in threading.enumerate() if t.name.startswith("i1-download-")]
    finally:
        server.shutdown()
        server.server_close()


def test_real_http_server_error_then_range_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = bytes(range(256)) * 1024
    fixture = Fixture({"file0.bin": body}, fail_once={"file0.bin"})
    server, port = _serve(fixture)
    try:
        _patch_local_http(monkeypatch, port)
        spec = setup_plan.DownloadSpec(
            model_id="resume",
            url="https://bench.invalid/file0.bin",
            target="models/http/file0.bin",
            size_bytes=len(body),
            sha256=hashlib.sha256(body).hexdigest(),
        )
        root = tmp_path / "ComfyUI"
        target = root / spec.target
        target.parent.mkdir(parents=True)
        part = target.with_name(target.name + ".part")
        part.write_bytes(body[:1000])
        with pytest.raises(setup_plan.DownloadError):
            setup_plan.download_verified(spec, root=root)
        assert part.read_bytes() == body[:1000]
        assert not target.exists()
        result = setup_plan.download_verified(spec, root=root)
        assert result.path.read_bytes() == body
        assert not part.exists()
    finally:
        server.shutdown()
        server.server_close()
