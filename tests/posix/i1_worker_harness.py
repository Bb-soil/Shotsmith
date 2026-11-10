#!/usr/bin/env python3
"""Production-client POSIX worker harness with real launcher/worker/receiver.

Uses scripts.i1.download.stage/start/read/recover (not a direct launcher-only
shortcut), actual frozen launcher, actual worker, actual no-clobber receiver,
local HTTPS tiny files, SIGKILL, recovery and resume.
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import os
import shutil
import signal
import ssl
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace


class Handler(http.server.BaseHTTPRequestHandler):
    files: dict[str, bytes] = {}

    def log_message(self, *args):
        pass

    def do_GET(self):
        name = self.path.split("?", 1)[0].lstrip("/")
        data = self.files.get(name)
        if data is None:
            self.send_error(404)
            return
        start = 0
        status = 200
        if self.headers.get("Range"):
            try:
                start = int(self.headers["Range"].split("=", 1)[1].split("-", 1)[0] or "0")
                status = 206
            except Exception:
                self.send_error(416)
                return
        self.send_response(status)
        self.send_header("Content-Length", str(len(data) - start))
        self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{len(data)-1}/{len(data)}")
        self.end_headers()
        try:
            for offset in range(start, len(data), 64 * 1024):
                self.wfile.write(data[offset : min(offset + 64 * 1024, len(data))])
                self.wfile.flush()
                time.sleep(0.01)
        except (BrokenPipeError, ssl.SSLError, ConnectionResetError):
            # Deliberate SIGKILL fault injection closes the TLS socket. Keep the
            # expected diagnostic instead of silently presenting an empty stderr.
            print("expected TLS EOF after simulated worker SIGKILL", file=sys.stderr)
            return


class LocalRunner:
    """Production-client runner mapped to the local POSIX fixture."""

    def __init__(self, *, workspace: Path, env: dict[str, str]) -> None:
        self.workspace = workspace
        self.env = env
        self.commands: list[tuple[str, ...]] = []

    def put_file(self, identity, remote_path, local_path, **kwargs):
        from scripts.p1.remote import _REMOTE_RECEIVER

        data = Path(local_path).read_bytes()
        expected = hashlib.sha256(data).hexdigest()
        mode = "same" if kwargs.get("allow_same_hash", True) else "noclobber"
        root = kwargs.get("root")
        if not root:
            raise AssertionError("put_file requires root")
        code = "exec(" + repr(_REMOTE_RECEIVER) + ")"
        result = subprocess.run(
            [sys.executable, "-c", code, str(remote_path), str(root), expected, mode],
            input=data,
            capture_output=True,
            timeout=kwargs.get("timeout", 60.0),
            env=self.env,
        )
        if result.returncode != 0:
            raise RuntimeError(f"receiver refused {remote_path} with {result.returncode}")

    def read_remote_exact_json(self, identity, *, helper_path, remote_path, root, max_bytes, timeout):
        value = json.loads(Path(remote_path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError("remote JSON must be an object")
        return value

    def run_remote(self, identity, argv, timeout=None, **kwargs):
        args = list(argv)
        if args and args[0] == "python3":
            args[0] = sys.executable
        self.commands.append(tuple(args))
        result = subprocess.run(
            args,
            cwd=str(self.workspace),
            capture_output=True,
            text=True,
            timeout=timeout or 60.0,
            env=self.env,
        )
        return SimpleNamespace(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            timed_out=False,
        )


def _start_server(files: dict[str, bytes]):
    Handler.files = files
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _wait(predicate, timeout=60.0):
    end = time.time() + timeout
    while time.time() < end:
        try:
            if predicate():
                return True
        except Exception:
            pass
        time.sleep(0.1)
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    from scripts.i1 import download
    from scripts.i1.profile import validate_profile

    root = Path(tempfile.mkdtemp(prefix="i1-worker-"))
    server = None
    worker_pid = None
    try:
        files = {f"file{i}.bin": bytes([65 + i]) * (4 * 1024 * 1024) for i in range(4)}
        server, port = _start_server(files)
        cert = root / "cert.pem"
        key = root / "key.pem"
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                "-keyout", str(key), "-out", str(cert), "-days", "1",
                "-subj", "/CN=127.0.0.1", "-addext", "subjectAltName=IP:127.0.0.1",
            ],
            check=True,
            capture_output=True,
        )
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(str(cert), str(key))
        server.socket = context.wrap_socket(server.socket, server_side=True)

        workspace = root / "workspace"
        workspace.mkdir(parents=True)
        owned = workspace / ".avs-i1"
        target_home = root / "home"
        target_home.mkdir()
        target_id = "t" * 64
        target_lock_path = target_home / ".cache" / "anime-video-studio" / "i1" / "targets" / f"{target_id}.lock"
        profile_data = json.loads(
            (repo / "config" / "i1" / "wan22-14b-ab.profile.json").read_text(encoding="utf-8")
        )
        profile_data["profile_id"] = "worker-tiny"
        profile_data["disk"]["install_overhead_bytes"] = 0
        profile_data["disk"]["reserve_bytes"] = 0
        for index, item in enumerate(profile_data["models"]):
            name = f"file{index}.bin"
            item["url"] = f"https://127.0.0.1:{port}/{name}"
            item["size_bytes"] = len(files[name])
            item["sha256"] = hashlib.sha256(files[name]).hexdigest()
        profile = validate_profile(profile_data)
        env = dict(os.environ)
        env["SSL_CERT_FILE"] = str(cert)
        runner = LocalRunner(workspace=workspace, env=env)

        staged = download.stage(
            runner,
            object(),
            root=str(owned),
            workspace=str(workspace),
            profile=profile,
            run_id="r",
            nonce="n",
            deadline_epoch=time.time() + 3600,
            target_home=str(target_home),
            target_id=target_id,
            target_lock_path=str(target_lock_path),
            attempt="0001",
        )
        try:
            started = download.start(
                runner, object(), root=str(owned), run_id="r", nonce="n", attempt="0001"
            )
        except Exception as exc:
            log = Path(staged["paths"]["log"])
            print(json.dumps({"status": "START_EXCEPTION", "error": str(exc), "log": log.read_text(encoding="utf-8", errors="replace")[-4000:] if log.exists() else ""}, ensure_ascii=False))
            raise
        if started.get("status") != "STARTED":
            print(json.dumps({"status": "START_NOT_STARTED", "started": started}, ensure_ascii=False))
            return 1
        worker_pid = int(started["pid"])
        proof_dir = Path(staged["paths"]["proof_dir"])
        if not list(proof_dir.glob("birth-*.json")):
            print(json.dumps({"status": "REMOTE_BIRTH_NOT_WRITTEN"}, ensure_ascii=False))
            return 1
        if not list(proof_dir.glob("current-*.json")):
            print(json.dumps({"status": "REMOTE_PROOF_NOT_WRITTEN"}, ensure_ascii=False))
            return 1
        # Simulate SSH loss after STARTED but before the local client could
        # upload anything: the remote guard must recover from owner/ack + /proc.
        birth_before = sorted(path.name for path in proof_dir.glob("birth-*.json"))
        for proof_file in proof_dir.glob("*.json"):
            if proof_file.name.startswith(("proof-", "current-")):
                proof_file.unlink()
        if sorted(path.name for path in proof_dir.glob("birth-*.json")) != birth_before:
            print(json.dumps({"status": "BIRTH_DID_NOT_SURVIVE_PROOF_REMOVAL"}, ensure_ascii=False))
            return 1
        concurrent = download.start(
            runner, object(), root=str(owned), run_id="r", nonce="n", attempt="0001"
        )
        if concurrent.get("status") != "ATTACHED":
            print(json.dumps({"status": "CONCURRENT_NOT_ATTACHED", "result": concurrent}, ensure_ascii=False))
            return 1
        if int(concurrent.get("pid", -1)) != worker_pid:
            print(json.dumps({"status": "PROOF_RECOVERY_CHANGED_PID"}, ensure_ascii=False))
            return 1
        if not list(proof_dir.glob("current-*.json")):
            print(json.dumps({"status": "PROOF_NOT_RECOVERED"}, ensure_ascii=False))
            return 1
        first_target = workspace / "ComfyUI" / profile_data["models"][0]["target"]
        part = first_target.with_name(first_target.name + ".part")
        if not _wait(lambda: part.exists(), 30):
            print(json.dumps({"status": "PART_NOT_CREATED"}, ensure_ascii=False))
            return 1
        os.kill(worker_pid, signal.SIGKILL)
        time.sleep(0.7)
        recovered = download.recover(
            runner,
            object(),
            root=str(owned),
            run_id="r",
            nonce="n",
            attempt="0001",
            target_lock_path=str(target_lock_path),
            stale_seconds=0.0,
        )
        resumed = download.start(
            runner,
            object(),
            root=str(owned),
            run_id="r",
            nonce="n",
            attempt="0001",
            resume_dead=True,
        )
        if resumed.get("status") not in {"STARTED", "ATTACHED_COMPLETE"}:
            print(json.dumps({"status": "RESUME_BAD", "resumed": resumed}, ensure_ascii=False))
            return 1
        if not _wait(
            lambda: download.read(
                runner, object(), root=str(owned), run_id="r", nonce="n", attempt="0001"
            )["status"]["status"] == "COMPLETE",
            90,
        ):
            print(json.dumps({"status": "RESUME_NOT_COMPLETE"}, ensure_ascii=False))
            return 1
        complete = download.read(
            runner, object(), root=str(owned), run_id="r", nonce="n", attempt="0001"
        )
        if complete["status"]["status"] != "COMPLETE":
            print(json.dumps({"status": "READ_NOT_COMPLETE"}, ensure_ascii=False))
            return 1
        hashes_before = {}
        for item in profile_data["models"]:
            final = workspace / "ComfyUI" / item["target"]
            hashes_before[item["target"]] = _sha(final)
            if hashes_before[item["target"]] != item["sha256"]:
                print(json.dumps({"status": "FINAL_HASH_MISMATCH", "target": item["target"]}, ensure_ascii=False))
                return 1
        # Existing final files must be skipped on a second production client start.
        complete_again = download.start(
            runner, object(), root=str(owned), run_id="r", nonce="n", attempt="0001", resume_dead=True
        )
        if complete_again.get("status") != "ATTACHED_COMPLETE":
            print(json.dumps({"status": "COMPLETE_ATTACH_BAD", "result": complete_again}, ensure_ascii=False))
            return 1
        # Run the actual worker once more to prove download_verified hash-skip.
        skip_env = dict(env)
        skip_env["P1_RESUME_DEAD"] = "1"
        skipped = subprocess.run(
            [sys.executable, str(Path(staged["paths"]["worker"])), str(Path(staged["paths"]["job"]))],
            cwd=str(Path(staged["paths"]["attempt"])),
            capture_output=True,
            text=True,
            timeout=120,
            env=skip_env,
        )
        if skipped.returncode != 0:
            print(json.dumps({"status": "SKIP_WORKER_FAILED", "stderr": skipped.stderr}, ensure_ascii=False))
            return 1
        log_text = (Path(staged["paths"]["attempt"]) / "download.log").read_text(encoding="utf-8")
        if log_text.count("skip verified") < 4:
            print(json.dumps({"status": "FINAL_NOT_SKIPPED"}, ensure_ascii=False))
            return 1
        if any(_sha(workspace / "ComfyUI" / target) != digest for target, digest in hashes_before.items()):
            print(json.dumps({"status": "FINAL_CHANGED_ON_SKIP"}, ensure_ascii=False))
            return 1
        print(
            json.dumps(
                {
                    "status": "OFFLINE_VERIFIED",
                    "worker_pid": worker_pid,
                    "concurrent": "ATTACHED",
                    "recovered": recovered.get("status"),
                    "models": 4,
                    "final_hashes_skipped": True,
                    "production_client": True,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    finally:
        if worker_pid:
            try:
                os.kill(worker_pid, signal.SIGKILL)
            except OSError:
                pass
        if server is not None:
            try:
                server.shutdown()
                server.server_close()
            except Exception:
                pass
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
