#!/usr/bin/env python3
"""One connected cold-root I1 flow: production download -> Comfy -> tunnel -> probe.

No external SSH/cloud/GPU/weights. A fake SSH executable maps the owned tunnel
to a fake local Comfy HTTP process; model bytes come from tiny local HTTPS.
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
import stat
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _fake_ssh(path: Path) -> None:
    _write(
        path,
        "#!/usr/bin/env python3\n"
        "import socket,sys,threading\n"
        "a=sys.argv[1:]\n"
        "i=a.index('-L'); spec=a[i+1]\n"
        "parts=spec.split(':');lp=parts[1];rp=parts[3]\n"
        "def pipe(src,dst):\n"
        "    try:\n"
        "        while True:\n"
        "            d=src.recv(65536)\n"
        "            if not d: break\n"
        "            dst.sendall(d)\n"
        "    except OSError: pass\n"
        "    finally:\n"
        "        try: src.close()\n"
        "        except OSError: pass\n"
        "        try: dst.close()\n"
        "        except OSError: pass\n"
        "def handle(c):\n"
        "    try: u=socket.create_connection(('127.0.0.1',int(rp)))\n"
        "    except OSError: c.close(); return\n"
        "    threading.Thread(target=pipe,args=(c,u),daemon=True).start()\n"
        "    threading.Thread(target=pipe,args=(u,c),daemon=True).start()\n"
        "s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',int(lp)));s.listen()\n"
        "while True:\n"
        "    c,_=s.accept(); threading.Thread(target=handle,args=(c,),daemon=True).start()\n",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _fake_comfy_main(path: Path, object_info: Path, system_stats: Path) -> None:
    _write(
        path,
        "import json,sys,os\n"
        "from pathlib import Path\n"
        "import sibling_fixture\n"
        "from nested_fixture import helper\n"
        "origin={'sys_path_0':sys.path[0],'sibling_file':sibling_fixture.ORIGIN,"
        "'nested_file':helper.ORIGIN,'main_file':__file__,"
        "'birth_present':any((Path(__file__).resolve().parent.parent/'comfy-process-proofs').glob(f'birth-{os.getpid()}-*.json'))}\n"
        "Path(__file__).resolve().parent.parent.joinpath('comfy-origin.json').write_text(json.dumps(origin),encoding='utf-8')\n"
        "from http.server import BaseHTTPRequestHandler,HTTPServer\n"
        "args=sys.argv[1:]\n"
        "port=int(args[args.index('--port')+1])\n"
        f"OBJECT_INFO={str(object_info)!r}\n"
        f"SYSTEM_STATS={str(system_stats)!r}\n"
        "class H(BaseHTTPRequestHandler):\n"
        "    def log_message(self,*a): pass\n"
        "    def do_GET(self):\n"
        "        p=self.path.split('?',1)[0]\n"
        "        if p=='/object_info': d=open(OBJECT_INFO,'rb').read()\n"
        "        elif p=='/system_stats': d=open(SYSTEM_STATS,'rb').read()\n"
        "        else: self.send_error(404); return\n"
        "        self.send_response(200); self.send_header('Content-Length',str(len(d))); self.end_headers(); self.wfile.write(d)\n"
        "HTTPServer(('127.0.0.1',port),H).serve_forever()\n",
    )


def _recv_exact(sock, size):
    out = b""
    while len(out) < size:
        chunk = sock.recv(size - len(out))
        if not chunk:
            break
        out += chunk
    return out


class LocalRunner:
    def __init__(self, *, workspace: Path, env: dict[str, str]) -> None:
        self.workspace = workspace
        self.env = env

    def put_file(self, identity, remote_path, local_path, **kwargs):
        from scripts.p1.remote import _REMOTE_RECEIVER

        data = Path(local_path).read_bytes()
        expected = hashlib.sha256(data).hexdigest()
        mode = "same" if kwargs.get("allow_same_hash", True) else "noclobber"
        result = subprocess.run(
            [sys.executable, "-c", "exec(" + repr(_REMOTE_RECEIVER) + ")",
             str(remote_path), str(kwargs["root"]), expected, mode],
            input=data, capture_output=True, env=self.env,
        )
        if result.returncode != 0:
            raise RuntimeError(f"receiver refused {remote_path}: {result.returncode}")

    def read_remote_exact_json(self, identity, *, helper_path, remote_path, root, max_bytes, timeout):
        return json.loads(Path(remote_path).read_text(encoding="utf-8"))

    def run_remote(self, identity, argv, timeout=None, **kwargs):
        args = list(argv)
        if args and args[0] == "python3":
            args[0] = sys.executable
        result = subprocess.run(
            args, cwd=str(self.workspace), capture_output=True, text=True,
            timeout=timeout or 120.0, env=self.env,
        )
        return SimpleNamespace(
            returncode=result.returncode, stdout=result.stdout,
            stderr=result.stderr, timed_out=False,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--fast-download", action="store_true")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "tests" / "posix"))
    import i1_posix_harness as ph  # type: ignore
    import i1_worker_harness as wh  # type: ignore
    from scripts.i1 import download, ops, state as state_mod, tunnel
    from scripts.i1.profile import validate_profile

    root = Path(tempfile.mkdtemp(prefix="i1-connected-"))
    run_id = "r" * 32
    nonce = "n" * 32
    server = None
    comfy_pid = None
    workspace = None
    try:
        size = 4096 if args.fast_download else (2 * 1024 * 1024)
        files = {f"file{i}.bin": bytes([70 + i]) * size for i in range(4)}
        server, port = wh._start_server(files)
        cert = root / "cert.pem"
        key = root / "key.pem"
        subprocess.run(
            ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
             "-keyout", str(key), "-out", str(cert), "-days", "1",
             "-subj", "/CN=127.0.0.1", "-addext", "subjectAltName=IP:127.0.0.1"],
            check=True, capture_output=True,
        )
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(str(cert), str(key))
        server.socket = context.wrap_socket(server.socket, server_side=True)

        workspace = root / "workspace"
        workspace.mkdir()
        owned = workspace / ".avs-i1"
        target_home = root / "home"
        target_home.mkdir()
        target_id = state_mod.target_id(
            host="example.test", user="root", port=22, fingerprint="SHA256:x"
        )
        target_lock = target_home / ".cache" / "anime-video-studio" / "i1" / "targets" / f"{target_id}.lock"
        profile_data = json.loads(
            (repo / "config" / "i1" / "wan22-14b-ab.profile.json").read_text(encoding="utf-8")
        )
        profile_data["profile_id"] = "connected-tiny"
        profile_data["disk"]["install_overhead_bytes"] = 0
        profile_data["disk"]["reserve_bytes"] = 0
        for index, item in enumerate(profile_data["models"]):
            name = f"file{index}.bin"
            item["url"] = f"https://127.0.0.1:{port}/{name}"
            item["size_bytes"] = len(files[name])
            item["sha256"] = hashlib.sha256(files[name]).hexdigest()
        profile = validate_profile(profile_data)
        bin_dir = root / "bin"
        bin_dir.mkdir()
        _fake_ssh(bin_dir / "ssh")
        env = dict(os.environ)
        env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
        env["SSL_CERT_FILE"] = str(cert)
        env["PIP_NO_INDEX"] = "1"
        old_path = os.environ.get("PATH", "")
        # OwnedTunnel.start spawns the fake ssh from the harness process itself.
        os.environ["PATH"] = env["PATH"]
        runner = LocalRunner(workspace=workspace, env=env)

        staged = download.stage(
            runner, object(), root=str(owned), workspace=str(workspace),
            profile=profile, run_id=run_id, nonce=nonce,
            deadline_epoch=time.time() + 3600,
            target_home=str(target_home), target_id=target_id,
            target_lock_path=str(target_lock), attempt="0001",
        )
        started = download.start(
            runner, object(), root=str(owned), run_id=run_id, nonce=nonce,
            capture_delay=5.0 if args.fast_download else 0.0,
        )
        expected_start = "COMPLETED" if args.fast_download else "STARTED"
        if started.get("status") != expected_start:
            print(json.dumps({
                "status": "DOWNLOAD_START_FAILED",
                "expected": expected_start,
                "actual": started.get("status"),
            }, ensure_ascii=False))
            return 1
        worker_pid = int(started.get("pid", 0))
        if not wh._wait(
            lambda: download.read(runner, object(), root=str(owned), run_id=run_id, nonce=nonce)["status"]["status"] == "COMPLETE",
            90,
        ):
            print(json.dumps({"status": "DOWNLOAD_NOT_COMPLETE"}, ensure_ascii=False))
            return 1
        attempt = Path(staged["paths"]["attempt"])
        comfy = workspace / "ComfyUI"
        (comfy / ".git").mkdir(exist_ok=True)
        (comfy / "comfy").mkdir(exist_ok=True)
        _write(comfy / "comfy" / "__init__.py", "")
        _write(comfy / "nodes.py", "")
        _write(comfy / "requirements.txt", "")
        _write(comfy / "sibling_fixture.py", "ORIGIN = __file__\n")
        _write(comfy / "nested_fixture" / "__init__.py", "")
        _write(comfy / "nested_fixture" / "helper.py", "ORIGIN = __file__\n")
        venv = workspace / "p1-venv"
        ph._make_fake_venv(venv)
        ph._make_git(bin_dir / "git")
        ph._make_nvidia_smi(bin_dir / "nvidia-smi")
        env["I1_FAKE_COMMIT"] = profile_data["comfyui"]["commit"]
        object_info = workspace / "p1-evidence" / "object_info.json"
        system_stats = workspace / "p1-evidence" / "system_stats.json"
        _write(object_info, json.dumps(ph._object_info(profile_data, repo), indent=2))
        _write(system_stats, json.dumps({"devices": [{"name": "Fake GPU", "vram_total": 1}], "system": {"comfyui_version": "fake"}}))
        _fake_comfy_main(comfy / "main.py", object_info, system_stats)
        api_target = owned / "workflow" / "connected-tiny.api.json"
        api_target.parent.mkdir(parents=True)
        shutil.copy2(repo / "workflows" / "P1" / "wan22-14b-shot02-alley-v1.api.json", api_target)
        record = {
            "schema": state_mod.RECORD_SCHEMA,
            "name": "connected",
            "identity": {
                "host": "example.test", "user": "root", "port": 22,
                "fingerprint": "SHA256:x", "workspace": str(workspace),
                "profile_id": profile.profile_id, "profile_sha256": profile.sha256,
                "target_id": target_id,
            },
            "host_key_type": "ssh-ed25519",
            "auth": {"kind": "agent"},
            "known_hosts": str(root / "known_hosts"),
            "global_known_hosts": str(root / "global-empty"),
        }
        (root / "known_hosts").write_text("", encoding="utf-8")
        (root / "global-empty").write_text("", encoding="utf-8")
        state = state_mod.new_state(
            identity_key_value=state_mod.identity_key(
                host="example.test",
                user="root",
                port=22,
                fingerprint="SHA256:x",
                workspace=str(workspace),
                profile_sha256=profile.sha256,
            ),
            host="example.test", user="root", port=22,
            fingerprint="SHA256:x", workspace=str(workspace),
            profile_id=profile.profile_id, profile_sha256=profile.sha256,
            target_id_value=target_id,
        )
        state["phase"] = "INSTALLED"
        state["run_id"] = run_id
        state["nonce"] = nonce
        state["artifacts"] = {"download": "COMPLETE", "download_attempt": {"id": "0001", "dir": str(attempt)}}
        state["remote_facts"] = {"home": str(target_home)}
        state_dir = root / "state"
        record["known_hosts"] = str(root / "known_hosts")
        record["identity"]["workspace"] = str(workspace)
        ctx = ops.Context(
            state_dir=state_dir, repo_root=repo, profile_path=Path("config/i1/wan22-14b-ab.profile.json"),
            profile=profile, name="connected", record=record, runner=runner,
            identity=SimpleNamespace(
                host="example.test", user="root", port=22,
                identity_file=None, known_hosts=Path(root / "known_hosts"),
                global_known_hosts=Path(root / "global-empty"),
                host_key_algorithms="ssh-ed25519",
                destination="root@example.test",
            ),
            state=state, run_dir=root / "run",
        )
        first = ops._finalize_after_download(
            ctx,
            root=str(owned),
            workspace=str(workspace),
            profile=profile,
            state=state,
        )
        if not first.get("readiness", {}).get("ready"):
            print(json.dumps({"status": "FIRST_NOT_READY", "readiness": first.get("readiness")}, ensure_ascii=False))
            return 1
        origin_path = workspace / "comfy-origin.json"
        origin = json.loads(origin_path.read_text(encoding="utf-8")) if origin_path.is_file() else {}
        expected_comfy = (comfy).resolve()
        if (
            origin.get("sys_path_0") != str(expected_comfy)
            or Path(origin.get("sibling_file", "")).resolve() != (comfy / "sibling_fixture.py").resolve()
            or Path(origin.get("nested_file", "")).resolve() != (comfy / "nested_fixture" / "helper.py").resolve()
            or Path(origin.get("main_file", "")).resolve() != (comfy / "main.py").resolve()
            or not origin.get("birth_present")
        ):
            print(json.dumps({"status": "COMFY_IMPORT_ORIGIN_MISMATCH", "origin": origin}, ensure_ascii=False))
            return 1
        owner = json.loads((workspace / "comfy-owner.json").read_text(encoding="utf-8"))
        comfy_pid = int(owner["pid"])
        if not list((workspace / "comfy-process-proofs").glob("birth-*.json")):
            print(json.dumps({"status": "COMFY_BIRTH_NOT_WRITTEN"}, ensure_ascii=False))
            return 1
        # Production live attach: resume with the owned Comfy still running must
        # not wait on the port or spawn a duplicate.
        live_resume = ops.resume(ctx)
        if not live_resume.get("readiness", {}).get("ready"):
            print(json.dumps({"status": "LIVE_RESUME_NOT_READY", "readiness": live_resume.get("readiness")}, ensure_ascii=False))
            return 1
        if int(json.loads((workspace / "comfy-owner.json").read_text(encoding="utf-8"))["pid"]) != comfy_pid:
            print(json.dumps({"status": "LIVE_RESUME_DUPLICATED_COMFY"}, ensure_ascii=False))
            return 1
        if list((workspace / "comfy-archive").glob("comfy-owner.json.dead-*")):
            print(json.dumps({"status": "LIVE_RESUME_ARCHIVED_LIVE_OWNER"}, ensure_ascii=False))
            return 1
        # Simulate SSH/console loss after STARTED but before any local proof
        # upload: remove every proof file while the process stays alive.
        birth_before = sorted(path.name for path in (workspace / "comfy-process-proofs").glob("birth-*.json"))
        for proof_file in (workspace / "comfy-process-proofs").glob("*.json"):
            if proof_file.name.startswith(("proof-", "current-")):
                proof_file.unlink()
        if sorted(path.name for path in (workspace / "comfy-process-proofs").glob("birth-*.json")) != birth_before:
            print(json.dumps({"status": "COMFY_BIRTH_DID_NOT_SURVIVE_PROOF_REMOVAL"}, ensure_ascii=False))
            return 1
        recovered_proof = ops.resume(ctx)
        if not recovered_proof.get("readiness", {}).get("ready"):
            print(json.dumps({"status": "PROOF_RECOVERY_RESUME_NOT_READY", "readiness": recovered_proof.get("readiness")}, ensure_ascii=False))
            return 1
        if int(json.loads((workspace / "comfy-owner.json").read_text(encoding="utf-8"))["pid"]) != comfy_pid:
            print(json.dumps({"status": "PROOF_RECOVERY_DUPLICATED_COMFY"}, ensure_ascii=False))
            return 1
        if not list((workspace / "comfy-process-proofs").glob("current-*.json")):
            print(json.dumps({"status": "PROOF_NOT_RECOVERED"}, ensure_ascii=False))
            return 1
        ops.disconnect(ctx)
        try:
            opened = ops.open_comfy(ctx, browser=False)
        except Exception as exc:
            log_path = root / "run" / "tunnel.log"
            print(json.dumps({
                "status": "OPEN_FAILED",
                "error": type(exc).__name__,
                "reason": str(exc),
                "tunnel_log": log_path.read_text(encoding="utf-8", errors="replace")[-4000:] if log_path.exists() else "",
                "tunnel_owner": (root / "run" / "tunnel-owner.json").read_text(encoding="utf-8", errors="replace") if (root / "run" / "tunnel-owner.json").exists() else None,
            }, ensure_ascii=False))
            return 1
        if int(json.loads((workspace / "comfy-owner.json").read_text(encoding="utf-8"))["pid"]) != comfy_pid:
            print(json.dumps({"status": "OPEN_DUPLICATED_COMFY", "url": opened.get("url")}, ensure_ascii=False))
            return 1
        ops.disconnect(ctx)
        # A proved-dead owner is archived and replaced by exactly one new spawn.
        os.kill(comfy_pid, signal.SIGKILL)
        time.sleep(0.5)
        resumed = ops.resume(ctx)
        if not resumed.get("readiness", {}).get("ready"):
            print(json.dumps({"status": "COMFY_RESUME_NOT_READY", "readiness": resumed.get("readiness")}, ensure_ascii=False))
            return 1
        old = list((workspace / "comfy-archive").glob("comfy-owner.json.dead-*"))
        if not old:
            print(json.dumps({"status": "COMFY_OLD_OWNER_NOT_RETAINED"}, ensure_ascii=False))
            return 1
        comfy_pid = int(
            json.loads((workspace / "comfy-owner.json").read_text(encoding="utf-8"))["pid"]
        )
        # Re-run the production tunnel adapter across repeated disconnect/resume
        # cycles on the same run/nonce.  Every prior terminal state must be
        # archived by the preflight before the next monitor spawn.
        ops.disconnect(ctx)
        archive_before = len(list((ctx.run_dir / "archive").glob("owned-tunnel-*")))
        for _ in range(5):
            tunnel.ensure(
                identity=ctx.identity,
                run_dir=ctx.run_dir,
                run_id=ctx.state["run_id"],
                nonce=ctx.state["nonce"],
                fingerprint=ctx.state["identity"]["fingerprint"],
                local_port=int(profile.data["runtime"]["local_tunnel_port"]),
                remote_port=int(profile.data["runtime"]["comfy_remote_port"]),
            )
            if not tunnel.stop(
                identity=ctx.identity,
                run_dir=ctx.run_dir,
                run_id=ctx.state["run_id"],
                nonce=ctx.state["nonce"],
                fingerprint=ctx.state["identity"]["fingerprint"],
            ):
                print(json.dumps({"status": "REPEAT_TUNNEL_STOP_FAILED"}, ensure_ascii=False))
                return 1
        archives = list((ctx.run_dir / "archive").glob("owned-tunnel-*"))
        repeat_archives = len(archives) - archive_before
        if repeat_archives != 5:
            print(json.dumps({
                "status": "REPEAT_TUNNEL_ARCHIVE_COUNT_MISMATCH",
                "archives_before": archive_before,
                "archives_after": len(archives),
            }, ensure_ascii=False))
            return 1
        print(json.dumps({
            "status": "OFFLINE_VERIFIED",
            "production_flow": True,
            "comfy_import_origins_ok": True,
            "fast_download_completed": bool(args.fast_download),
            "download_worker_pid": worker_pid,
            "live_resume_same_pid": True,
            "proof_recovered_same_pid": True,
            "disconnect_open_same_pid": True,
            "comfy_killed_and_resumed": True,
            "old_owner_retained": True,
            "repeat_disconnect_resume_cycles": 5,
            "repeat_archive_count": repeat_archives,
            "models": 4,
        }, ensure_ascii=False, sort_keys=True))
        return 0
    finally:
        try:
            os.environ["PATH"] = old_path
        except NameError:
            pass
        try:
            if workspace is not None:
                owner = json.loads(
                    (workspace / "comfy-owner.json").read_text(encoding="utf-8")
                )
                comfy_pid = int(owner["pid"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
        if comfy_pid:
            try:
                os.kill(comfy_pid, signal.SIGKILL)
            except OSError:
                pass
        try:
            tunnel_owner = json.loads(
                (root / "run" / "tunnel-owner.json").read_text(encoding="utf-8")
            )
            for key in ("monitor_pid", "ssh_pid"):
                pid = tunnel_owner.get(key)
                if isinstance(pid, int) and pid > 0:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except OSError:
                        pass
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
        if server is not None:
            try:
                server.shutdown(); server.server_close()
            except Exception:
                pass
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
