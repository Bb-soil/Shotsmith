"""Trusted read-only remote snapshot source (executed via python3 -c).

This file is not imported by the local application.  ``observation.py`` reads
its text and sends it as a single, fixed Python literal.  It must stay
standard-library-only and strictly read-only.
"""

import base64
import hashlib
import json
import os
import selectors
import shutil
import signal
import subprocess
import sys
import time
import zlib

sys.dont_write_bytecode = True

(
    marker,
    ws,
    comfy,
    commit,
    root,
    attempt,
    run_id,
    nonce,
    guard_sha,
    targets_json,
    validator,
    mode,
    guard_b64,
    expected_job_sha256,
    expected_launch_sha256,
    collector_budget,
    expected_deadline_epoch,
) = sys.argv[1:18]
targets = json.loads(targets_json)
DEADLINE = time.monotonic() + max(0.1, min(6.0, float(collector_budget)))
CLEANUP_RESERVE = min(1.0, max(0.5, float(collector_budget) * 0.15))
WORK_DEADLINE = DEADLINE - CLEANUP_RESERVE
budget_exhausted = False
partial = False


def key(argv):
    return json.dumps([str(item) for item in argv], ensure_ascii=False, separators=(",", ":"))


def _set_nonblocking(fd):
    try:
        os.set_blocking(fd, False)
    except (AttributeError, OSError):
        pass


def run_subprocess(
    argv,
    *,
    work_deadline,
    cleanup_deadline,
    input_bytes=None,
    max_bytes=65536,
):
    global partial, budget_exhausted
    started = time.monotonic()
    if started >= work_deadline:
        budget_exhausted = True
        partial = True
        return {
            "exit": 124,
            "out": "",
            "err": "budget_expired",
            "timed_out": True,
            "unrun": True,
            "overflow": False,
            "elapsed": 0.0,
        }
    try:
        process = subprocess.Popen(
            [str(item) for item in argv],
            stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        return {
            "exit": 127,
            "out": "",
            "err": type(exc).__name__,
            "timed_out": False,
            "unrun": False,
            "overflow": False,
            "elapsed": time.monotonic() - started,
        }
    pgid = process.pid
    stdout_fd = process.stdout.fileno()
    stderr_fd = process.stderr.fileno()
    stdin_fd = process.stdin.fileno() if input_bytes is not None else None
    for fd in (stdin_fd, stdout_fd, stderr_fd):
        if fd is not None:
            _set_nonblocking(fd)
    selector = selectors.DefaultSelector()
    if stdin_fd is not None:
        selector.register(stdin_fd, selectors.EVENT_WRITE, "stdin")
    selector.register(stdout_fd, selectors.EVENT_READ, "stdout")
    selector.register(stderr_fd, selectors.EVENT_READ, "stderr")
    out = bytearray()
    err = bytearray()
    offset = 0
    stdin_open = stdin_fd is not None
    timed_out = False
    overflow = False
    killed = False

    def kill_group():
        nonlocal killed
        if killed:
            return
        killed = True
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass

    def close_stdin():
        nonlocal stdin_open, stdin_fd
        if not stdin_open:
            return
        stdin_open = False
        try:
            selector.unregister(stdin_fd)
        except (KeyError, ValueError):
            pass
        try:
            process.stdin.close()
        except (OSError, ValueError):
            pass
        stdin_fd = None

    def close_fd(fd):
        try:
            selector.unregister(fd)
        except (KeyError, ValueError):
            pass
        try:
            os.close(fd)
        except OSError:
            pass

    def drain_events(events, cap):
        nonlocal overflow
        for key, _mask in events:
            if key.data == "stdin":
                continue
            try:
                chunk = os.read(key.fd, 65536)
            except BlockingIOError:
                continue
            except OSError:
                close_fd(key.fd)
                continue
            if not chunk:
                close_fd(key.fd)
                continue
            target = out if key.data == "stdout" else err
            room = max(0, cap - len(target))
            if len(chunk) > room:
                overflow = True
                target.extend(chunk[:room])
                kill_group()
            else:
                target.extend(chunk)

    while True:
        now = time.monotonic()
        if now >= work_deadline:
            timed_out = True
            partial = True
            budget_exhausted = True
            kill_group()
            break
        if stdin_open and offset >= len(input_bytes or b""):
            close_stdin()
        try:
            events = selector.select(timeout=max(0.0, min(0.05, work_deadline - now)))
        except OSError:
            events = []
        drain_events(events, max_bytes)
        if overflow:
            timed_out = True
            partial = True
            budget_exhausted = True
            break
        if stdin_open and events:
            for key, mask in events:
                if key.data != "stdin" or not (mask & selectors.EVENT_WRITE):
                    continue
                try:
                    written = os.write(
                        stdin_fd,
                        (input_bytes or b"")[offset : offset + 65536],
                    )
                    offset += written
                except BlockingIOError:
                    pass
                except OSError:
                    close_stdin()
        if not selector.get_map() and process.poll() is not None:
            break
    if timed_out or overflow or process.poll() is None:
        kill_group()
        while time.monotonic() < cleanup_deadline and selector.get_map():
            try:
                events = selector.select(
                    timeout=max(0.0, min(0.05, cleanup_deadline - time.monotonic()))
                )
            except OSError:
                events = []
            drain_events(events, max_bytes)
    close_stdin()
    for fd in (stdout_fd, stderr_fd):
        close_fd(fd)
    if process.poll() is None:
        remaining = max(0.0, min(0.2, cleanup_deadline - time.monotonic()))
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            pass
    if timed_out or overflow:
        exit_code = 124
    else:
        polled = process.poll()
        if polled is None:
            exit_code = 124
            timed_out = True
            partial = True
            budget_exhausted = True
        elif polled < 0:
            exit_code = min(255, 128 + (-polled))
        else:
            exit_code = min(255, polled)
    err_text = err.decode("utf-8", "replace")[:max_bytes]
    if not err_text and timed_out:
        err_text = "timeout"
    elif not err_text and overflow:
        err_text = "output_limit"
    return {
        "exit": exit_code,
        "out": out.decode("utf-8", "replace")[:max_bytes],
        "err": err_text,
        "timed_out": timed_out,
        "unrun": False,
        "overflow": overflow,
        "elapsed": time.monotonic() - started,
    }


def run(argv, timeout=10.0):
    result = run_subprocess(
        argv,
        work_deadline=min(WORK_DEADLINE, time.monotonic() + float(timeout)),
        cleanup_deadline=DEADLINE,
        max_bytes=65536,
    )
    return result


commands = []


def add(argv, timeout=10.0):
    result = run(argv, timeout)
    commands.append({"key": key(argv), **result})
    return result


OBSERVER_CHILD = r'''
import base64,hashlib,json,os,sys,time,zlib
sys.dont_write_bytecode=True
payload=json.loads(sys.stdin.read())
raw=zlib.decompress(base64.b64decode(payload["guard_b64"]))
if hashlib.sha256(raw).hexdigest()!=payload["guard_sha"]:
    raise SystemExit(3)
namespace={"__name__":"i1_embedded_guard","__file__":"<embedded>"}
exec(compile(raw,"<embedded-process-guard>","exec"),namespace)
observe=namespace["observe_download"]
decision=observe(
    root=payload["root"],
    attempt_dir=payload["attempt"],
    attempt_id=payload["attempt_id"],
    run_id=payload["run_id"],
    nonce=payload["nonce"],
    expected_job_sha256=payload["expected_job"],
    expected_launch_sha256=payload["expected_launch"],
    expected_deadline_epoch=payload["expected_deadline"],
    deadline=time.monotonic()+max(0.1,float(payload["remaining"])),
)
sys.stdout.write(json.dumps(decision,sort_keys=True))
'''


def run_observer_child(payload, timeout, max_bytes=262144):
    result = run_subprocess(
        [sys.executable, "-c", OBSERVER_CHILD],
        work_deadline=min(WORK_DEADLINE, time.monotonic() + float(timeout)),
        cleanup_deadline=DEADLINE,
        input_bytes=json.dumps(payload).encode("utf-8"),
        max_bytes=max_bytes,
    )
    return (
        result["exit"],
        result["out"].encode("utf-8"),
        result["err"].encode("utf-8"),
        result["timed_out"],
        result["overflow"],
    )
if mode == "environment":
    add(["uname", "-s"])
    add(["uname", "-m"])
    add(["id", "-u"])
    add(["python3", "-c", "import os;print(os.path.expanduser('~'))"])
    for name in ("bash", "git", "python3"):
        add([name, "--version"])
    add(["python3", "-c", "import venv"])
    add(["python3", "-c", __CONTAINMENT_CODE__, ws])
    add(["test", "-w", ws])
    add(["df", "-Pk", ws])
    add(["python3", "-c", __MEM_CODE__])
    add(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    venv = os.path.join(ws, "p1-venv", "bin", "python")
    add(["test", "-x", venv])
    py = venv if os.access(venv, os.X_OK) else "python3"
    add([py, "-c", __TORCH_CODE__], 30.0)
    add(["test", "-d", comfy])
    add(["test", "-d", os.path.join(comfy, ".git")])
    add(["git", "-C", comfy, "rev-parse", "HEAD"])
    add(["git", "-C", comfy, "status", "--porcelain", "--untracked-files=no"])
    add(["python3", "-B", "-c", validator, comfy, commit], 60.0)
    add(["test", "-f", os.path.join(comfy, "requirements.txt")])
    add([venv, "-m", "pip", "check"], 60.0)
    for target in targets:
        add(["stat", "-c", "%s", os.path.join(comfy, target)])

tool_presence = {
    "bash": None,
    "git": None,
    "python3": None,
    "nvidia-smi": None,
}
for item in commands:
    try:
        argv = json.loads(item["key"])
    except (TypeError, json.JSONDecodeError):
        continue
    if not isinstance(argv, list) or not argv:
        continue
    name = str(argv[0])
    if name not in tool_presence:
        continue
    if item.get("unrun"):
        continue
    if item.get("exit") == 0:
        tool_presence[name] = True
    elif item.get("exit") == 127:
        tool_presence[name] = False

attempt_data = {"state": "INCONCLUSIVE", "reason": "not_requested"}
if mode == "attempt" and attempt:
    expected_deadline = None
    if expected_deadline_epoch not in ("", None):
        try:
            expected_deadline = float(expected_deadline_epoch)
        except ValueError:
            attempt_data = {
                "state": "INCONCLUSIVE",
                "reason": "expected_deadline_invalid",
            }
            expected_deadline = None
    if attempt_data.get("reason") != "expected_deadline_invalid":
        if not guard_b64:
            attempt_data = {"state": "INCONCLUSIVE", "reason": "guard_source_missing"}
        else:
            remaining = max(0.1, WORK_DEADLINE - time.monotonic())
            payload = {
                "guard_b64": guard_b64,
                "guard_sha": guard_sha,
                "root": root,
                "attempt": attempt,
                "attempt_id": os.path.basename(attempt),
                "run_id": run_id,
                "nonce": nonce,
                "expected_job": expected_job_sha256,
                "expected_launch": expected_launch_sha256,
                "expected_deadline": expected_deadline,
                "remaining": remaining,
            }
            rc, out, err, timed_out, overflow = run_observer_child(
                payload, remaining, 262144
            )
            if timed_out:
                attempt_data = {
                    "state": "INCONCLUSIVE",
                    "reason": "observer_child_timeout",
                }
            elif overflow:
                attempt_data = {
                    "state": "INCONCLUSIVE",
                    "reason": "observer_child_output_too_large",
                }
            elif rc != 0:
                attempt_data = {
                    "state": "INCONCLUSIVE",
                    "reason": "observer_child_failed",
                }
            else:
                try:
                    decision = json.loads(out.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    decision = None
                if not isinstance(decision, dict):
                    attempt_data = {
                        "state": "INCONCLUSIVE",
                        "reason": "observer_child_json",
                    }
                else:
                    attempt_data = {
                        "state": decision.get("state"),
                        "reason": decision.get("reason"),
                        "decision": decision,
                    }

print(
    json.dumps(
        {
            "schema": "anime-video-studio/i1-observation-snapshot/v1",
            "marker": marker,
            "run_id": run_id,
            "nonce": nonce,
            "python3": True,
            "tool_presence": tool_presence,
            "commands": commands,
            "attempt": attempt_data,
            "partial": bool(partial),
            "budget_exhausted": bool(budget_exhausted),
            "collector_budget_seconds": float(collector_budget),
        },
        sort_keys=True,
    )
)
