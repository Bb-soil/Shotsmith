"""Explicit host-key trust without touching the user's global ssh files."""

from __future__ import annotations

import base64
import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from scripts.p1.remote import RemoteError, run_bounded_process

from .errors import HostTrustError

Runner = Callable[..., Any]
_IS_WINDOWS = os.name == "nt"


@dataclass(frozen=True)
class HostKey:
    key_type: str
    key_base64: str
    fingerprint: str
    line: str


def host_key_name(host: str, port: int) -> str:
    # OpenSSH uses the bare host for the default port and bracket form only for
    # non-standard ports.  I1's dedicated known_hosts must match that convention.
    if int(port) == 22:
        return host
    return f"[{host}]:{int(port)}"


def fingerprint_for_key(key_base64: str) -> str:
    try:
        blob = base64.b64decode(key_base64.encode("ascii"), validate=True)
    except Exception as exc:
        raise HostTrustError("host key is not valid base64") from exc
    digest = base64.b64encode(hashlib.sha256(blob).digest()).decode("ascii").rstrip("=")
    return f"SHA256:{digest}"


def _run(
    argv: Sequence[str],
    *,
    timeout: float,
    max_output_bytes: int,
    runner: Runner | None,
) -> Any:
    if runner is not None:
        return runner(tuple(str(item) for item in argv), timeout=timeout, max_output_bytes=max_output_bytes)
    return run_bounded_process(
        tuple(str(item) for item in argv),
        timeout=timeout,
        max_output_bytes=max_output_bytes,
    )


def parse_keyscan(text: str, *, host: str, port: int) -> list[HostKey]:
    result: list[HostKey] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 3:
            continue
        key_type, key_b64 = fields[1], fields[2]
        if not key_type.startswith(("ssh-", "ecdsa-", "sk-")):
            continue
        fingerprint = fingerprint_for_key(key_b64)
        identity = f"{key_type}:{fingerprint}"
        if identity in seen:
            continue
        seen.add(identity)
        normalized = f"{host_key_name(host, port)} {key_type} {key_b64}"
        result.append(HostKey(key_type, key_b64, fingerprint, normalized))
    if not result:
        raise HostTrustError("ssh-keyscan returned no usable host key")
    return result


def scan_host(
    *,
    host: str,
    port: int,
    keyscan_path: str | None = None,
    timeout: float = 15.0,
    runner: Runner | None = None,
) -> list[HostKey]:
    executable = keyscan_path or shutil.which("ssh-keyscan")
    if not executable:
        raise HostTrustError("ssh-keyscan is unavailable")
    result = _run(
        [executable, "-T", str(max(1, int(timeout))), "-p", str(int(port)), host],
        timeout=timeout,
        max_output_bytes=256 * 1024,
        runner=runner,
    )
    native_timed_out = bool(getattr(result, "timed_out", False))
    if native_timed_out and not _IS_WINDOWS:
        raise HostTrustError("host-key scan timed out")
    stdout = getattr(result, "stdout", b"")
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="replace")
    if native_timed_out or (int(getattr(result, "returncode", 1)) != 0 and not stdout.strip()):
        stderr = getattr(result, "stderr", b"")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        # Windows OpenSSH 9.5 keyscan can fail on the peer's hybrid KEX even
        # when ordinary ssh successfully negotiates curve25519. Use only an
        # already installed WSL scanner for this compatibility error or a
        # bounded native timeout. Never accept partial output from a timeout.
        # This is key discovery, never trust: the caller still checks the
        # accepted fingerprint and refuses changed host keys.
        if _IS_WINDOWS and (native_timed_out or "unsupported KEX method" in str(stderr)):
            fallback_reason = (
                "Windows ssh-keyscan timeout" if native_timed_out
                else "Windows ssh-keyscan KEX incompatibility"
            )
            wsl = shutil.which("wsl.exe")
            if not wsl:
                if native_timed_out:
                    raise HostTrustError("host-key scan timed out; WSL scanner is unavailable")
                raise HostTrustError(
                    "Windows ssh-keyscan: incompatible KEX; update OpenSSH "
                    "or provide WSL with ssh-keyscan. Host reachability is not disproven."
                )
            try:
                fallback = _run(
                    [wsl, "--exec", "ssh-keyscan", "-T",
                     str(max(1, int(timeout) - 2)), "-p", str(int(port)),
                     "-t", "ed25519", host],
                    # WSL startup/cleanup is outside keyscan's own -T budget.
                    # Keep a finite allowance rather than killing it at -T.
                    timeout=2.0 * timeout,
                    max_output_bytes=256 * 1024,
                    runner=runner,
                )
            except (OSError, RemoteError) as exc:
                raise HostTrustError(
                    f"{fallback_reason}; WSL scanner failed "
                    f"({type(exc).__name__}). Host key was not trusted."
                ) from exc
            if getattr(fallback, "timed_out", False):
                raise HostTrustError("WSL host-key scan timed out; host key was not trusted")
            raw = getattr(fallback, "stdout", b"")
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            if int(getattr(fallback, "returncode", 1)) != 0:
                raise HostTrustError(
                    f"{fallback_reason}; WSL scanner failed. "
                    "Host key was not trusted."
                )
            return parse_keyscan(raw, host=host, port=port)
        raise HostTrustError(
            "host-key scan failed; check DNS, SSH port and scanner compatibility"
        )
    return parse_keyscan(stdout, host=host, port=port)


def known_host_fingerprints(
    *,
    known_hosts: Path | str,
    host: str,
    port: int,
    ssh_keygen_path: str | None = None,
    timeout: float = 10.0,
    runner: Runner | None = None,
) -> list[HostKey]:
    executable = ssh_keygen_path or shutil.which("ssh-keygen")
    if not executable:
        raise HostTrustError("ssh-keygen is unavailable")
    path = Path(known_hosts)
    if not path.is_file():
        raise HostTrustError("explicit known_hosts file is missing")
    result = _run(
        [executable, "-F", host_key_name(host, port), "-f", str(path)],
        timeout=timeout,
        max_output_bytes=128 * 1024,
        runner=runner,
    )
    if getattr(result, "timed_out", False):
        raise HostTrustError("host-key lookup timed out")
    stdout = getattr(result, "stdout", b"")
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="replace")
    if int(getattr(result, "returncode", 1)) == 1 or not stdout.strip():
        raise HostTrustError("host key is absent from the explicit known_hosts file")
    if int(getattr(result, "returncode", 1)) != 0:
        raise HostTrustError("host-key lookup failed")
    keys: list[HostKey] = []
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("|1|"):
            continue
        fields = line.split()
        if len(fields) < 3 or not fields[1].startswith(("ssh-", "ecdsa-", "sk-")):
            continue
        try:
            fingerprint = fingerprint_for_key(fields[2])
        except HostTrustError:
            continue
        keys.append(
            HostKey(
                fields[1],
                fields[2],
                fingerprint,
                f"{host_key_name(host, port)} {fields[1]} {fields[2]}",
            )
        )
    if not keys:
        raise HostTrustError("known_hosts entry has no usable host key")
    return keys


def ensure_known_hosts(path: Path | str) -> Path:
    target = Path(path)
    if target.exists() and target.is_symlink():
        raise HostTrustError("known_hosts must not be a symlink")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text("", encoding="utf-8", newline="\n")
        try:
            os.chmod(target, 0o600)
        except OSError:
            pass
    return target


def append_key(path: Path | str, key: HostKey) -> None:
    target = ensure_known_hosts(path)
    existing = target.read_text(encoding="utf-8", errors="replace")
    for line in existing.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[1] == key.key_type and fields[2] == key.key_base64:
            return
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(key.line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_sole_key(path: Path | str, key: HostKey) -> None:
    """Write exactly one accepted key; never retain a second host key."""

    target = ensure_known_hosts(path)
    existing = [
        line.strip()
        for line in target.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if existing and existing != [key.line]:
        raise HostTrustError("known_hosts already contains a different host key")
    temp = target.with_name(target.name + ".tmp")
    try:
        temp.write_text(key.line + "\n", encoding="utf-8", newline="\n")
        os.chmod(temp, 0o600)
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def verify_trusted(
    *,
    known_hosts: Path | str,
    host: str,
    port: int,
    expected_fingerprint: str,
    ssh_keygen_path: str | None = None,
    live_keys: Iterable[HostKey] | None = None,
    runner: Runner | None = None,
) -> None:
    if not isinstance(expected_fingerprint, str) or not expected_fingerprint.startswith(
        "SHA256:"
    ):
        raise HostTrustError("trusted host fingerprint is missing")
    keys = known_host_fingerprints(
        known_hosts=known_hosts,
        host=host,
        port=port,
        ssh_keygen_path=ssh_keygen_path,
        runner=runner,
    )
    known = {item.fingerprint for item in keys}
    if known != {expected_fingerprint}:
        raise HostTrustError(
            "dedicated known_hosts must contain exactly the accepted host key"
        )
    if live_keys is not None:
        live = {item.fingerprint for item in live_keys}
        if live != {expected_fingerprint}:
            raise HostTrustError("live host key differs from the trusted fingerprint")


def trust_scanned_key(
    *,
    known_hosts: Path | str,
    host: str,
    port: int,
    expected_fingerprint: str,
    keyscan_path: str | None = None,
    runner: Runner | None = None,
) -> HostKey:
    keys = scan_host(
        host=host,
        port=port,
        keyscan_path=keyscan_path,
        runner=runner,
    )
    matching = [item for item in keys if item.fingerprint == expected_fingerprint]
    if len(matching) != 1:
        raise HostTrustError(
            "scanned host key does not have exactly one accepted fingerprint"
        )
    write_sole_key(known_hosts, matching[0])
    return matching[0]


def remove_untrusted_key(*, known_hosts: Path | str, host: str, port: int) -> None:
    """Never called automatically; kept explicit for an owner-driven repair."""

    raise HostTrustError("automatic known_hosts removal is forbidden")


__all__ = [
    "HostKey",
    "append_key",
    "ensure_known_hosts",
    "fingerprint_for_key",
    "host_key_name",
    "known_host_fingerprints",
    "parse_keyscan",
    "remove_untrusted_key",
    "scan_host",
    "trust_scanned_key",
    "verify_trusted",
    "write_sole_key",
]
