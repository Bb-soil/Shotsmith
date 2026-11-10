from __future__ import annotations

import base64
from types import SimpleNamespace
import pytest
from scripts.i1 import hosttrust

KEY = base64.b64encode(b"keyscan-regression-public-key").decode()
LINE = f"[example.test]:1258 ssh-ed25519 {KEY}\n"

def result(rc=0, out="", err="", timed_out=False):
    return SimpleNamespace(returncode=rc, stdout=out, stderr=err, timed_out=timed_out)

def which(name):
    return {"ssh-keyscan": "keyscan", "wsl.exe": "wsl"}.get(name)

def test_windows_kex_fallback_still_checks_fingerprint(monkeypatch, tmp_path):
    monkeypatch.setattr(hosttrust, "_IS_WINDOWS", True)
    monkeypatch.setattr(hosttrust.shutil, "which", which)
    calls=[]
    def run(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "keyscan":
            return result(1, err=b"choose_kex: unsupported KEX method sntrup761x25519-sha512@openssh.com")
        assert argv[:3] == ("wsl", "--exec", "ssh-keyscan")
        assert kwargs["timeout"] == 30.0
        assert kwargs["max_output_bytes"] == 256*1024
        return result(out=LINE.encode())
    keys=hosttrust.scan_host(host="example.test", port=1258, runner=run)
    assert keys[0].key_base64 == KEY
    known=tmp_path/"known_hosts"
    with pytest.raises(hosttrust.HostTrustError):
        hosttrust.trust_scanned_key(known_hosts=known,host="example.test",port=1258,
                                   expected_fingerprint="SHA256:wrong",runner=run)
    assert not known.exists() or known.read_text()==""
    assert len(calls)==4

def test_normal_scan_never_uses_wsl(monkeypatch):
    monkeypatch.setattr(hosttrust.shutil, "which", which)
    calls=[]
    def run(argv, **kwargs):
        calls.append(argv); return result(out=LINE)
    assert hosttrust.scan_host(host="example.test",port=1258,runner=run)
    assert len(calls)==1


def test_windows_timeout_uses_bounded_fallback_not_partial_output(monkeypatch):
    monkeypatch.setattr(hosttrust, "_IS_WINDOWS", True)
    monkeypatch.setattr(hosttrust.shutil, "which", which)
    other = base64.b64encode(b"fallback-public-key").decode()
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "keyscan":
            return result(out=LINE, timed_out=True)
        assert kwargs["timeout"] == 30.0
        assert argv[argv.index("-T") + 1] == "13"
        return result(out=f"[example.test]:1258 ssh-ed25519 {other}\n")
    keys = hosttrust.scan_host(host="example.test", port=1258, runner=run)
    assert len(calls) == 2
    assert keys[0].key_base64 == other


@pytest.mark.parametrize("windows", [True, False])
def test_timeout_without_fallback_does_not_accept_partial_key(monkeypatch, windows):
    monkeypatch.setattr(hosttrust, "_IS_WINDOWS", windows)
    monkeypatch.setattr(hosttrust.shutil, "which", lambda name: "keyscan" if name == "ssh-keyscan" else None)
    with pytest.raises(hosttrust.HostTrustError, match="timed out"):
        hosttrust.scan_host(host="example.test", port=1258,
                            runner=lambda *a, **k: result(out=LINE, timed_out=True))


@pytest.mark.parametrize("fallback", [result(1, out=LINE), result(out=LINE, timed_out=True), result(out="bad output")])
def test_native_timeout_and_failed_fallback_never_accepts_key(monkeypatch, fallback):
    monkeypatch.setattr(hosttrust, "_IS_WINDOWS", True)
    monkeypatch.setattr(hosttrust.shutil, "which", which)
    def run(argv, **kwargs):
        return result(out=LINE, timed_out=True) if argv[0] == "keyscan" else fallback
    with pytest.raises(hosttrust.HostTrustError):
        hosttrust.scan_host(host="example.test", port=1258, runner=run)

@pytest.mark.parametrize("windows,error", [(False,"unsupported KEX method x"),(True,"connection refused")])
def test_no_fallback_for_nonwindows_or_other_error(monkeypatch,windows,error):
    monkeypatch.setattr(hosttrust,"_IS_WINDOWS",windows)
    monkeypatch.setattr(hosttrust.shutil,"which",which)
    calls=[]
    def run(argv,**kwargs):calls.append(argv);return result(1,err=error)
    with pytest.raises(hosttrust.HostTrustError,match="scanner compatibility"):
        hosttrust.scan_host(host="example.test",port=1258,runner=run)
    assert len(calls)==1

def test_missing_wsl_reports_compatibility_not_unreachable(monkeypatch):
    monkeypatch.setattr(hosttrust,"_IS_WINDOWS",True)
    monkeypatch.setattr(hosttrust.shutil,"which",lambda name:"keyscan" if name=="ssh-keyscan" else None)
    with pytest.raises(hosttrust.HostTrustError,match="incompatible KEX"):
        hosttrust.scan_host(host="example.test",port=1258,runner=lambda *a,**k:result(1,err="unsupported KEX method x"))

@pytest.mark.parametrize("fallback",[result(1),result(out=LINE,timed_out=True),result(out="bad output")])
def test_fallback_error_timeout_or_malformed_never_trusts(monkeypatch,fallback):
    monkeypatch.setattr(hosttrust,"_IS_WINDOWS",True)
    monkeypatch.setattr(hosttrust.shutil,"which",which)
    def run(argv,**kwargs):
        return result(1,err="unsupported KEX method x") if argv[0]=="keyscan" else fallback
    with pytest.raises(hosttrust.HostTrustError):
        hosttrust.scan_host(host="example.test",port=1258,runner=run)
