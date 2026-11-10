import pytest
from scripts.i1 import ops
from scripts.i1.errors import RemoteFailure, BlockedError


def test_health_recovers_once_then_requires_real_response(monkeypatch):
    calls=[]
    responses=iter([RemoteFailure("closed"), {"devices": ["cuda"]}])
    def health(**kw):
        value=next(responses)
        if isinstance(value, Exception): raise value
        return value
    monkeypatch.setattr(ops.tunnel, "health", health)
    monkeypatch.setattr(ops.time, "sleep", lambda _: None)
    assert ops._wait_health(8189, timeout=2, recover=lambda: calls.append(1)) == {"devices": ["cuda"]}
    assert calls == [1]


def test_recovery_does_not_override_unknown_owner(monkeypatch):
    monkeypatch.setattr(ops.tunnel, "health", lambda **kw: (_ for _ in ()).throw(RemoteFailure("closed")))
    def recover(): raise BlockedError("owner unknown")
    with pytest.raises(BlockedError, match="unknown"):
        ops._wait_health(8189, timeout=2, recover=recover)


def test_unhealthy_recovery_is_bounded(monkeypatch):
    ticks=iter([0, 0.1, 0.2, 0.3, 3])
    monkeypatch.setattr(ops.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(ops.time, "sleep", lambda _: None)
    monkeypatch.setattr(ops.tunnel, "health", lambda **kw: (_ for _ in ()).throw(RemoteFailure("closed")))
    calls=[]
    with pytest.raises(RemoteFailure):
        ops._wait_health(8189, timeout=2, recover=lambda: calls.append(1))
    assert calls == [1]
