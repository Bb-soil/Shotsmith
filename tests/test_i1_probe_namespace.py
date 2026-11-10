"""Regressions for pinned ComfyUI's real namespace-package layout."""
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.i1.probe import ProbeError, _comfy_module_origin


def test_real_namespace_without_init_is_accepted(tmp_path, monkeypatch):
    checkout = tmp_path.resolve()
    package = checkout / "comfy"
    package.mkdir()
    monkeypatch.syspath_prepend(str(checkout))
    monkeypatch.delitem(sys.modules, "comfy", raising=False)
    module = importlib.import_module("comfy")
    try:
        assert module.__file__ is None
        assert _comfy_module_origin(module, checkout) == package
    finally:
        sys.modules.pop("comfy", None)


@pytest.mark.parametrize("case", ["foreign", "multiple", "missing_spec", "wrong_spec", "fileless_regular"])
def test_unsafe_namespace_paths_are_rejected(tmp_path, case):
    checkout = tmp_path.resolve()
    package = checkout / "comfy"
    package.mkdir()
    foreign = checkout / "other"
    foreign.mkdir()
    paths = [str(package)]
    spec_paths = list(paths)
    spec = SimpleNamespace(origin=None, submodule_search_locations=spec_paths)
    if case == "foreign":
        paths = [str(foreign)]
        spec_paths[:] = paths
    elif case == "multiple":
        paths.append(str(foreign))
        spec_paths[:] = paths
    elif case == "missing_spec":
        spec = None
    elif case == "wrong_spec":
        spec_paths[:] = [str(foreign)]
    else:
        (package / "__init__.py").write_text("")
    module = SimpleNamespace(__file__=None, __path__=paths, __spec__=spec)
    with pytest.raises(ProbeError):
        _comfy_module_origin(module, checkout)


def test_regular_package_remains_supported_and_foreign_file_rejected(tmp_path):
    checkout = tmp_path.resolve()
    package = checkout / "comfy"
    package.mkdir()
    init = package / "__init__.py"
    init.write_text("")
    assert _comfy_module_origin(SimpleNamespace(__file__=str(init)), checkout) == init
    with pytest.raises(ProbeError):
        _comfy_module_origin(SimpleNamespace(__file__=str(checkout / "foreign.py")), checkout)
