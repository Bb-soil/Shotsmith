from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.kf2.assets import import_image, import_script, validate_library_asset
from scripts.kf2.errors import AssetError
from scripts.kf2.store import ProjectStore
from tests.kf1_fake_comfy import _png


def _project(tmp_path: Path):
    store = ProjectStore(tmp_path / "projects")
    project, path = store.create("p1", title="Assets", orientation="portrait")
    return project, path


def test_image_import_quotes_spaces_cyrillic_and_immutable_source(tmp_path: Path) -> None:
    _project_data, project_dir = _project(tmp_path)
    source_dir = tmp_path / "Мои картинки"
    source_dir.mkdir()
    source = source_dir / "герой 1.png"
    data = _png(64, 64, (10, 20, 30))
    source.write_bytes(data)
    before = source.read_bytes()
    metadata = import_image(project_dir, f'"{source}"', role="provided_keyframe")
    assert metadata["sha256"] == __import__("hashlib").sha256(data).hexdigest()
    assert metadata["width"] == 64 and metadata["height"] == 64
    stored = project_dir / metadata["stored_path"]
    assert stored.read_bytes() == data
    assert source.read_bytes() == before
    assert "reused" in metadata


def test_same_bytes_different_roles_have_distinct_authoring_ids(tmp_path: Path) -> None:
    _project_data, project_dir = _project(tmp_path)
    source = tmp_path / "same.png"
    source.write_bytes(_png(32, 32))
    first = import_image(project_dir, str(source), role="provided_keyframe")
    second = import_image(
        project_dir,
        str(source),
        role="object",
        subject_id="car",
        library=[first],
    )
    assert first["id"] != second["id"]
    assert first["stored_path"] == second["stored_path"]


def test_image_import_rejects_bad_extension_content_and_sensitive_name(tmp_path: Path) -> None:
    _project_data, project_dir = _project(tmp_path)
    gif = tmp_path / "image.gif"
    gif.write_bytes(b"GIF89a")
    with pytest.raises(AssetError):
        import_image(project_dir, str(gif), role="provided_keyframe")
    fake = tmp_path / "fake.png"
    fake.write_bytes(b"not an image")
    with pytest.raises(AssetError):
        import_image(project_dir, str(fake), role="provided_keyframe")
    secret = tmp_path / "credentials.png"
    secret.write_bytes(_png(8, 8))
    with pytest.raises(AssetError):
        import_image(project_dir, str(secret), role="provided_keyframe")


def test_provenance_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    _project_data, project_dir = _project(tmp_path)
    source = tmp_path / "image.png"
    source.write_bytes(_png(8, 8))
    with pytest.raises(AssetError):
        import_image(
            project_dir,
            str(source),
            role="provided_keyframe",
            provenance={"content_sha256": "0" * 64},
        )


def test_script_import_preserves_bom_bytes_and_utf8_text(tmp_path: Path) -> None:
    _project_data, project_dir = _project(tmp_path)
    source = tmp_path / "сценарий.txt"
    raw = "\ufefftitle: Тест\r\ndescription:\r\nПервая строка\r\n---\r\n".encode("utf-8")
    source.write_bytes(raw)
    metadata, text = import_script(project_dir, str(source))
    stored = project_dir / metadata["stored_path"]
    assert stored.read_bytes() == raw
    assert text.startswith("title: Тест")
    assert "Первая строка" in text
    with pytest.raises(AssetError):
        import_script(project_dir, str(source.with_suffix(".md")))


def test_library_asset_tamper_and_limits(tmp_path: Path, monkeypatch) -> None:
    _project_data, project_dir = _project(tmp_path)
    source = tmp_path / "image.png"
    source.write_bytes(_png(8, 8))
    metadata = import_image(project_dir, str(source), role="provided_keyframe")
    assert validate_library_asset(project_dir, metadata)["sha256"] == metadata["sha256"]
    (project_dir / metadata["stored_path"]).write_bytes(_png(8, 8, (1, 1, 1)))
    with pytest.raises(AssetError):
        validate_library_asset(project_dir, metadata)
    monkeypatch.setattr("scripts.kf2.assets.MAX_PROJECT_BYTES", 1)
    with pytest.raises(AssetError):
        import_image(project_dir, str(source), role="provided_keyframe")


def test_source_symlink_rejected_when_supported(tmp_path: Path) -> None:
    _project_data, project_dir = _project(tmp_path)
    source = tmp_path / "image.png"
    source.write_bytes(_png(8, 8))
    link = tmp_path / "link.png"
    try:
        os.symlink(source, link)
    except OSError:
        pytest.skip("symlink privilege unavailable")
    with pytest.raises(AssetError):
        import_image(project_dir, str(link), role="provided_keyframe")
