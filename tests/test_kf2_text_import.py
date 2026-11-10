from __future__ import annotations

from scripts.kf2.store import new_project
from scripts.kf2.text_import import apply_proposal, parse_text_document


def _bom(text: str) -> bytes:
    return ("\ufeff" + text).encode("utf-8")


def test_bom_multiline_russian_terminator_and_escape() -> None:
    data = _bom(
        "title: Две сцены\n"
        "orientation: portrait\n"
        "style:\n"
        "Мягкий рисунок\n"
        "---\n"
        "scene: shot-01\n"
        "description:\n"
        "Героиня стоит у окна.\n"
        "Она смотрит вдаль.\n"
        "---\n"
        "a_instruction: \n"
        "b_instruction:\n"
        "Сохранить камеру.\n"
        "\\---\n"
        "Это literal terminator.\n"
        "---\n"
        "scene: shot-02\n"
        "description: Вторая сцена.\n"
    )
    proposal = parse_text_document(data)
    assert proposal["mode"] == "keyvalue"
    assert proposal["fields"]["title"] == "Две сцены"
    assert "Мягкий рисунок" in proposal["fields"]["style"]
    assert proposal["scenes"][0]["description"].startswith("Героиня стоит")
    assert "Она смотрит вдаль." in proposal["scenes"][0]["description"]
    assert proposal["scenes"][0]["final"]["instruction"] == (
        "Сохранить камеру.\n---\nЭто literal terminator."
    )
    assert proposal["scenes"][1]["id"] == "shot-02"


def test_numbered_parser_is_only_a_description_proposal() -> None:
    data = "1. Первая сцена\nОна идёт.\n\n2) Вторая сцена\nОн ждёт.\n".encode("utf-8")
    proposal = parse_text_document(data, numbered=True)
    assert proposal["mode"] == "numbered"
    assert [scene["id"] for scene in proposal["scenes"]] == ["shot-01", "shot-02"]
    assert proposal["scenes"][0]["description"] == "Первая сцена\nОна идёт."
    assert proposal["scenes"][0]["action"] == ""
    assert proposal["scenes"][0]["camera"] == ""
    assert proposal["scenes"][0]["final"]["instruction"] == ""


def test_apply_proposal_updates_existing_scene_without_duplicate() -> None:
    project = new_project("p1", title="x", orientation="portrait")
    proposal = parse_text_document(
        "scene: shot-01\ndescription: Новый текст.\na_instruction: A text.\n".encode("utf-8")
    )
    project = apply_proposal(project, proposal)
    project = apply_proposal(project, proposal)
    assert len(project["scenes"]) == 1
    assert project["scenes"][0]["description"] == "Новый текст."
    assert project["scenes"][0]["initial"]["instruction"] == "A text."
    assert project["import"]["needs_confirmation"] is True


def test_invalid_utf8_is_rejected() -> None:
    import pytest
    from scripts.kf2.errors import TextImportError

    with pytest.raises(TextImportError):
        parse_text_document(b"\xff\xfe\x00bad")
