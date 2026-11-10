from __future__ import annotations

from pathlib import Path

from scripts.kf1.plan import (
    EXAMPLE_PLAN,
    interactive_plan,
    parse_text_plan,
    preview_lines,
    resolve_plan,
)

ROOT = Path(__file__).resolve().parents[1]


def test_example_plan_has_a1_three_generated_frames_and_hash() -> None:
    before = EXAMPLE_PLAN.read_bytes()
    plan = resolve_plan(EXAMPLE_PLAN)
    assert [frame.frame_id for frame in plan.frames] == [
        "shot-01/A",
        "shot-01/B",
        "shot-02/A",
        "shot-02/B",
    ]
    assert [frame.frame_id for frame in plan.generated_frames] == [
        "shot-01/B",
        "shot-02/A",
        "shot-02/B",
    ]
    assert plan.generation_limit == 3
    assert plan.plan_sha256 == resolve_plan(EXAMPLE_PLAN).plan_sha256
    assert plan.refs["first_frame"].sha256 == (
        "925eccbe02a49b74bd2e8cd190fe26bd9557628bb996149f093b894136c53add"
    )
    assert plan.refs["character_reference"].sha256 == (
        "5cc170534dc1d1404f6bdfcb548489681dc43d0f61d3078eb5185ac96016adde"
    )
    assert EXAMPLE_PLAN.read_bytes() == before
    text = "\n".join(preview_lines(plan))
    assert "A1" in text
    assert "character reference" in text
    assert "визуальная приёмка" in text


def test_plain_text_plan_preserves_russian_and_no_json(tmp_path: Path) -> None:
    path = tmp_path / "plan.txt"
    path.write_text(
        "\n".join(
            [
                "title: Две сцены",
                "language: ru",
                "first_frame: assets/frames/derived/shot-01-start-v3-704x1280.png",
                "character_reference: assets/references/character-sheet.png",
                "background_reference:",
                "style: Минималистичный аниме-стиль.",
                "seed: 424242",
                "generation_limit: 3",
                "scene: shot-01",
                "description: Героиня удивляется.",
                "a_source: first_frame",
                "b_source: generate_from_scene_a",
                "b_edit: Keep the exact camera. The heroine looks surprised.",
                "scene: shot-02",
                "description: Она идёт в переулок.",
                "a_source: generate_from_previous_b",
                "a_edit: Create a new fixed camera inside the alley.",
                "b_source: generate_from_scene_a",
                "b_edit: She walks toward the camera.",
            ]
        ),
        encoding="utf-8",
    )
    data = parse_text_plan(path.read_text(encoding="utf-8"))
    assert data["title"] == "Две сцены"
    assert data["scenes"][0]["description"] == "Героиня удивляется."
    plan_path = tmp_path / "plan.json"
    import json

    plan_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    plan = resolve_plan(plan_path)
    assert plan.data["title"] == "Две сцены"
    assert [frame.frame_id for frame in plan.generated_frames] == [
        "shot-01/B",
        "shot-02/A",
        "shot-02/B",
    ]


def test_interactive_plan_uses_russian_answers() -> None:
    answers = iter(
        [
            "Мой план",
            "ru",
            "assets/frames/derived/shot-01-start-v3-704x1280.png",
            "assets/references/character-sheet.png",
            "",
            "Стиль сохранён",
            "424242",
            "3",
            "contain",
            "shot-01",
            "Героиня удивляется",
            "Keep the exact camera and show surprise.",
            "",
            "",
        ]
    )
    data = interactive_plan(input_fn=lambda _prompt: next(answers), print_fn=lambda _line: None)
    assert data["title"] == "Мой план"
    assert data["scenes"][0]["description"] == "Героиня удивляется"
