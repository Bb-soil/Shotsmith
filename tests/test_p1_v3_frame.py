from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.p1.png_frame import (
    PngFrameError,
    _decompress_bounded,
    crop_rgb,
    decode_png_rgb,
    derive_frame,
    resize_area_rgb,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets" / "frames" / "shot-01-start-v3.png"
SOURCE_SHA = "5e3029aa7edb9207c1503635606beeeed9b012995b2a296616668c84fe68083a"
DERIVED_SHA = "925eccbe02a49b74bd2e8cd190fe26bd9557628bb996149f093b894136c53add"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_and_actual_derived_dimensions_and_hash(tmp_path: Path) -> None:
    assert _sha(SOURCE) == SOURCE_SHA
    output = tmp_path / "derived.png"
    policy_path = tmp_path / "derived.policy.json"
    policy = derive_frame(
        source_path=SOURCE,
        output_path=output,
        policy_path=policy_path,
        expected_source_sha256=SOURCE_SHA,
    )
    assert _sha(output) == DERIVED_SHA
    assert policy["target"]["sha256"] == DERIVED_SHA
    width, height, _pixels = decode_png_rgb(output)
    assert (width, height) == (704, 1280)
    saved = json.loads(policy_path.read_text(encoding="utf-8"))
    assert saved["crop"]["box_left_top_right_bottom"] == [14, 6, 927, 1666]
    assert saved["crop"]["width"] == 913
    assert saved["crop"]["height"] == 1660
    assert saved["resize"]["anisotropic"] is False
    assert saved["resize"]["scale_width"] == "704/913"
    assert saved["resize"]["scale_height"] == "1280/1660"
    assert 704 * 1660 == 1280 * 913


def test_derive_is_idempotent_and_does_not_touch_source(tmp_path: Path) -> None:
    output = tmp_path / "derived.png"
    policy = tmp_path / "derived.policy.json"
    before = _sha(SOURCE)
    first = derive_frame(
        source_path=SOURCE,
        output_path=output,
        policy_path=policy,
        expected_source_sha256=SOURCE_SHA,
    )
    second = derive_frame(
        source_path=SOURCE,
        output_path=output,
        policy_path=policy,
        expected_source_sha256=SOURCE_SHA,
    )
    assert first == second
    assert _sha(SOURCE) == before
    assert _sha(output) == DERIVED_SHA


def test_existing_policy_mismatch_fails_closed(tmp_path: Path) -> None:
    output = tmp_path / "derived.png"
    policy = tmp_path / "derived.policy.json"
    derive_frame(
        source_path=SOURCE,
        output_path=output,
        policy_path=policy,
        expected_source_sha256=SOURCE_SHA,
    )
    data = json.loads(policy.read_text(encoding="utf-8"))
    data["target"]["sha256"] = "0" * 64
    policy.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(PngFrameError):
        derive_frame(
            source_path=SOURCE,
            output_path=output,
            policy_path=policy,
            expected_source_sha256=SOURCE_SHA,
        )


def test_wrong_source_hash_fails_before_output(tmp_path: Path) -> None:
    with pytest.raises(PngFrameError):
        derive_frame(
            source_path=SOURCE,
            output_path=tmp_path / "derived.png",
            policy_path=tmp_path / "derived.policy.json",
            expected_source_sha256="0" * 64,
        )
    assert not (tmp_path / "derived.png").exists()


def test_small_uniform_resize_is_deterministic() -> None:
    pixels = bytes(
        [
            0, 0, 0, 10, 10, 10, 20, 20, 20, 30, 30, 30,
            40, 40, 40, 50, 50, 50, 60, 60, 60, 70, 70, 70,
            80, 80, 80, 90, 90, 90, 100, 100, 100, 110, 110, 110,
            120, 120, 120, 130, 130, 130, 140, 140, 140, 150, 150, 150,
        ]
    )
    first = resize_area_rgb(4, 4, pixels, 2, 2)
    second = resize_area_rgb(4, 4, pixels, 2, 2)
    assert first == second
    assert len(first) == 2 * 2 * 3


def test_crop_rejects_out_of_bounds() -> None:
    with pytest.raises(PngFrameError):
        crop_rgb(4, 4, b"\x00" * 48, (0, 0, 5, 4))


def test_zlib_stream_must_reach_eof_and_have_no_unused_data() -> None:
    import zlib

    compressed = zlib.compress(b"abc")
    with pytest.raises(PngFrameError):
        _decompress_bounded(compressed[:-1], 3)
    with pytest.raises(PngFrameError):
        _decompress_bounded(compressed + b"extra", 3)


def test_existing_output_must_match_recomputed_pin(tmp_path: Path) -> None:
    output = tmp_path / "derived.png"
    policy = tmp_path / "derived.policy.json"
    derive_frame(
        source_path=SOURCE,
        output_path=output,
        policy_path=policy,
        expected_source_sha256=SOURCE_SHA,
    )
    original = output.read_bytes()
    output.write_bytes(original[:-10] + b"corrupt")
    with pytest.raises(PngFrameError):
        derive_frame(
            source_path=SOURCE,
            output_path=output,
            policy_path=policy,
            expected_source_sha256=SOURCE_SHA,
        )
