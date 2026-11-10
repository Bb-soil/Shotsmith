"""Realistic offline ``/object_info`` fixtures for KF1/F1 tests and the WSL probe.

The previous fake used ``["*", {}]`` for every input, which hid real enum,
type, range, and dynamic-upload failures.  This module builds finite schemas
with real node class names, real model file names and real sampler/scheduler
enums.  ``LoadImage.image`` starts as an empty file enum with
``image_upload: True`` exactly like a clean worker; uploads can be listed later
to model the finite named enum returned by ComfyUI after an upload.

No network, no model bytes, no Comfy import.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

_SAVEVIDEO_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "p1"
    / "savevideo-object_info-7a283320.json"
)

SIMPLE_TYPES = {
    "IMAGE",
    "MASK",
    "MODEL",
    "CLIP",
    "VAE",
    "LATENT",
    "CONDITIONING",
    "VIDEO",
    "STRING",
    "INT",
    "FLOAT",
    "BOOLEAN",
}

QWEN_MODELS = {
    "unet_name": [
        "qwen_image_edit_2511_fp8mixed.safetensors",
        "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
        "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
    ],
    "clip_name": [
        "qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    ],
    "vae_name": ["qwen_image_vae.safetensors", "wan_2.1_vae.safetensors"],
}

SAMPLERS = [
    "euler",
    "euler_ancestral",
    "heun",
    "heunpp2",
    "dpm_2",
    "dpm_2_ancestral",
    "lms",
    "dpm_fast",
    "dpm_adaptive",
    "dpmpp_2s_ancestral",
    "dpmpp_sde",
    "dpmpp_sde_gpu",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu",
    "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu",
    "ddim",
    "uni_pc",
    "uni_pc_bh2",
]

SCHEDULERS = [
    "normal",
    "karras",
    "exponential",
    "sgm_uniform",
    "simple",
    "ddim_uniform",
    "beta",
    "linear_quadratic",
    "kl_optimal",
]

WEIGHT_DTYPES = ["default", "fp8_e4m3fn", "fp8_e5m2", "fp16", "bf16"]

KF1_CLASSES: dict[str, dict[str, Any]] = {
    "LoadImage": {
        "required": {"image": ("COMBO", {"image_upload": True})},
        "optional": {},
        "outputs": ["IMAGE", "MASK"],
    },
    "ImageScale": {
        "required": {
            "image": ("IMAGE", {}),
            "upscale_method": ("COMBO", ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]),
            "width": ("INT", {"min": 64, "max": 8192, "step": 8}),
            "height": ("INT", {"min": 64, "max": 8192, "step": 8}),
            "crop": ("COMBO", ["disabled", "center"]),
        },
        "optional": {},
        "outputs": ["IMAGE"],
    },
    "UNETLoader": {
        "required": {
            "unet_name": ("COMBO", QWEN_MODELS["unet_name"]),
            "weight_dtype": ("COMBO", WEIGHT_DTYPES),
        },
        "optional": {},
        "outputs": ["MODEL"],
    },
    "CLIPLoader": {
        "required": {
            "clip_name": ("COMBO", QWEN_MODELS["clip_name"]),
            "type": (
                "COMBO",
                [
                    "stable_diffusion",
                    "stable_cascade",
                    "sd3",
                    "stable_audio",
                    "mochi",
                    "ltxv",
                    "pixart",
                    "cosmos",
                    "lumina2",
                    "wan",
                    "hidream",
                    "chroma",
                    "ace",
                    "omnigen2",
                    "qwen_image",
                    "hunyuan_image",
                    "flux2",
                    "ovis",
                ],
            ),
        },
        "optional": {"device": ("COMBO", ["default", "cpu"])},
        "outputs": ["CLIP"],
    },
    "VAELoader": {
        "required": {"vae_name": ("COMBO", QWEN_MODELS["vae_name"])},
        "optional": {},
        "outputs": ["VAE"],
    },
    "ModelSamplingAuraFlow": {
        "required": {"model": ("MODEL", {}), "shift": ("FLOAT", {"min": 0.0, "max": 100.0})},
        "optional": {"sampling": ("COMBO", ["disabled", "nearest", "linear"])},
        "outputs": ["MODEL"],
    },
    "CFGNorm": {
        "required": {"model": ("MODEL", {}), "strength": ("FLOAT", {"min": 0.0, "max": 100.0})},
        "optional": {"pre_cfg": ("BOOLEAN", {})},
        "outputs": ["MODEL"],
    },
    "TextEncodeQwenImageEditPlus": {
        "required": {"clip": ("CLIP", {}), "prompt": ("STRING", {"multiline": True})},
        "optional": {
            "vae": ("VAE", {}),
            "image1": ("IMAGE", {}),
            "image2": ("IMAGE", {}),
            "image3": ("IMAGE", {}),
        },
        "outputs": ["CONDITIONING"],
    },
    "VAEEncode": {
        "required": {"pixels": ("IMAGE", {}), "vae": ("VAE", {})},
        "optional": {},
        "outputs": ["LATENT"],
    },
    "KSampler": {
        "required": {
            "model": ("MODEL", {}),
            "seed": ("INT", {"min": 0, "max": 18446744073709551615}),
            "steps": ("INT", {"min": 1, "max": 10000}),
            "cfg": ("FLOAT", {"min": 0.0, "max": 100.0}),
            "sampler_name": ("COMBO", SAMPLERS),
            "scheduler": ("COMBO", SCHEDULERS),
            "positive": ("CONDITIONING", {}),
            "negative": ("CONDITIONING", {}),
            "latent_image": ("LATENT", {}),
            "denoise": ("FLOAT", {"min": 0.0, "max": 1.0}),
        },
        "optional": {},
        "outputs": ["LATENT"],
    },
    "VAEDecode": {
        "required": {"samples": ("LATENT", {}), "vae": ("VAE", {})},
        "optional": {},
        "outputs": ["IMAGE"],
    },
    "SaveImage": {
        "required": {"images": ("IMAGE", {}), "filename_prefix": ("STRING", {})},
        "optional": {},
        "outputs": ["IMAGE"],
    },
}

WAN_CLASSES: dict[str, dict[str, Any]] = {
    "ModelSamplingSD3": {
        "required": {"model": ("MODEL", {}), "shift": ("FLOAT", {"min": 0.0, "max": 100.0})},
        "optional": {"sampling": ("COMBO", ["disabled", "nearest", "linear"])},
        "outputs": ["MODEL"],
    },
    "CLIPTextEncode": {
        "required": {"clip": ("CLIP", {}), "text": ("STRING", {"multiline": True})},
        "optional": {},
        "outputs": ["CONDITIONING"],
    },
    "WanFirstLastFrameToVideo": {
        "required": {
            "positive": ("CONDITIONING", {}),
            "negative": ("CONDITIONING", {}),
            "vae": ("VAE", {}),
            "start_image": ("IMAGE", {}),
            "end_image": ("IMAGE", {}),
            "width": ("INT", {"min": 16, "max": 8192}),
            "height": ("INT", {"min": 16, "max": 8192}),
            "length": ("INT", {"min": 1, "max": 10000}),
            "batch_size": ("INT", {"min": 1, "max": 4096}),
        },
        "optional": {},
        "outputs": ["CONDITIONING", "CONDITIONING", "LATENT"],
    },
    "CreateVideo": {
        "required": {"images": ("IMAGE", {}), "fps": ("FLOAT", {"min": 1.0, "max": 240.0})},
        "optional": {},
        "outputs": ["VIDEO"],
    },
    "KSamplerAdvanced": {
        "required": {
            "model": ("MODEL", {}),
            "positive": ("CONDITIONING", {}),
            "negative": ("CONDITIONING", {}),
            "latent_image": ("LATENT", {}),
            "add_noise": ("COMBO", ["enable", "disable"]),
            "noise_seed": ("INT", {"min": 0, "max": 18446744073709551615}),
            "steps": ("INT", {"min": 1, "max": 10000}),
            "cfg": ("FLOAT", {"min": 0.0, "max": 100.0}),
            "sampler_name": ("COMBO", SAMPLERS),
            "scheduler": ("COMBO", SCHEDULERS),
            "start_at_step": ("INT", {"min": 0, "max": 10000}),
            "end_at_step": ("INT", {"min": 0, "max": 10000}),
            "return_with_leftover_noise": ("COMBO", ["enable", "disable"]),
        },
        "optional": {},
        "outputs": ["LATENT"],
    },
    "SaveVideo": {
        "required": {
            "video": ("VIDEO", {}),
            "filename_prefix": ("STRING", {}),
            "format": ("COMFY_DYNAMICCOMBO_V3", {}),
        },
        "optional": {},
        "outputs": ["VIDEO"],
    },
}


def _entry(value: tuple[Any, Mapping[str, Any]]) -> list[Any]:
    kind, options = value
    if kind == "COMBO":
        # COMBO is serialized as a list of allowed literals.
        return [list(options), {}] if isinstance(options, (list, tuple)) else [str(options)]
    return [kind, dict(options)]


def _class_schema(spec: Mapping[str, Any]) -> dict[str, Any]:
    required = {name: _entry(value) for name, value in spec.get("required", {}).items()}
    optional = {name: _entry(value) for name, value in spec.get("optional", {}).items()}
    output = list(spec.get("outputs") or [])
    return {
        "input": {"required": required, "optional": optional},
        "output": output,
        "output_name": output,
    }


def build_object_info(
    *,
    load_image_files: Sequence[str] | None = None,
    include_wan: bool = False,
) -> dict[str, Any]:
    """Build a finite object_info mapping.

    ``load_image_files`` models ``folder_paths.get_input_files()``.  Default is
    an empty clean worker.  After uploads the caller may pass the finite list of
    uploaded names to model the real enum ComfyUI would return.
    """

    files = list(load_image_files or [])
    classes: dict[str, Any] = {}
    for name, spec in KF1_CLASSES.items():
        schema = _class_schema(spec)
        if name == "LoadImage":
            schema["input"]["required"]["image"] = [files, {"image_upload": True}]
        classes[name] = schema
    if include_wan:
        real_savevideo = json.loads(_SAVEVIDEO_FIXTURE.read_text(encoding="utf-8"))[
            "SaveVideo"
        ]
        for name, spec in WAN_CLASSES.items():
            if name == "SaveVideo":
                # Use the pinned real nested DynamicCombo schema so the actual
                # probe follows mp4 -> h264 -> auto, not a synthetic shortcut.
                classes[name] = real_savevideo
                continue
            classes[name] = _class_schema(spec)
    return classes


__all__ = [
    "KF1_CLASSES",
    "SAMPLERS",
    "SCHEDULERS",
    "WAN_CLASSES",
    "build_object_info",
]
