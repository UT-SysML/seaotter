"""Dataset loaders for the encode-complexity harness.

Each loader returns a `list[PIL.Image]` (RGB, fully-loaded into memory)
of the right resolution for the given dataset label. The adapter's
`prepare()` step then converts each PIL image to the encoder's native
input form (float tensor for FRAPPE/WaLLoC; uint8 tensor for SEAOTTER
standalone JPEG; identity for Pillow codecs).

Supported labels:
  - cls_384      : 384x384 squash of ImageNet val
  - seg_512      : 512x512 squash of ADE20K val (scene_parse_150)
  - clip_naflex  : naflex resize of ImageNet val (variable per-image (H,W))
  - kodak_native : 768x512 / 512x768 native crops of Kodak (24 images)

For throughput timing we sample a fixed number of images (default 256
per dataset; Kodak is naturally 24). The CLI accepts `--n_images` to
override.
"""

from __future__ import annotations

from typing import Callable

import PIL.Image
import datasets as hf_datasets


CROP_FOR_DATASET = {
    "cls_384":      (384, 384),
    "seg_512":      (512, 512),
    "clip_naflex":  None,   # variable per image
    "kodak_native": None,   # native shapes
}

# Naflex parameters (mirror harness/preprocessing.py).
NAFLEX_MAX_PATCHES = 256
NAFLEX_PATCH_SIZE = 16
NAFLEX_SNAP = 32  # multiples-of-32 to accommodate FRAPPE max_ps=32


def _squash(img: PIL.Image.Image, side: int) -> PIL.Image.Image:
    out = img.convert("RGB").resize((side, side), PIL.Image.Resampling.BICUBIC)
    out.load()
    return out


def _naflex(img: PIL.Image.Image) -> PIL.Image.Image:
    """Naflex aspect-preserving resize with the iter-6 patch budget."""
    from transformers.models.siglip2.image_processing_siglip2 import (
        get_image_size_for_max_num_patches,
    )

    rgb = img.convert("RGB")
    h, w = rgb.height, rgb.width
    step = NAFLEX_SNAP // NAFLEX_PATCH_SIZE
    snap_budget = NAFLEX_MAX_PATCHES // (step * step)
    new_h, new_w = get_image_size_for_max_num_patches(
        h, w, patch_size=NAFLEX_SNAP, max_num_patches=snap_budget,
    )
    out = rgb.resize((new_w, new_h), PIL.Image.Resampling.BICUBIC)
    out.load()
    return out


def _load_imagenet_val(n_images: int):
    """Streaming/select first `n_images` from `timm/imagenet-1k-wds` val."""
    # webdataset variant returns a dict with "jpg" key for the image.
    ds = hf_datasets.load_dataset(
        "timm/imagenet-1k-wds", split="validation", streaming=False,
    )
    n = min(n_images, ds.num_rows)
    sub = ds.select(range(n))
    return [sub[i]["jpg"] for i in range(n)]


def _load_ade20k_val(n_images: int):
    ds = hf_datasets.load_dataset(
        "danjacobellis/scene_parse_150", split="validation",
    )
    n = min(n_images, ds.num_rows)
    sub = ds.select(range(n))
    return [sub[i]["image"] for i in range(n)]


def _load_kodak():
    ds = hf_datasets.load_dataset("danjacobellis/kodak", split="validation")
    return [s["image"] for s in ds]


def load_inputs(label: str, n_images: int) -> list[PIL.Image.Image]:
    """Return a list of fully-loaded PIL images for the given dataset.

    The returned images are already at the right resolution for the
    encoder; the per-codec adapter only handles the *encoder's native
    input form* conversion (e.g. PIL -> float tensor for FRAPPE).
    """
    if label == "cls_384":
        raw = _load_imagenet_val(n_images)
        out = [_squash(r, 384) for r in raw]
    elif label == "seg_512":
        raw = _load_ade20k_val(n_images)
        out = [_squash(r, 512) for r in raw]
    elif label == "clip_naflex":
        raw = _load_imagenet_val(n_images)
        out = [_naflex(r) for r in raw]
    elif label == "kodak_native":
        out = _load_kodak()
        for img in out:
            img.load()
    else:
        raise ValueError(f"unknown dataset label {label!r}")
    return out


def shape_distribution(imgs: list[PIL.Image.Image]) -> dict[str, int]:
    """Tally (height, width) counts as `"HxW" -> count` for the naflex JSON."""
    counts: dict[str, int] = {}
    for img in imgs:
        key = f"{img.height}x{img.width}"
        counts[key] = counts.get(key, 0) + 1
    return counts
