"""Preprocessing helpers shared by accuracy + throughput entry points.

- squash_resize: bicubic resize to (size, size). Matches timm's
  convnext_tiny.in12k_ft_in1k_384 (crop_pct=1.0, crop_mode=squash) for
  cls; the upernet-convnext-tiny seg teacher already expects 512x512.
- naflex_resize: SigLIP-2 naflex aspect-preserving resize with a fixed
  patch budget. Both dims are multiples of 32 so the FRAPPE encoder
  (max_ps=32) and WaLLoC encoder (SNAP_MULTIPLE=16) both accept the
  shape without further snapping.
- ADE20K label offset: class 0 in the raw annotation = "unannotated"
  → 255 ignore index.
"""

from __future__ import annotations

import PIL.Image
import datasets
import torch

from .schema import (
    CLIP_NAFLEX_MAX_PATCHES, CLIP_NAFLEX_PATCH_SIZE, CLIP_NAFLEX_SNAP,
    CROP_FOR_TASK, VAL_DS,
)


def squash_resize(img: PIL.Image.Image, size: int) -> PIL.Image.Image:
    return img.convert("RGB").resize(
        (size, size), PIL.Image.Resampling.BICUBIC,
    )


def naflex_resize(
    img: PIL.Image.Image,
    max_num_patches: int = CLIP_NAFLEX_MAX_PATCHES,
    patch_size: int = CLIP_NAFLEX_PATCH_SIZE,
    snap: int = CLIP_NAFLEX_SNAP,
) -> PIL.Image.Image:
    """Naflex aspect-preserving resize with a fixed patch budget.

    Compute (H', W') such that (H'/patch_size) * (W'/patch_size) ≤
    max_num_patches, aspect ratio ≈ H/W, and both dims multiples of
    `snap` (default 32 — accommodates FRAPPE's max_ps=32 and WaLLoC's
    SNAP_MULTIPLE=16 simultaneously).

    Implementation: reuse transformers' Siglip2 helper at the snap
    granularity. Calling it with patch_size=snap and
    max_num_patches=max_num_patches*(patch_size/snap)**2 yields a
    grid where each "patch" of size `snap` corresponds to (snap/patch_size)**2
    SigLIP-2 patches; the patch budget at SigLIP-2's true patch_size is
    preserved.
    """
    from transformers.models.siglip2.image_processing_siglip2 import (
        get_image_size_for_max_num_patches,
    )

    img = img.convert("RGB")
    h, w = img.height, img.width
    # snap/patch_size must be a positive integer (we use 32/16=2).
    if snap % patch_size != 0:
        raise ValueError(
            f"snap ({snap}) must be a multiple of patch_size ({patch_size})"
        )
    step = snap // patch_size
    snap_budget = max_num_patches // (step * step)
    new_h, new_w = get_image_size_for_max_num_patches(
        h, w, patch_size=snap, max_num_patches=snap_budget,
    )
    return img.resize((new_w, new_h), PIL.Image.Resampling.BICUBIC)


def task_crop_pil(img: PIL.Image.Image, task: str) -> PIL.Image.Image:
    """Dispatch by task. Cls/seg → fixed squash. Clip → variable naflex."""
    if task == "clip":
        return naflex_resize(img)
    return squash_resize(img, CROP_FOR_TASK[task])


def load_val(task: str):
    ds_name, split = VAL_DS[task]
    return datasets.load_dataset(ds_name, split=split)


def ade_label_offset(label_uint8: torch.Tensor) -> torch.Tensor:
    """ADE20K: raw class 0 = 'unannotated' → 255 ignore."""
    y = label_uint8.to(torch.int16) - 1
    y[y < 0] = 255
    return y.to(torch.uint8)
