"""Datasets + collates for the six recipes.

Cls (ImageNet+GT, squash 384²) — `ClsSquashCollate`. Returns (x_uint8, y_long).
Seg (LSDIR+pseudolabel, 4 ep, random crop 512²) — `LSDIRRandCropCollate`.
    Returns x_uint8 only; the trainer obtains pseudolabels from the teacher.
Clip (ImageNet, naflex variable shape) — `NaflexClipCollate`. Returns x_uint8
    at variable (H', W'), bs=1.

The naflex_resize helper is ported inline so this module does not import
from the archived iter-6 harness.
"""

from __future__ import annotations

from typing import Callable, Optional

import PIL.Image
import datasets
import torch
from torchvision.transforms.v2 import (
    ColorJitter,
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)
from torchvision.transforms.v2.functional import pil_to_tensor


# --- naflex resize ----------------------------------------------------------


def naflex_resize(
    img: PIL.Image.Image,
    max_num_patches: int = 256,
    patch_size: int = 16,
    snap: int = 32,
) -> PIL.Image.Image:
    """Naflex aspect-preserving resize with a fixed patch budget.

    Returns a PIL image at (H', W') such that:
      - aspect ratio ≈ H/W (preserved within snap rounding),
      - both dims are multiples of `snap` (default 32 — accommodates FRAPPE
        max_ps=32 and WaLLoC SNAP_MULTIPLE=16),
      - (H'/patch_size) * (W'/patch_size) ≤ max_num_patches.

    Implementation calls transformers'
    `get_image_size_for_max_num_patches` at the snap granularity with a
    rescaled budget so the SigLIP-2 16²-patch count stays under the cap.
    """
    from transformers.models.siglip2.image_processing_siglip2 import (
        get_image_size_for_max_num_patches,
    )
    img = img.convert("RGB")
    h, w = img.height, img.width
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


# --- dataset loaders --------------------------------------------------------


def load_imagenet_train(samples: Optional[int] = None):
    ds = datasets.load_dataset("timm/imagenet-1k-wds", split="train")
    if samples is not None:
        ds = ds.select(range(min(samples, ds.num_rows)))
    return ds


def load_lsdir_train(samples: Optional[int] = None):
    ds = datasets.load_dataset("danjacobellis/LSDIR", split="train")
    if samples is not None:
        ds = ds.select(range(min(samples, ds.num_rows)))
    return ds


def load_kodak_calib():
    return datasets.load_dataset("danjacobellis/kodak", split="validation")


def load_train(name: str, samples: Optional[int] = None):
    """Dispatch by dataset path. Adds new HF dataset names here as needed."""
    if name == "timm/imagenet-1k-wds":
        return load_imagenet_train(samples)
    if name == "danjacobellis/LSDIR":
        return load_lsdir_train(samples)
    ds = datasets.load_dataset(name, split="train")
    if samples is not None:
        ds = ds.select(range(min(samples, ds.num_rows)))
    return ds


# --- collates ---------------------------------------------------------------


def _get_sample_pil(sample) -> PIL.Image.Image:
    """LSDIR uses 'image'; timm/imagenet-1k-wds uses 'jpg'."""
    if "image" in sample:
        return sample["image"]
    if "jpg" in sample:
        return sample["jpg"]
    raise KeyError(f"sample has no PIL field; keys={list(sample.keys())}")


def make_cls_squash_collate(crop_size: int) -> Callable:
    """ImageNet train: squash to (crop, crop) BICUBIC + HFlip + GT label.

    Mirrors the iter-5 / iter-10 training collate. Sample must have a
    `cls` integer label (timm/imagenet-1k-wds convention).
    """
    flip = RandomHorizontalFlip()

    def collate(batch):
        xs, ys = [], []
        for sample in batch:
            img = _get_sample_pil(sample).convert("RGB").resize(
                (crop_size, crop_size), PIL.Image.Resampling.BICUBIC,
            )
            xs.append(pil_to_tensor(img).unsqueeze(0))
            ys.append(int(sample["cls"]))
        x = flip(torch.cat(xs, dim=0))
        y = torch.tensor(ys, dtype=torch.long)
        return x, y

    return collate


def make_lsdir_train_collate(crop_size: int) -> Callable:
    """Iter-1 LSDIR collate for the seg path: Resize → ColorJitter →
    RandomCrop → HFlip. No labels — the trainer derives pseudolabels from
    the teacher on the fly."""
    def collate(batch):
        xs = []
        for sample in batch:
            transform = Compose([
                Resize(
                    int(crop_size * 1.2),
                    interpolation=PIL.Image.Resampling.BICUBIC,
                    max_size=int(crop_size * 2.5),
                ),
                ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.0),
                RandomCrop(
                    (crop_size, crop_size),
                    pad_if_needed=True, padding_mode="symmetric",
                ),
            ])
            img = transform(_get_sample_pil(sample).convert("RGB"))
            xs.append(pil_to_tensor(img).unsqueeze(0))
        x = torch.cat(xs, dim=0)
        x = RandomHorizontalFlip()(x)
        return x

    return collate


def make_naflex_clip_collate() -> Callable:
    """Clip-task collate: naflex_resize → HFlip → (1, 3, H', W') uint8.

    No ColorJitter / RandomCrop — the eval pipeline only naflex-resizes,
    and we keep the train distribution aligned. bs=1 hard-coded.
    """
    flip = RandomHorizontalFlip()

    def collate(batch):
        if len(batch) != 1:
            raise ValueError(f"clip collate requires bs=1, got bs={len(batch)}")
        pil = _get_sample_pil(batch[0]).convert("RGB")
        pil = naflex_resize(pil)
        x = pil_to_tensor(pil).unsqueeze(0)
        return flip(x)

    return collate


def make_collate(task: str, train_ds: str, crop_size: Optional[int]) -> Callable:
    """Dispatch by task → return a collate that pairs with `train_ds`.

    Cls: ImageNet+GT → ClsSquashCollate (uint8, label_long).
    Seg: LSDIR+pseudolabel → LSDIRRandCropCollate (uint8 only).
    Clip: ImageNet → NaflexClipCollate (uint8 at variable (H', W'), bs=1).
    """
    if task == "cls":
        assert crop_size is not None
        return make_cls_squash_collate(crop_size)
    if task == "seg":
        assert crop_size is not None
        return make_lsdir_train_collate(crop_size)
    if task == "clip":
        return make_naflex_clip_collate()
    raise ValueError(f"unknown task={task!r}")
