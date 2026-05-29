"""Teacher loader — thin re-export from `pre_trained_convnext.pipeline`.

Iter-6 deliberately points at the canonical iter-1 deployment teachers
(`convnext_tiny.in12k_ft_in1k_384` for cls, `smp-hub/upernet-convnext-tiny`
for seg). Iter-9 adds `google/siglip2-base-patch16-naflex` for the clip
task. All numbers across pipelines are reported against these.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


from ._pipeline import (  # noqa: E402
    DEFAULT_TEACHER_NAME,
    in1k_timm_from_uint8,
    load_teacher as _load_teacher,
)


def load_teacher(task: str, device: torch.device) -> torch.nn.Module:
    return _load_teacher(task, device, name=DEFAULT_TEACHER_NAME[task])


def teacher_logits_from_uint8(
    teacher: torch.nn.Module,
    x_uint8: torch.Tensor,
) -> torch.Tensor:
    """uint8 [0,255] (B, 3, H, W) → teacher logits.

    For cls/seg teachers: routes through ImageNet-timm normalization.
    For the clip teacher: routes through SigLIP-2's own pre-patchifier
    normalization and returns cosine-sim logits against the cached
    ImageNet prototype matrix (i.e., the same shape (B, 1000) that
    the cls teacher emits).
    """
    if hasattr(teacher, "zero_shot_logits"):
        return teacher.zero_shot_logits(x_uint8)
    return teacher(in1k_timm_from_uint8(x_uint8))
