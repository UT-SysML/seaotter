"""Task-specific losses for the trainer.

- cls_loss : cross-entropy on per-image class logits.
- seg_loss : cross-entropy on per-pixel class logits with ignore_index=255.
- clip_loss: hybrid per-patch + pooled cosine, alpha-weighted toward
  per-patch (default α=0.7 per iter-9 design). Returns (loss, aux_dict)
  for logging.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cls_loss(logits_s: torch.Tensor, y_long: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits_s, y_long)


def seg_loss(logits_s: torch.Tensor, y_long: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits_s, y_long, ignore_index=255)


def clip_loss(
    out_clean: dict, out_codec: dict, alpha_clip: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Hybrid per-patch + pooled cosine.

    out_clean / out_codec come from `SiglipTeacherWrapper.vision_forward` and
    contain {per_patch, pooled, patch_mask}. The per-patch term is a
    masked-mean cosine over the valid (non-padded) patches; the pooled
    term is cosine on the SigLIP-2 MAP pooler embedding.
    """
    per_patch_cos = F.cosine_similarity(
        out_clean["per_patch"], out_codec["per_patch"], dim=-1,
    )  # (B, M)
    mask = out_clean["patch_mask"].to(per_patch_cos.dtype)
    n_valid = mask.sum().clamp_min(1.0)
    loss_patch = 1.0 - (per_patch_cos * mask).sum() / n_valid

    loss_pool = 1.0 - F.cosine_similarity(
        out_clean["pooled"], out_codec["pooled"], dim=-1,
    ).mean()

    loss = alpha_clip * loss_patch + (1.0 - alpha_clip) * loss_pool
    return loss, {
        "loss_patch": float(loss_patch.detach().item()),
        "loss_pool": float(loss_pool.detach().item()),
    }
