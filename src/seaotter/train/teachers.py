"""Frozen downstream teachers for the three tasks.

- `cls`  → timm convnext_tiny.in12k_ft_in1k_384 (squash 384²)
- `seg`  → smp-hub upernet-convnext-tiny (squash 512²)
- `clip` → google/siglip2-base-patch16-naflex (variable per-image)

See pre_trained_convnext/teachers.md (kept in the archive) for the full
asymmetry / baseline discussion. The asymmetry between the `_384` cls
backbone and the non-`_384` seg backbone is intentional and load-bearing
— do not unify them on one pretrain variant.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from gigatorch import normalize as gt_normalize


REPO_ROOT = Path(__file__).resolve().parents[3]
CLIP_CACHE_DIR = REPO_ROOT / "cache"
CLIP_NAFLEX_MAX_PATCHES = 256
CLIP_NAFLEX_PATCH_SIZE = 16

DEFAULT_TEACHER_NAME = {
    "cls": "convnext_tiny.in12k_ft_in1k_384",
    "seg": "smp-hub/upernet-convnext-tiny",
    "clip": "google/siglip2-base-patch16-naflex",
}


def in1k_timm_from_uint8(x_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 [0,255] → [-1,1] → ImageNet timm normalization (cls/seg)."""
    return gt_normalize.in1k_timm(x_uint8.to(torch.float32) / 127.5 - 1.0)


def in1k_timm_from_rgb_codec_out(rgb_hat_0_255: torch.Tensor) -> torch.Tensor:
    """rgb_hat ≈ [0,255] float → ImageNet timm normalization, grad-preserving."""
    return gt_normalize.in1k_timm(rgb_hat_0_255 / 127.5 - 1.0)


# --- ConvNeXt / UperNet ------------------------------------------------------


def load_convnext_cls_teacher(
    device: torch.device | str,
    name: str = DEFAULT_TEACHER_NAME["cls"],
) -> nn.Module:
    import timm
    m = timm.create_model(name, pretrained=True)
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()
    m.to(device)
    return m


def load_upernet_seg_teacher(
    device: torch.device | str,
    name: str = DEFAULT_TEACHER_NAME["seg"],
) -> nn.Module:
    import segmentation_models_pytorch as smp
    m = smp.from_pretrained(name)
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()
    m.to(device)
    return m


# --- SigLIP-2 naflex wrapper -------------------------------------------------


class SiglipTeacherWrapper(nn.Module):
    """Wraps a SigLIP-2 naflex model with a differentiable patchifier so the
    vision encoder can be invoked on arbitrary multiples-of-16 RGB tensors.

    Bit-exact parity with `Siglip2ImageProcessor.convert_image_to_patches`
    has been verified (cosine=1.000000 on aligned inputs).

    `vision_forward((B, 3, H, W) in [0, 255])` returns a dict:

      - "per_patch":  (B, max_num_patches, hidden) — last_hidden_state
      - "pooled":     (B, hidden) L2-normalized — MAP pooler_output
      - "patch_mask": (B, max_num_patches) bool — True for real patches
    """

    def __init__(self, model, processor, max_num_patches: int, patch_size: int):
        super().__init__()
        self.model = model
        self.processor = processor
        self.max_num_patches = int(max_num_patches)
        self.patch_size = int(patch_size)
        self._prototype_cache: torch.Tensor | None = None
        self._eval_classnames: tuple[str, ...] | None = None
        self._templates_used: tuple[str, ...] | None = None
        for p in self.parameters():
            p.requires_grad_(False)
        self.model.eval()

    # ---- differentiable patchifier --------------------------------------

    def _patchify(self, x_rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x_rgb.dim() != 4 or x_rgb.shape[1] != 3:
            raise ValueError(f"expected (B, 3, H, W); got {tuple(x_rgb.shape)}")
        B, C, H, W = x_rgb.shape
        P = self.patch_size
        if H % P != 0 or W % P != 0:
            raise ValueError(
                f"H={H}, W={W} must be multiples of patch_size={P}"
            )
        n_h, n_w = H // P, W // P
        n_patches = n_h * n_w
        if n_patches > self.max_num_patches:
            raise ValueError(
                f"n_patches={n_patches} exceeds max_num_patches={self.max_num_patches}"
            )
        x_float = x_rgb if x_rgb.is_floating_point() else x_rgb.to(torch.float32)
        # SigLIP-2 normalization (mean=0.5, std=0.5) on rescaled-to-[0,1] is
        # equivalent to x/127.5 - 1.0 from [0, 255].
        x_norm = x_float / 127.5 - 1.0
        patches = (
            x_norm.permute(0, 2, 3, 1)
                  .reshape(B, n_h, P, n_w, P, C)
                  .permute(0, 1, 3, 2, 4, 5)
                  .reshape(B, n_patches, P * P * C)
        )
        M = self.max_num_patches
        device = x_rgb.device
        if n_patches < M:
            pad = patches.new_zeros((B, M - n_patches, P * P * C))
            pixel_values = torch.cat([patches, pad], dim=1)
        else:
            pixel_values = patches
        attention_mask = torch.zeros((B, M), dtype=torch.int32, device=device)
        attention_mask[:, :n_patches] = 1
        spatial_shapes = torch.tensor(
            [[n_h, n_w]] * B, dtype=torch.long, device=device,
        )
        return pixel_values, attention_mask, spatial_shapes

    # ---- forward API ----------------------------------------------------

    def vision_forward(self, x_rgb: torch.Tensor) -> dict:
        pixel_values, attention_mask, spatial_shapes = self._patchify(x_rgb)
        out = self.model.vision_model(
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            spatial_shapes=spatial_shapes,
        )
        per_patch = out.last_hidden_state
        pooled = torch.nn.functional.normalize(out.pooler_output, dim=-1)
        return {
            "per_patch": per_patch,
            "pooled": pooled,
            "patch_mask": attention_mask.bool(),
        }

    def vision_embed(self, x_rgb: torch.Tensor) -> torch.Tensor:
        return self.vision_forward(x_rgb)["pooled"]

    # ---- ImageNet prototype matrix (text encoder, cached) ---------------

    def compute_clip_prototypes(self) -> torch.Tensor:
        """Build (or load) the 1000-class L2-normalized text prototype matrix.

        Uses open_clip's SIMPLE_IMAGENET_TEMPLATES (7 templates) + 1000-class
        IMAGENET_CLASSNAMES; the SigLIP-2 HF model card has no canonical
        zero-shot template list. Cache lives at
        `<repo>/cache/siglip2_base_patch16_naflex_imagenet_prototypes.pt`
        (~3 MB).
        """
        if self._prototype_cache is not None:
            return self._prototype_cache

        device = next(self.model.parameters()).device
        cache_path = CLIP_CACHE_DIR / "siglip2_base_patch16_naflex_imagenet_prototypes.pt"
        if cache_path.exists():
            P = torch.load(cache_path, map_location=device, weights_only=True)
            self._prototype_cache = P
            return P

        import open_clip.zero_shot_metadata as zmd
        classnames = zmd.IMAGENET_CLASSNAMES
        templates = zmd.SIMPLE_IMAGENET_TEMPLATES
        self._eval_classnames = tuple(classnames)
        self._templates_used = tuple(t("{}") for t in templates)

        prototypes = []
        for classname in classnames:
            prompts = [tmpl(classname) for tmpl in templates]
            toks = self.processor(
                text=prompts, padding="max_length", return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                text_out = self.model.get_text_features(**toks)
            text_emb = torch.nn.functional.normalize(
                text_out.pooler_output, dim=-1,
            )
            proto = text_emb.mean(dim=0)
            proto = torch.nn.functional.normalize(proto, dim=-1)
            prototypes.append(proto)
        Pm = torch.stack(prototypes, dim=0).contiguous()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(Pm, cache_path)
        self._prototype_cache = Pm
        return Pm

    def zero_shot_logits(self, x_rgb: torch.Tensor) -> torch.Tensor:
        embed = self.vision_forward(x_rgb)["pooled"]  # already L2-normalized
        P = self.compute_clip_prototypes()
        return embed @ P.T


def load_siglip_clip_teacher(
    device: torch.device | str,
    name: str = DEFAULT_TEACHER_NAME["clip"],
) -> SiglipTeacherWrapper:
    from transformers import AutoModel, AutoProcessor
    processor = AutoProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    model.eval()
    model.to(device)
    return SiglipTeacherWrapper(
        model=model, processor=processor,
        max_num_patches=CLIP_NAFLEX_MAX_PATCHES,
        patch_size=CLIP_NAFLEX_PATCH_SIZE,
    )


def load_teacher(task: str, device: torch.device | str):
    """Dispatch on task → load + freeze the canonical teacher."""
    if task == "cls":
        return load_convnext_cls_teacher(device)
    if task == "seg":
        return load_upernet_seg_teacher(device)
    if task == "clip":
        return load_siglip_clip_teacher(device)
    raise ValueError(f"unknown task={task!r}")
