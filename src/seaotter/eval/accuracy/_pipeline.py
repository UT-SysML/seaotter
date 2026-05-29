# ARCHIVED 2026-05-22. New training code lives in src/seaotter/train/.
# See prompts/organize_src_code.md and src/seaotter/train/recipes/.
# The iter-1 .. iter-10 checkpoints under this tree remain readable; the
# consolidated trainer in src/seaotter/train/ is the canonical entry
# point going forward. The eval-side iter-6 harness still imports from
# this file; the train-side code path is replaced.

"""Shared SEA OTTER pipeline plumbing for the pre_trained_convnext branch.

Builds and composes:
  - FRAPPE encoder + decoder (frozen always at the encoder; frozen-or-hot at
    the decoder per --frappe_decoder_hot)
  - JPEG sandwich: ForwardTransform + JPEGProxy + InverseTransform
    (warm-started from a phase-2 K=3 checkpoint at the picked k; frozen-or-hot
    per --jpeg_proxy_hot)
  - Teacher model (frozen, eval) — convnext_tiny.in12k_ft_in1k_384 for cls,
    smp-hub/upernet-convnext-tiny for seg

Forward (training):
  x_uint8 (B, 3, H, W) on the device
    → x_float = x_uint8 / 127.5 - 1.0                 # in [-1, 1]
    → with no_grad: int8 latents from FRAPPE encoder (FROZEN, never hot)
    → FRAPPE decoder ([-1, 1])                         # FROZEN or HOT
    → rgb_codec_in = (rgb_decoded + 1) * 127.5         # [0, 255] float
    → fwd(rgb_codec_in) → proxy(z) → inv(z_codec)      # FROZEN or HOT
    → rgb_hat ([0, 255] approximately)
    → x_hat_m1_1 = rgb_hat / 127.5 - 1.0               # in [-1, 1]
    → student_normalized = in1k_timm(x_hat_m1_1)
    → logits_student = teacher(student_normalized)     # grad flows through

Pseudolabel (no grad, computed once per step):
  → teacher_normalized = in1k_timm(x_float)
  → with no_grad: pseudolabel = teacher(teacher_normalized).argmax(dim=1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from compressors.frappe.model import load_from_hub, load_progressive_model
from compressors.frappe.quantize import srgb_to_linear
from gigatorch import normalize as gt_normalize
from seaotter.color_transform import ForwardTransform, InverseTransform
from seaotter.proxy import JPEGProxy


PHASE2_K3_PATH_DEFAULT = (
    "/home/dgj335/danjacobellis/seaotter/shared_color/experiments/"
    "round16_dual_goal/iter8/S3_K3_lams_0p75_0p4_0p22_w_0p3_0p7_1p5.pth"
)
PHASE2_K_DEFAULT = 2


@dataclass
class FrappeBundle:
    model: nn.Module
    config: object
    n_ch: int
    max_ps: int
    linear_input: bool


def load_frappe(n_ch: int, device: torch.device | str) -> FrappeBundle:
    """Load pretrained FRAPPE at the requested n_ch tier and freeze both halves."""
    config, weights, n_trained = load_from_hub()
    if n_ch < 1 or n_ch > n_trained:
        raise ValueError(f"frappe_n_ch must be in [1, {n_trained}], got {n_ch}")
    model = load_progressive_model(weights, config, n_ch, device)
    model._linear_input = getattr(config, "linear_input", False)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    max_ps = max(model.scale_groups[i][0] for i in range(len(model.scale_groups)))
    return FrappeBundle(
        model=model, config=config, n_ch=n_ch,
        max_ps=max_ps, linear_input=model._linear_input,
    )


def load_sandwich(
    phase2_init: str,
    phase2_k: int,
    device: torch.device | str,
) -> tuple[ForwardTransform, JPEGProxy, InverseTransform, dict]:
    """Warm-start the JPEG sandwich from a phase-2 K-tier checkpoint.

    Returns (fwd, proxy, inv, meta) where meta contains arch, lambdas, k.
    """
    ckpt = torch.load(phase2_init, map_location=device, weights_only=False)
    arch = ckpt.get("config", {}).get("arch")
    if arch not in ("A", "B", "D", "E", "F", "G"):
        raise ValueError(f"phase2 checkpoint arch={arch!r} not supported")
    proxy_state_dicts = ckpt["proxy_state_dicts"]
    if not 0 <= phase2_k < len(proxy_state_dicts):
        raise ValueError(
            f"phase2_k={phase2_k} out of range [0, {len(proxy_state_dicts) - 1}]"
        )

    fwd = ForwardTransform(arch=arch).to(device)
    inv = InverseTransform(arch=arch).to(device)
    proxy = JPEGProxy(init=torch.full((3, 8, 8), 8.0)).to(device)
    fwd.load_state_dict(ckpt["fwd_state_dict"])
    inv.load_state_dict(ckpt["inv_state_dict"])
    proxy.load_state_dict(proxy_state_dicts[phase2_k])

    meta = {
        "arch": arch,
        "lambdas": ckpt.get("lambdas", []),
        "phase2_k": phase2_k,
        "phase2_init": phase2_init,
    }
    return fwd, proxy, inv, meta


def set_sandwich_freeze(
    fwd: ForwardTransform,
    proxy: JPEGProxy,
    inv: InverseTransform,
    hot: bool,
) -> None:
    """Toggle requires_grad on the sandwich. eval() vs train() is the caller's
    responsibility (it affects JPEGProxy noise + boundary noise)."""
    for m in (fwd, proxy, inv):
        for p in m.parameters():
            p.requires_grad_(bool(hot))


def set_decoder_freeze(frappe: nn.Module, hot: bool) -> None:
    """Toggle requires_grad on the FRAPPE decoder (encoder stays frozen)."""
    for p in frappe.decoder.parameters():
        p.requires_grad_(bool(hot))


# ---------------------------------------------------------------------------
# Forward helpers (composed pipeline)
# ---------------------------------------------------------------------------


def uint8_to_frappe_input(x_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 in [0, 255] → float in [-1, 1] (FRAPPE encoder input domain)."""
    return x_uint8.to(torch.float32) / 127.5 - 1.0


# ---------------------------------------------------------------------------
# WaLLoC codec — load + forward helpers for the iter-7 walft path
# ---------------------------------------------------------------------------


@dataclass
class WallocBundle:
    codec: nn.Module
    info: object
    crop_size: int
    pixel_ratio: float
    snap_h: int
    snap_w: int


def load_walloc(
    pixel_ratio: float,
    crop_size: int,
    device: torch.device | str,
) -> WallocBundle:
    """Load the pretrained WaLLoC RGB_16x codec, freeze the encoder, leave the
    decoder requires_grad-able.

    Returns a WallocBundle that pins the operating-point (pixel_ratio + the
    pre-snapped resize target derived from the eval crop). The encoder is
    frozen here once and for all; the caller controls decoder hot/cold via
    set_walloc_decoder_freeze.
    """
    from compressors.walloc._codec import load_codec, snap_shape
    codec, info = load_codec(device=device, torch_dtype=torch.float32)
    if not 0 < pixel_ratio <= 100:
        raise ValueError(f"pixel_ratio must be in (0, 100], got {pixel_ratio}")
    snap_h, snap_w = snap_shape(crop_size, crop_size, pixel_ratio)
    # Freeze encoder pieces. encoder is nn.Sequential([Encoder, ToUniform, Round]).
    for p in codec.encoder.parameters():
        p.requires_grad_(False)
    # Decoder hot by default for walft; caller can toggle.
    for p in codec.decoder.parameters():
        p.requires_grad_(True)
    codec.eval()  # mode flips set later
    return WallocBundle(
        codec=codec, info=info, crop_size=crop_size,
        pixel_ratio=float(pixel_ratio), snap_h=snap_h, snap_w=snap_w,
    )


def set_walloc_decoder_freeze(codec: nn.Module, hot: bool) -> None:
    """Toggle requires_grad on the WaLLoC decoder. encoder always frozen."""
    for p in codec.decoder.parameters():
        p.requires_grad_(bool(hot))


def walloc_encode_latent(codec: nn.Module, x_01: torch.Tensor) -> torch.Tensor:
    """Frozen WaLLoC encoder under no_grad. x_01 ∈ [0, 1] at the snapped shape.

    Returns the rounded integer latent (same as decoder consumes during the
    zero-shot walsand path).
    """
    from compressors.walloc._codec import encode_to_latent, to_model_input
    with torch.no_grad():
        x_in = to_model_input(x_01)
        z_hat = encode_to_latent(codec, x_in)
    return z_hat


def walloc_decode_latent(
    codec: nn.Module, z_hat: torch.Tensor, *, decoder_hot: bool,
) -> torch.Tensor:
    """Run the WaLLoC decoder (hot or cold) → reconstruction in [-0.5, 0.5]."""
    from compressors.walloc._codec import decode_from_latent
    if decoder_hot:
        x_hat = decode_from_latent(codec, z_hat)
    else:
        with torch.no_grad():
            x_hat = decode_from_latent(codec, z_hat)
    return x_hat


def walloc_pipeline_forward(
    walloc: nn.Module,
    fwd: ForwardTransform,
    proxy: JPEGProxy,
    inv: InverseTransform,
    x_uint8: torch.Tensor,
    *,
    snap_h: int,
    snap_w: int,
    crop_size: int,
    decoder_hot: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """WaLLoC analog of pipeline_forward.

    x_uint8 : (B, 3, crop, crop) uint8 on device.
    Returns (rgb_codec_in, rgb_hat) — both ~[0, 255] float, (B, 3, crop, crop).

    Steps:
      uint8 → x_01 → resize to (snap_h, snap_w) → WaLLoC enc (frozen) →
      WaLLoC dec (hot/cold) → [0,1] → resize back to crop → *255 → sandwich.
    """
    from compressors.walloc._codec import (
        from_model_output, resize_bicubic,
    )

    x_01 = x_uint8.to(torch.float32) / 255.0
    H, W = x_01.shape[-2:]
    if (H, W) != (snap_h, snap_w):
        x_snap = resize_bicubic(x_01, (snap_h, snap_w)).clamp(0, 1)
    else:
        x_snap = x_01
    z_hat = walloc_encode_latent(walloc, x_snap)
    x_dec = walloc_decode_latent(walloc, z_hat, decoder_hot=decoder_hot)
    recon_01 = from_model_output(x_dec).clamp(0, 1)
    if (snap_h, snap_w) != (crop_size, crop_size):
        recon_01 = resize_bicubic(recon_01, (crop_size, crop_size)).clamp(0, 1)
    rgb_codec_in = recon_01 * 255.0
    z = fwd(rgb_codec_in)
    z_codec = proxy(z)
    rgb_hat = inv(z_codec)
    return rgb_codec_in, rgb_hat


def in1k_timm_from_uint8(x_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 [0,255] → [-1,1] → ImageNet timm-style normalization."""
    return gt_normalize.in1k_timm(uint8_to_frappe_input(x_uint8))


def in1k_timm_from_rgb_codec_out(rgb_hat_0_255: torch.Tensor) -> torch.Tensor:
    """rgb_hat ≈ [0, 255] float → ImageNet timm-style normalization, keeping grad."""
    return gt_normalize.in1k_timm(rgb_hat_0_255 / 127.5 - 1.0)


def frappe_encode_int8(frappe: nn.Module, x_float_m1_1: torch.Tensor) -> list[torch.Tensor]:
    """Run the frozen FRAPPE encoder and return int8-rounded latents (fp32 with
    integer entries in [-127, 127]). Always under no_grad — the encoder is
    frozen across the entire Phase-4 branch."""
    x_in = srgb_to_linear(x_float_m1_1) if getattr(frappe, "_linear_input", False) else x_float_m1_1
    with torch.no_grad():
        latents = frappe.encode(x_in)
        latents = [z.round().clamp(-127, 127) for z in latents]
    return latents


def pipeline_forward(
    frappe: nn.Module,
    fwd: ForwardTransform,
    proxy: JPEGProxy,
    inv: InverseTransform,
    x_uint8: torch.Tensor,
    *,
    decoder_hot: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full SEA OTTER pipeline on a batch of uint8 RGB inputs.

    Returns (rgb_codec_in, rgb_hat):
      - rgb_codec_in : (B, 3, H, W) float in ~[0, 255]; FRAPPE-decoder RGB
        that's fed into the JPEG sandwich. Carries grad iff decoder_hot.
      - rgb_hat      : (B, 3, H, W) float in ~[0, 255]; pipeline output.

    The JPEG proxy stores `_last_dct`, `_last_q_map` for downstream rate proxies.
    """
    x_float = uint8_to_frappe_input(x_uint8)
    latents = frappe_encode_int8(frappe, x_float)

    if decoder_hot:
        rgb_decoded = frappe.decode(latents).clamp(-1.0, 1.0)
    else:
        with torch.no_grad():
            rgb_decoded = frappe.decode(latents).clamp(-1.0, 1.0)

    rgb_codec_in = (rgb_decoded + 1.0) * 127.5
    z = fwd(rgb_codec_in)
    z_codec = proxy(z)
    rgb_hat = inv(z_codec)
    return rgb_codec_in, rgb_hat


# ---------------------------------------------------------------------------
# Teacher models
# ---------------------------------------------------------------------------

DEFAULT_TEACHER_NAME = {
    "cls": "convnext_tiny.in12k_ft_in1k_384",
    "seg": "smp-hub/upernet-convnext-tiny",
    "clip": "google/siglip2-base-patch16-naflex",
}


# iter-9: clip task — SigLIP-2 naflex prototype-matrix cache lives here.
CLIP_CACHE_DIR = "/home/dgj335/danjacobellis/seaotter/pre_trained_convnext/cache"
CLIP_NAFLEX_MAX_PATCHES = 256
CLIP_NAFLEX_PATCH_SIZE = 16


class _SiglipTeacherWrapper(nn.Module):
    """Wraps a SigLIP-2 naflex model with a differentiable patchifier so the
    vision encoder can be invoked on arbitrary multiples-of-16 RGB tensors.

    `vision_forward(x)` accepts a `(B, 3, H, W)` tensor of pixel values in
    [0, 255] (uint8 or float; float keeps gradient). H, W must be multiples
    of 16 with `(H/16) * (W/16) <= max_num_patches`. Returns a dict:

      - "per_patch":  (B, max_num_patches, hidden) — last_hidden_state
      - "pooled":     (B, hidden) L2-normalized — MAP pooler_output
      - "patch_mask": (B, max_num_patches) bool — True for real patches

    Padded positions get the standard SigLIP-2 zero patch + mask=False so
    they do not contribute to attention or the masked per-patch loss.

    The wrapper is frozen + `.eval()` after construction.
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
        """(B, 3, H, W) in [0, 255] → (pixel_values, attention_mask, spatial_shapes).

        Matches Siglip2ImageProcessor.convert_image_to_patches geometry
        exactly:
          image (H, W, C) → reshape (n_h, P, n_w, P, C) → permute
            (n_h, n_w, P, P, C) → flatten (n_h*n_w, P*P*C).

        Translated to (B, C, H, W):
          permute (B, H, W, C) → reshape (B, n_h, P, n_w, P, C) →
            permute (B, n_h, n_w, P, P, C) → flatten (B, n_h*n_w, P*P*C).
        """
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
        # SigLIP-2 normalization: mean=0.5, std=0.5 on rescaled-to-[0,1] inputs
        # → equivalent to x/127.5 - 1.0 from [0, 255].
        x_norm = x_float / 127.5 - 1.0
        patches = (
            x_norm.permute(0, 2, 3, 1)  # (B, H, W, C)
                  .reshape(B, n_h, P, n_w, P, C)
                  .permute(0, 1, 3, 2, 4, 5)  # (B, n_h, n_w, P, P, C)
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
        """Run the SigLIP-2 vision tower on (B, 3, H, W) in [0, 255].

        Gradient flows back through x_rgb iff it requires grad.
        """
        pixel_values, attention_mask, spatial_shapes = self._patchify(x_rgb)
        out = self.model.vision_model(
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            spatial_shapes=spatial_shapes,
        )
        per_patch = out.last_hidden_state
        pooled = out.pooler_output
        pooled = torch.nn.functional.normalize(pooled, dim=-1)
        return {
            "per_patch": per_patch,
            "pooled": pooled,
            "patch_mask": attention_mask.bool(),
        }

    def vision_embed(self, x_rgb: torch.Tensor) -> torch.Tensor:
        """Convenience: L2-normalized pooled embedding only. (B, hidden)."""
        return self.vision_forward(x_rgb)["pooled"]

    # ---- ImageNet prototype matrix (text encoder, cached) ---------------

    def compute_clip_prototypes(self) -> torch.Tensor:
        """Build (or load) the 1000-class L2-normalized text prototype matrix.

        Templates come from `open_clip.zero_shot_metadata.SIMPLE_IMAGENET_TEMPLATES`
        (7 templates) — the SigLIP-2 HF model card does not expose a canonical
        template list, so we fall back to the published "simple" prompt set
        per the iter-9 §D3 prescription. Classnames are
        `open_clip.zero_shot_metadata.IMAGENET_CLASSNAMES` (1000 entries).

        Cache key incorporates the SigLIP-2 model name and patch budget; if
        the user changes either, regenerate the cache by deleting the file.
        """
        if self._prototype_cache is not None:
            return self._prototype_cache

        from pathlib import Path
        device = next(self.model.parameters()).device
        cache_path = Path(CLIP_CACHE_DIR) / (
            "siglip2_base_patch16_naflex_imagenet_prototypes.pt"
        )
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
            # Use the processor's text path directly — handles SigLIP-2's
            # max_length=64 default and the right padding token automatically.
            toks = self.processor(
                text=prompts, padding="max_length", return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                text_out = self.model.get_text_features(**toks)
            # get_text_features returns BaseModelOutputWithPooling; the
            # SigLIP contrastive embedding lives in `.pooler_output`.
            text_emb = torch.nn.functional.normalize(
                text_out.pooler_output, dim=-1,
            )
            proto = text_emb.mean(dim=0)
            proto = torch.nn.functional.normalize(proto, dim=-1)
            prototypes.append(proto)
        Pm = torch.stack(prototypes, dim=0).contiguous()  # (1000, hidden)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(Pm, cache_path)
        self._prototype_cache = Pm
        return Pm

    def zero_shot_logits(self, x_rgb: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) [0, 255] → (B, 1000) cosine-similarity logits."""
        embed = self.vision_forward(x_rgb)["pooled"]  # already L2-normalized
        P = self.compute_clip_prototypes()
        return embed @ P.T


def load_teacher(
    task: str,
    device: torch.device | str,
    name: str | None = None,
) -> nn.Module:
    """Load and freeze a teacher model in .eval() mode.

    `name` defaults to the iter-1 deployment teacher (small ConvNeXt-Tiny
    variant for both tasks). Iter 2 may swap in a larger pseudolabel teacher
    via `name="convnext_large.fb_in22k_ft_in1k_384"` (cls) or
    `name="smp-hub/upernet-convnext-large"` (seg) without changing the rest
    of the pipeline.

    Resolution:
      - Names starting with "google/siglip" route to a SigLIP-2 wrapper.
      - Other names containing "/" are segmentation_models.pytorch HF paths.
      - The rest go through timm.create_model.
    """
    name = name or DEFAULT_TEACHER_NAME[task]
    if name.startswith("google/siglip"):
        from transformers import AutoModel, AutoProcessor
        processor = AutoProcessor.from_pretrained(name)
        model = AutoModel.from_pretrained(name)
        model.eval()
        model.to(device)
        wrapper = _SiglipTeacherWrapper(
            model=model, processor=processor,
            max_num_patches=CLIP_NAFLEX_MAX_PATCHES,
            patch_size=CLIP_NAFLEX_PATCH_SIZE,
        )
        return wrapper
    if "/" in name:
        import segmentation_models_pytorch as smp
        m = smp.from_pretrained(name)
    else:
        import timm
        m = timm.create_model(name, pretrained=True)
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()
    m.to(device)
    return m


CROP_FOR_TASK = {"cls": 384, "seg": 512, "clip": None}
NUM_CLASSES_FOR_TASK = {"cls": 1000, "seg": 150, "clip": 1000}


def teacher_forward(
    teacher: nn.Module,
    x_uint8_or_rgb_hat: torch.Tensor,
    *,
    is_uint8: bool,
) -> torch.Tensor:
    """Convenience wrapper that applies the ImageNet-timm normalization and
    runs the teacher. Caller controls grad context (no_grad for pseudolabel,
    grad for student loss)."""
    if is_uint8:
        x_norm = in1k_timm_from_uint8(x_uint8_or_rgb_hat)
    else:
        x_norm = in1k_timm_from_rgb_codec_out(x_uint8_or_rgb_hat)
    return teacher(x_norm)
