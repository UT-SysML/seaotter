"""Rate-proxy calibration set.

Builds the list of `z_codec_in_uint8` tensors that `RunLengthRate.__init__`
needs to fit its single-scalar bpp calibration constant `alpha`.

Task-aware:
- cls / seg → centre-crop / squash to the recipe's crop_size before encoding.
- clip      → naflex_resize so the calibration distribution matches the
              training and eval input distributions.

Codec-aware:
- FRAPPE → run FRAPPE encode → decode → fwd.codec_input_uint8.
- WaLLoC → resize-in to snap → encode → decode → resize-out → fwd.codec_input_uint8.
"""

from __future__ import annotations

from typing import Optional

import PIL.Image
import torch
from torchvision.transforms.v2.functional import pil_to_tensor

from seaotter.train.codecs import FrappeBundle, WallocBundle
from seaotter.train.data import naflex_resize


@torch.no_grad()
def _calib_uint8_images(
    calib_ds, task: str, crop_size: Optional[int], n: int,
) -> list[PIL.Image.Image]:
    imgs: list[PIL.Image.Image] = []
    for i, sample in enumerate(calib_ds):
        if i >= n:
            break
        img = sample["image"].convert("RGB")
        if task == "clip":
            img = naflex_resize(img)
        else:
            assert crop_size is not None
            w, h = img.size
            h_rs = (h // crop_size) * crop_size
            w_rs = (w // crop_size) * crop_size
            if h_rs == 0 or w_rs == 0:
                h_rs = max(crop_size, (h // 32) * 32)
                w_rs = max(crop_size, (w // 32) * 32)
            img = img.crop((0, 0, w_rs, h_rs))
        imgs.append(img)
    return imgs


@torch.no_grad()
def _frappe_codec_input(
    frappe: FrappeBundle, fwd, x_uint8: torch.Tensor,
) -> torch.Tensor:
    from compressors.frappe.quantize import srgb_to_linear
    x_float = x_uint8.to(torch.float32) / 127.5 - 1.0
    x_in = (
        srgb_to_linear(x_float)
        if getattr(frappe.model, "_linear_input", False)
        else x_float
    )
    latents = frappe.model.encode(x_in)
    latents = [z.round().clamp(-127, 127) for z in latents]
    rgb_decoded = frappe.model.decode(latents).clamp(-1.0, 1.0)
    rgb_codec_in = (rgb_decoded + 1.0) * 127.5
    return fwd.codec_input_uint8(rgb_codec_in).squeeze(0).cpu()


@torch.no_grad()
def _walloc_codec_input(
    walloc: WallocBundle, fwd, x_uint8: torch.Tensor,
) -> torch.Tensor:
    from compressors.walloc._codec import (
        encode_to_latent, decode_from_latent,
        from_model_output, resize_bicubic, to_model_input,
    )
    x_01 = x_uint8.to(torch.float32) / 255.0
    H, W = x_01.shape[-2:]
    snap_h, snap_w = walloc.snap_h, walloc.snap_w
    if (H, W) != (snap_h, snap_w):
        x_snap = resize_bicubic(x_01, (snap_h, snap_w)).clamp(0, 1)
    else:
        x_snap = x_01
    x_in = to_model_input(x_snap)
    z_hat = encode_to_latent(walloc.codec, x_in)
    x_dec = decode_from_latent(walloc.codec, z_hat)
    recon_01 = from_model_output(x_dec).clamp(0, 1)
    if (snap_h, snap_w) != (H, W):
        recon_01 = resize_bicubic(recon_01, (H, W)).clamp(0, 1)
    rgb_codec_in = recon_01 * 255.0
    return fwd.codec_input_uint8(rgb_codec_in).squeeze(0).cpu()


@torch.no_grad()
def build_rate_calibration_set(
    codec_bundle: FrappeBundle | WallocBundle,
    fwd,
    calib_ds,
    *,
    task: str,
    crop_size: Optional[int],
    n: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Compose a list of (3, H, W) uint8 CPU tensors of codec_input pixels —
    the RunLengthRate calibration corpus.

    The ForwardTransform's eval-mode boundary noise is bypassed so the
    calibration sees the deterministic codec input."""
    fwd_was_train = fwd.training
    fwd.eval()
    try:
        imgs: list[torch.Tensor] = []
        for pil in _calib_uint8_images(calib_ds, task, crop_size, n):
            x_uint8 = pil_to_tensor(pil).unsqueeze(0).to(device)
            if isinstance(codec_bundle, FrappeBundle):
                z_u8 = _frappe_codec_input(codec_bundle, fwd, x_uint8)
            else:
                z_u8 = _walloc_codec_input(codec_bundle, fwd, x_uint8)
            imgs.append(z_u8)
    finally:
        fwd.train(fwd_was_train)
    return imgs
