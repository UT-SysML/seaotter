"""FRAPPE + WaLLoC codec loading and pipeline-forward helpers.

Ported from `pre_trained_convnext/pipeline.py` (now archived). The encoder
side is always frozen (sensor-side asymmetry — see CLAUDE.md). The decoder
hot/cold is toggled per recipe via the set_*_freeze helpers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


# --- FRAPPE ------------------------------------------------------------------


@dataclass
class FrappeBundle:
    model: nn.Module
    config: object
    n_ch: int
    max_ps: int
    linear_input: bool


def load_frappe(n_ch: int, device: torch.device | str) -> FrappeBundle:
    """Load FRAPPE at the requested n_ch tier; encoder frozen, decoder
    hot-able. Run model.encoder.eval() / model.decoder.train() at use site."""
    from compressors.frappe.model import load_from_hub, load_progressive_model

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


def set_frappe_decoder_freeze(bundle: FrappeBundle, hot: bool) -> None:
    for p in bundle.model.decoder.parameters():
        p.requires_grad_(bool(hot))


def _frappe_encode_int8(frappe: nn.Module, x_float_m1_1: torch.Tensor) -> list[torch.Tensor]:
    from compressors.frappe.quantize import srgb_to_linear
    x_in = (
        srgb_to_linear(x_float_m1_1)
        if getattr(frappe, "_linear_input", False)
        else x_float_m1_1
    )
    with torch.no_grad():
        latents = frappe.encode(x_in)
        latents = [z.round().clamp(-127, 127) for z in latents]
    return latents


def frappe_pipeline_forward(
    bundle: FrappeBundle,
    fwd, proxy, inv,
    x_uint8: torch.Tensor,
    *,
    decoder_hot: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """uint8 [0,255] → FRAPPE enc (frozen) → FRAPPE dec → sandwich.

    Returns (rgb_codec_in, rgb_hat) both float ~[0, 255], shape (B, 3, H, W).
    `proxy._last_dct` / `proxy._last_q_map` are populated for rate proxies.
    """
    x_float = x_uint8.to(torch.float32) / 127.5 - 1.0
    latents = _frappe_encode_int8(bundle.model, x_float)
    if decoder_hot:
        rgb_decoded = bundle.model.decode(latents).clamp(-1.0, 1.0)
    else:
        with torch.no_grad():
            rgb_decoded = bundle.model.decode(latents).clamp(-1.0, 1.0)
    rgb_codec_in = (rgb_decoded + 1.0) * 127.5
    z = fwd(rgb_codec_in)
    z_codec = proxy(z)
    rgb_hat = inv(z_codec)
    return rgb_codec_in, rgb_hat


# --- WaLLoC ------------------------------------------------------------------


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
    """Load the pretrained WaLLoC RGB_16x codec; encoder frozen; decoder
    hot-able. Snaps the codec input to the WaLLoC pixel-ratio grid."""
    from compressors.walloc._codec import load_codec, snap_shape

    codec, info = load_codec(device=device, torch_dtype=torch.float32)
    if not 0 < pixel_ratio <= 100:
        raise ValueError(f"pixel_ratio must be in (0, 100], got {pixel_ratio}")
    snap_h, snap_w = snap_shape(crop_size, crop_size, pixel_ratio)
    for p in codec.encoder.parameters():
        p.requires_grad_(False)
    for p in codec.decoder.parameters():
        p.requires_grad_(True)
    codec.eval()
    return WallocBundle(
        codec=codec, info=info, crop_size=crop_size,
        pixel_ratio=float(pixel_ratio), snap_h=snap_h, snap_w=snap_w,
    )


def set_walloc_decoder_freeze(bundle: WallocBundle, hot: bool) -> None:
    for p in bundle.codec.decoder.parameters():
        p.requires_grad_(bool(hot))


def walloc_pipeline_forward(
    bundle: WallocBundle,
    fwd, proxy, inv,
    x_uint8: torch.Tensor,
    *,
    decoder_hot: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """uint8 [0,255] → WaLLoC enc (frozen) → WaLLoC dec → sandwich.

    The codec operates at (snap_h, snap_w); we resize-in and resize-out
    around it so the sandwich always sees `(crop_size, crop_size)`.
    """
    from compressors.walloc._codec import (
        encode_to_latent, decode_from_latent,
        from_model_output, resize_bicubic, to_model_input,
    )

    x_01 = x_uint8.to(torch.float32) / 255.0
    H, W = x_01.shape[-2:]
    snap_h, snap_w = bundle.snap_h, bundle.snap_w
    if (H, W) != (snap_h, snap_w):
        x_snap = resize_bicubic(x_01, (snap_h, snap_w)).clamp(0, 1)
    else:
        x_snap = x_01

    with torch.no_grad():
        x_in = to_model_input(x_snap)
        z_hat = encode_to_latent(bundle.codec, x_in)

    if decoder_hot:
        x_dec = decode_from_latent(bundle.codec, z_hat)
    else:
        with torch.no_grad():
            x_dec = decode_from_latent(bundle.codec, z_hat)
    recon_01 = from_model_output(x_dec).clamp(0, 1)
    if (snap_h, snap_w) != (H, W):
        recon_01 = resize_bicubic(recon_01, (H, W)).clamp(0, 1)
    rgb_codec_in = recon_01 * 255.0
    z = fwd(rgb_codec_in)
    z_codec = proxy(z)
    rgb_hat = inv(z_codec)
    return rgb_codec_in, rgb_hat
