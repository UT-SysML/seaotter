"""Experimental pipelines for the "is the SEA OTTER sandwich a generic
domain-aligner?" question raised after iter-6.

  - `sandbox` : raw uint8 → SEA OTTER sandwich → uint8 → ConvNeXt.
                One op (phase2_k tier). Compares against the 85.13%
                raw-input anchor measured in iter-4 §0.
  - `walsand` : raw uint8 → WaLLoC encode/decode → sandwich → uint8.
                5 ops matching `wal`; compares against `wal` at matched
                transmit bpp.

Both use the phase-2 K=3 warm-start checkpoint at `phase2_k=2` (the
PHASE2_K_DEFAULT used by every other SEA OTTER pipeline). No fine-tuning.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import PIL.Image
import torch
from torchvision.transforms.v2.functional import pil_to_tensor


from .._pipeline import (  # noqa: E402
    PHASE2_K3_PATH_DEFAULT, PHASE2_K_DEFAULT, load_sandwich,
)
from seaotter import jpeg_codec  # noqa: E402

from ._base import Pipeline


# --------------------------------------------------------------------------
# shared sandwich loader (used by both classes; CPU copy fork-inherited)
# --------------------------------------------------------------------------

def _load_sandwich_both(device, phase2_k):
    fwd_dev, proxy_dev, inv_dev, meta = load_sandwich(
        PHASE2_K3_PATH_DEFAULT, phase2_k, device,
    )
    fwd_dev.eval(); proxy_dev.eval(); inv_dev.eval()
    fwd_cpu, _, inv_cpu, _ = load_sandwich(
        PHASE2_K3_PATH_DEFAULT, phase2_k, torch.device("cpu"),
    )
    fwd_cpu.eval(); inv_cpu.eval()
    qt = torch.clamp(proxy_dev.qtable().detach().round(), 1, 255).to(torch.int32).cpu()
    return fwd_dev, inv_dev, fwd_cpu, inv_cpu, qt, meta


# --------------------------------------------------------------------------
# sandbox — raw uint8 → sandwich → uint8 → ConvNeXt
# --------------------------------------------------------------------------

class SandboxPipeline(Pipeline):
    """`sandbox` = sandwich-only, no front-end codec. Tests whether the
    SEA OTTER sandwich on its own lifts accuracy vs raw input — i.e., whether
    the seab/seaft lift is mostly a domain-alignment effect.
    """

    short = "sandbox"

    def __init__(self, *, op: dict, task: str, device: torch.device):
        super().__init__(op=op, task=task, device=device)
        self.phase2_k = int(op["value"])
        (self.fwd_dev, self.inv_dev,
         self.fwd_cpu, self.inv_cpu,
         self._qtable_int, self._meta) = _load_sandwich_both(device, self.phase2_k)

    @torch.no_grad()
    def collate_encode(self, pil_img: PIL.Image.Image) -> bytes:
        x_u8 = pil_to_tensor(pil_img.convert("RGB")).unsqueeze(0).to(torch.float32)
        # x_u8 is already in [0, 255] from pil_to_tensor (it returns uint8 by default).
        # Feed as float for the sandwich's codec_input_uint8.
        z_codec_in_u8 = self.fwd_cpu.codec_input_uint8(x_u8).squeeze(0)
        return jpeg_codec.encode(z_codec_in_u8, self._qtable_int, subsampling=0)

    @torch.no_grad()
    def decode_blobs_batch(self, blobs):
        out_list = []
        storage_bytes: list[int] = []
        for blob in blobs:
            z_codec_out_u8 = jpeg_codec.decode(blob)
            z_float = z_codec_out_u8.to(torch.float32).unsqueeze(0).to(self.device)
            rgb_hat_u8 = self.inv_dev.output_uint8(z_float)
            out_list.append(rgb_hat_u8.squeeze(0))
            storage_bytes.append(len(blob))
        recon = torch.stack(out_list, dim=0)
        return recon, [len(b) for b in blobs], None

    @torch.no_grad()
    def decode_only_consumer(self, blob_bytes: bytes) -> torch.Tensor:
        z_codec_out_u8 = jpeg_codec.decode(blob_bytes)
        z_float = z_codec_out_u8.to(torch.float32).unsqueeze(0).to(self.device)
        return self.inv_dev.output_uint8(z_float).squeeze(0)

    def config_block(self) -> dict:
        return {
            "codec": "sandbox",
            "phase2_init": self._meta.get("phase2_init"),
            "phase2_k": self.phase2_k,
            "phase2_arch": self._meta.get("arch"),
        }


# --------------------------------------------------------------------------
# walsand — WaLLoC encode/decode → sandwich → uint8 → ConvNeXt
# --------------------------------------------------------------------------

class WalSandwichPipeline(Pipeline):
    """`walsand` = WaLLoC front-end + SEA OTTER sandwich on the consumer side.

    Transmit blob = WaLLoC WebP-lossless latent bytes (same as `wal`).
    Storage bytes = JPEG sandwich bytes (additional on-disk cost).
    """

    short = "walsand"

    def __init__(self, *, op: dict, task: str, device: torch.device):
        super().__init__(op=op, task=task, device=device)
        self.q = float(op["value"])
        if not 0 < self.q <= 100:
            raise ValueError(f"walsand: pixel_ratio must be in (0, 100], got {self.q}")
        from compressors.walloc._codec import load_codec, SNAP_MULTIPLE
        self._SNAP_MULTIPLE = SNAP_MULTIPLE
        self._codec_dev, self._info = load_codec(device=device, torch_dtype=torch.float32)
        self._codec_cpu, _ = load_codec(device=torch.device("cpu"), torch_dtype=torch.float32)
        (self.fwd_dev, self.inv_dev,
         self.fwd_cpu, self.inv_cpu,
         self._qtable_int, self._meta) = _load_sandwich_both(device, PHASE2_K_DEFAULT)

    def _snap_hw(self, h: int, w: int) -> tuple[int, int]:
        from compressors.walloc._codec import snap_shape
        return snap_shape(h, w, self.q)

    @torch.no_grad()
    def collate_encode(self, pil_img: PIL.Image.Image) -> bytes:
        """Sensor-side blob = WaLLoC's WebP-lossless latent bytes (the
        transmission unit). The sandwich runs at the consumer step in
        decode_blobs_batch — its bytes are storage, not transmit."""
        from compressors.walloc._codec import (
            encode_to_latent, latent_to_webp_bytes, resize_bicubic, to_model_input,
        )
        x = pil_to_tensor(pil_img.convert("RGB")).unsqueeze(0).to(torch.float32) / 255.0
        _, _, H, W = x.shape
        snap_h, snap_w = self._snap_hw(H, W)
        x_r = resize_bicubic(x, (snap_h, snap_w)).clamp(0, 1)
        x_in = to_model_input(x_r)
        z_hat = encode_to_latent(self._codec_cpu, x_in)
        latent_bytes = latent_to_webp_bytes(z_hat, self._info.latent_bits)
        # 4-byte (H, W) header; excluded from transmit accounting (see
        # decode_blobs_batch). Needed for the clip task's variable naflex shape.
        return struct.pack("<HH", H, W) + latent_bytes

    @torch.no_grad()
    def decode_blobs_batch(self, blobs):
        from compressors.walloc._codec import (
            decode_from_blob, from_model_output, resize_bicubic,
        )
        from ..schema import CROP_FOR_TASK
        crop = CROP_FOR_TASK[self.task]

        recons = []
        storage_bytes: list[int] = []
        transmit_bytes: list[int] = []
        for blob in blobs:
            H_orig, W_orig = struct.unpack("<HH", blob[:4])
            walloc_blob = blob[4:]
            # Stage 1: WaLLoC decode (server) → uint8 RGB at crop / naflex shape.
            x_hat = decode_from_blob(
                self._codec_dev, walloc_blob,
                self._info.latent_dim, self._info.latent_bits,
                self.device, torch.float32,
            )
            recon_01 = from_model_output(x_hat).clamp(0, 1)
            if crop is not None:
                recon_01 = resize_bicubic(recon_01, (crop, crop)).clamp(0, 1)
            else:
                # clip: upsample back to (H_orig, W_orig) so SigLIP-2 naflex
                # sees the same patch grid as the clean branch.
                recon_01 = resize_bicubic(recon_01, (H_orig, W_orig)).clamp(0, 1)
            x_u8 = (recon_01 * 255.0).round().clamp(0, 255)  # float [0,255]
            # Stage 2: SEA OTTER sandwich roundtrip on the WaLLoC reconstruction.
            z_codec_in_u8 = self.fwd_dev.codec_input_uint8(x_u8)
            z_codec_in_cpu = z_codec_in_u8.squeeze(0).cpu()
            jpeg_bytes = jpeg_codec.encode(z_codec_in_cpu, self._qtable_int, subsampling=0)
            z_codec_out_u8 = jpeg_codec.decode(jpeg_bytes)
            z_float = z_codec_out_u8.to(torch.float32).unsqueeze(0).to(self.device)
            rgb_hat_u8 = self.inv_dev.output_uint8(z_float).squeeze(0)
            recons.append(rgb_hat_u8)
            storage_bytes.append(len(jpeg_bytes))
            transmit_bytes.append(len(walloc_blob))
        recon = torch.stack(recons, dim=0)
        return recon, transmit_bytes, storage_bytes

    @torch.no_grad()
    def decode_only_consumer(self, blob_bytes: bytes) -> torch.Tensor:
        recon, _, _ = self.decode_blobs_batch([blob_bytes])
        return recon.squeeze(0)

    @torch.no_grad()
    def transcode_to_storage_blob(self, sensor_blob: bytes) -> bytes:
        """Cloud-side one-time transcode: WaLLoC latents → sandwich → JPEG file."""
        from compressors.walloc._codec import (
            decode_from_blob, from_model_output, resize_bicubic,
        )
        from ..schema import CROP_FOR_TASK
        crop = CROP_FOR_TASK[self.task]
        H_orig, W_orig = struct.unpack("<HH", sensor_blob[:4])
        walloc_blob = sensor_blob[4:]
        x_hat = decode_from_blob(
            self._codec_dev, walloc_blob,
            self._info.latent_dim, self._info.latent_bits,
            self.device, torch.float32,
        )
        recon_01 = from_model_output(x_hat).clamp(0, 1)
        if crop is not None:
            recon_01 = resize_bicubic(recon_01, (crop, crop)).clamp(0, 1)
        else:
            recon_01 = resize_bicubic(recon_01, (H_orig, W_orig)).clamp(0, 1)
        x_u8 = (recon_01 * 255.0).round().clamp(0, 255)
        z_codec_in_u8 = self.fwd_dev.codec_input_uint8(x_u8)
        z_codec_in_cpu = z_codec_in_u8.squeeze(0).cpu()
        jpeg_bytes = jpeg_codec.encode(z_codec_in_cpu, self._qtable_int, subsampling=0)
        return jpeg_bytes

    @torch.no_grad()
    def decode_steady_state_consumer(self, jpeg_bytes: bytes) -> torch.Tensor:
        """Deployed steady-state consumer-side decode: JPEG decode + F^{-1}."""
        z_codec_out_u8 = jpeg_codec.decode(jpeg_bytes)
        z_float = z_codec_out_u8.to(torch.float32).unsqueeze(0).to(self.device)
        rgb_hat_u8 = self.inv_dev.output_uint8(z_float)
        return rgb_hat_u8.squeeze(0)

    def config_block(self) -> dict:
        return {
            "codec": "walsand",
            "q_pixel_ratio": self.q,
            "phase2_k": PHASE2_K_DEFAULT,
            "phase2_arch": self._meta.get("arch"),
            "walloc_latent_dim": self._info.latent_dim,
            "walloc_latent_bits": self._info.latent_bits,
        }
