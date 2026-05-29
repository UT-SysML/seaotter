"""WaLLoC + ITU JPEG — `wal_jpeg`.

Sensor: WaLLoC encode → WebP-lossless on latents (transmit).
Cloud:  WaLLoC decode → resize back to crop / naflex shape → ITU JPEG
        (q=75, subsampling=0) round-trip.
Consumer: stock JPEG decode.

Transmit blob = WaLLoC WebP-lossless bytes; storage blob = JPEG bytes.
Mirrors walloc.py up to the decoded RGB; appends a vanilla JPEG sandwich
on the decoded RGB (in place of the SEAOTTER learned sandwich used by
walsand).

Knob: op.value = pixel_ratio (1..100). op.extras.jpeg_quality (default 75).
"""

from __future__ import annotations

import io
import struct

import PIL.Image
import torch
from torchvision.transforms.v2.functional import pil_to_tensor

from ._base import Pipeline
from ..schema import CROP_FOR_TASK


class WalJpegPipeline(Pipeline):
    short = "wal_jpeg"

    def __init__(self, *, op: dict, task: str, device: torch.device):
        super().__init__(op=op, task=task, device=device)
        self.q = float(op["value"])
        if not 0 < self.q <= 100:
            raise ValueError(f"wal_jpeg: pixel_ratio must be in (0, 100], got {self.q}")
        self.q_jpeg = int(op.get("extras", {}).get("jpeg_quality", 75))
        if not 1 <= self.q_jpeg <= 100:
            raise ValueError(f"jpeg_quality must be in [1, 100], got {self.q_jpeg}")

        from compressors.walloc._codec import load_codec, SNAP_MULTIPLE
        self._SNAP_MULTIPLE = SNAP_MULTIPLE
        self._codec_dev, self._info = load_codec(device=device, torch_dtype=torch.float32)
        self._codec_cpu, _ = load_codec(device=torch.device("cpu"), torch_dtype=torch.float32)

    def _snap_hw(self, h: int, w: int) -> tuple[int, int]:
        from compressors.walloc._codec import snap_shape
        if h % self._SNAP_MULTIPLE or w % self._SNAP_MULTIPLE:
            raise ValueError(f"wal_jpeg: H, W must be multiples of 16, got {h}x{w}")
        return snap_shape(h, w, self.q)

    def _jpeg_roundtrip(self, rgb_chw_u8_cpu: torch.Tensor) -> tuple[torch.Tensor, int]:
        arr = rgb_chw_u8_cpu.permute(1, 2, 0).numpy()
        pil = PIL.Image.fromarray(arr, mode="RGB")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=self.q_jpeg, subsampling=0)
        jpeg_bytes = buf.getvalue()
        dec = PIL.Image.open(io.BytesIO(jpeg_bytes))
        dec.load()
        recon = pil_to_tensor(dec.convert("RGB"))
        return recon, len(jpeg_bytes)

    # ---- accuracy path -------------------------------------------------

    @torch.no_grad()
    def collate_encode(self, pil_img: PIL.Image.Image) -> bytes:
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
        # 4-byte (H, W) header (excluded from transmit accounting) so the
        # decoder can resize back to the original shape for the clip task.
        return struct.pack("<HH", H, W) + latent_bytes

    @torch.no_grad()
    def decode_blobs_batch(self, blobs):
        from compressors.walloc._codec import (
            decode_from_blob, from_model_output, resize_bicubic,
        )
        crop = CROP_FOR_TASK[self.task]
        recons: list[torch.Tensor] = []
        transmit_bytes: list[int] = []
        storage_bytes: list[int] = []
        for blob in blobs:
            H_orig, W_orig = struct.unpack("<HH", blob[:4])
            walloc_blob = blob[4:]
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
            rgb_u8 = (recon_01 * 255.0).round().clamp(0, 255).to(torch.uint8).squeeze(0)
            recon, n_jpeg = self._jpeg_roundtrip(rgb_u8.cpu())
            recons.append(recon)
            transmit_bytes.append(len(walloc_blob))
            storage_bytes.append(n_jpeg)
        recon_batch = torch.stack(recons, dim=0).to(self.device)
        return recon_batch, transmit_bytes, storage_bytes

    @torch.no_grad()
    def decode_only_consumer(self, blob_bytes: bytes) -> torch.Tensor:
        recon, _, _ = self.decode_blobs_batch([blob_bytes])
        return recon.squeeze(0)

    def config_block(self) -> dict:
        return {
            "format": "JPEG",
            "quality": self.q_jpeg,
            "subsampling": 0,
            "walloc_pixel_ratio": self.q,
            "walloc_latent_dim": self._info.latent_dim,
            "walloc_latent_bits": self._info.latent_bits,
            "walloc_J": self._info.J,
        }
