"""WaLLoC RGB_16x. Knob: op.value = target pixel ratio q% (1..100).

Workers run a fork-inherited CPU codec to produce WebP-lossless latent
blobs; the main loop runs the GPU codec to decode blobs, resize back
to the eval crop, and emit uint8 recons.
"""

from __future__ import annotations

import io
import struct

import PIL.Image
import torch
from torchvision.transforms.v2.functional import pil_to_tensor

from ._base import Pipeline
from ..schema import CROP_FOR_TASK


class WallocPipeline(Pipeline):
    short = "wal"

    def __init__(self, *, op: dict, task: str, device: torch.device):
        super().__init__(op=op, task=task, device=device)
        self.q = float(op["value"])
        if not 0 < self.q <= 100:
            raise ValueError(f"wal: pixel_ratio must be in (0, 100], got {self.q}")
        from compressors.walloc._codec import load_codec, SNAP_MULTIPLE
        self._SNAP_MULTIPLE = SNAP_MULTIPLE
        self._codec_dev, self._info = load_codec(device=device, torch_dtype=torch.float32)
        # CPU codec is fork-inherited by DataLoader workers.
        self._codec_cpu, _ = load_codec(device=torch.device("cpu"), torch_dtype=torch.float32)

    # ---- shared helpers ------------------------------------------------

    def _snap_hw(self, h: int, w: int) -> tuple[int, int]:
        from compressors.walloc._codec import snap_shape
        if h % self._SNAP_MULTIPLE or w % self._SNAP_MULTIPLE:
            raise ValueError(f"wal: H, W must be multiples of 16, got {h}x{w}")
        return snap_shape(h, w, self.q)

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
        # 4-byte (H, W) header (excluded from transmit accounting); needed for
        # the clip task's variable naflex shape so the decoder can resize back.
        return struct.pack("<HH", H, W) + latent_bytes

    @torch.no_grad()
    def decode_blobs_batch(self, blobs):
        from compressors.walloc._codec import (
            decode_from_blob, from_model_output, resize_bicubic,
        )
        crop = CROP_FOR_TASK[self.task]
        recons = []
        transmit_bytes: list[int] = []
        for blob in blobs:
            H_orig, W_orig = struct.unpack("<HH", blob[:4])
            walloc_blob = blob[4:]
            x_hat = decode_from_blob(
                self._codec_dev, walloc_blob,
                self._info.latent_dim, self._info.latent_bits,
                self.device, torch.float32,
            )
            recon = from_model_output(x_hat).clamp(0, 1)
            if crop is not None:
                recon = resize_bicubic(recon, (crop, crop)).clamp(0, 1)
            else:
                # clip: upsample back to (H_orig, W_orig) so SigLIP-2 sees the
                # same patch grid as the clean branch.
                recon = resize_bicubic(recon, (H_orig, W_orig)).clamp(0, 1)
            recons.append((recon * 255.0).round().clamp(0, 255).to(torch.uint8).squeeze(0))
            transmit_bytes.append(len(walloc_blob))
        recon_batch = torch.stack(recons, dim=0)
        return recon_batch, transmit_bytes, None

    @torch.no_grad()
    def decode_only_consumer(self, blob_bytes: bytes) -> torch.Tensor:
        from compressors.walloc._codec import (
            decode_from_blob, from_model_output, resize_bicubic,
        )
        # `collate_encode` prepends a 4-byte <HH> (H, W) header that
        # `decode_blobs_batch` strips with blob[4:]. The throughput-path
        # consumer must strip it too or PIL fails to open the WebP
        # latent (RIFF magic at the wrong offset).
        walloc_blob = blob_bytes[4:]
        crop = CROP_FOR_TASK[self.task]
        x_hat = decode_from_blob(
            self._codec_dev, walloc_blob,
            self._info.latent_dim, self._info.latent_bits,
            self.device, torch.float32,
        )
        recon = from_model_output(x_hat).clamp(0, 1)
        recon = resize_bicubic(recon, (crop, crop)).clamp(0, 1)
        return (recon * 255.0).round().clamp(0, 255).to(torch.uint8).squeeze(0)

    def config_block(self) -> dict:
        return {
            "codec": "walloc:RGB_16x",
            "q_pixel_ratio": self.q,
            "latent_dim": self._info.latent_dim,
            "latent_bits": self._info.latent_bits,
            "J": self._info.J,
        }
