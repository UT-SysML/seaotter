"""`walft` — WaLLoC + SEA OTTER sandwich, fine-tuned on the downstream task.

Companion to `harness/pipelines/walloc.py` (zero-shot `wal`) and
`harness/pipelines/sandbox.py::WalSandwichPipeline` (zero-shot `walsand`).
The walft pipeline loads an iter-7 checkpoint that contains a fine-tuned
WaLLoC decoder + sandwich (fwd / proxy / inv).

Encoder is always the frozen pretrained WaLLoC encoder (asymmetric premise).
Transmit blob = WaLLoC WebP-lossless latent bytes (same as `wal`/`walsand`).
Storage bytes = JPEG sandwich bytes.

Op type: ``pixel_ratio``. op_id: ``p{value}``.
Checkpoint path: ``experiments/iter7_walloc_finetune/production/checkpoint_iter7_{task}_walft_p{P}.pth``.
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import PIL.Image
import torch
from torchvision.transforms.v2.functional import pil_to_tensor

P4_ROOT = Path("/home/dgj335/danjacobellis/seaotter/pre_trained_convnext")

from .._pipeline import (  # noqa: E402
    PHASE2_K3_PATH_DEFAULT, PHASE2_K_DEFAULT, load_sandwich,
)
from seaotter import jpeg_codec  # noqa: E402

from ._base import Pipeline


ITER7_PROD = (
    P4_ROOT / "experiments/iter7_walloc_finetune/production"
)


def _walft_checkpoint_path(task: str, pixel_ratio: int) -> str:
    """Default iter-7 production path. Overridable via the
    `WALFT_CHECKPOINT_OVERRIDE` env var (iter-10 sweeps and any future
    walft hyperparameter search use this to point at per-cell checkpoints
    without editing the harness)."""
    override = os.environ.get("WALFT_CHECKPOINT_OVERRIDE")
    if override:
        return override
    return str(ITER7_PROD / f"checkpoint_iter7_{task}_walft_p{int(pixel_ratio)}.pth")


class WalftPipeline(Pipeline):
    short = "walft"

    def __init__(self, *, op: dict, task: str, device: torch.device):
        super().__init__(op=op, task=task, device=device)
        self.q = float(op["value"])
        if not 0 < self.q <= 100:
            raise ValueError(f"walft: pixel_ratio must be in (0, 100], got {self.q}")

        from compressors.walloc._codec import load_codec, SNAP_MULTIPLE, snap_shape
        self._SNAP_MULTIPLE = SNAP_MULTIPLE
        self._snap_shape = snap_shape

        # GPU codec is hot-overridden from the iter-7 checkpoint (decoder).
        self._codec_dev, self._info = load_codec(
            device=device, torch_dtype=torch.float32,
        )
        # CPU codec is the stock pretrained encoder — frozen, fork-inherited
        # by DataLoader workers. The decoder fine-tune is NOT applied to the
        # CPU copy (workers only need the encoder).
        self._codec_cpu, _ = load_codec(
            device=torch.device("cpu"), torch_dtype=torch.float32,
        )

        # Stock sandwich (later overwritten by checkpoint state).
        fwd, proxy, inv, meta = load_sandwich(
            PHASE2_K3_PATH_DEFAULT, PHASE2_K_DEFAULT, device,
        )
        self.fwd = fwd; self.proxy = proxy; self.inv = inv
        self._sandwich_meta = meta

        # Load iter-7 fine-tune. `op.extras.checkpoint` lets a caller point at
        # a non-canonical checkpoint (e.g., a clip-trained walloc run from the
        # clip_production sweep), mirroring the seaft pipeline.
        self.checkpoint = (
            op.get("extras", {}).get("checkpoint")
            or _walft_checkpoint_path(task, self.q)
        )
        ckpt = torch.load(self.checkpoint, map_location=device, weights_only=False)
        if "fwd_state_dict" in ckpt:
            self.fwd.load_state_dict(ckpt["fwd_state_dict"])
        if "proxy_state_dict" in ckpt:
            self.proxy.load_state_dict(ckpt["proxy_state_dict"])
        if "inv_state_dict" in ckpt:
            self.inv.load_state_dict(ckpt["inv_state_dict"])
        # iter-7 ckpts use "walloc_decoder_state_dict"; the consolidated trainer
        # in src/seaotter/train/ uses unified "decoder_state_dict" (keyed by the
        # saved "codec" field). Accept either.
        decoder_sd = ckpt.get("decoder_state_dict") or ckpt.get("walloc_decoder_state_dict")
        if decoder_sd is None:
            raise RuntimeError(
                f"walft checkpoint {self.checkpoint} missing decoder weights "
                "(tried 'decoder_state_dict' and 'walloc_decoder_state_dict')"
            )
        self._codec_dev.decoder.load_state_dict(decoder_sd)

        self._codec_dev.eval()
        self._codec_cpu.eval()
        for p in self._codec_dev.parameters():
            p.requires_grad_(False)
        self.fwd.eval(); self.proxy.eval(); self.inv.eval()
        self._qtable_int = (
            torch.clamp(self.proxy.qtable().detach().round(), 1, 255)
            .to(torch.int32).cpu()
        )

    def _snap_hw(self, h: int, w: int) -> tuple[int, int]:
        if h % self._SNAP_MULTIPLE or w % self._SNAP_MULTIPLE:
            raise ValueError(f"walft: H, W must be multiples of 16, got {h}x{w}")
        return self._snap_shape(h, w, self.q)

    # ---- accuracy path -------------------------------------------------

    @torch.no_grad()
    def collate_encode(self, pil_img: PIL.Image.Image) -> bytes:
        """Sensor-side blob = WaLLoC's WebP-lossless latent bytes (the
        transmission unit). Identical to `wal`/`walsand` encoder path."""
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
        # 4-byte header: little-endian uint16 H, uint16 W. Lets the decoder
        # resize the recon back to the original naflex shape for the clip task
        # (where SigLIP-2 expects the variable per-image shape). Negligible
        # overhead (4 bytes per image, < 0.0001 bpp at 384²).
        header = struct.pack("<HH", H, W)
        return header + latent_bytes

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
            # Strip the 4-byte (H, W) header (see collate_encode).
            H_orig, W_orig = struct.unpack("<HH", blob[:4])
            walloc_blob = blob[4:]
            # Stage 1: hot WaLLoC decoder (fine-tuned).
            x_hat = decode_from_blob(
                self._codec_dev, walloc_blob,
                self._info.latent_dim, self._info.latent_bits,
                self.device, torch.float32,
            )
            recon_01 = from_model_output(x_hat).clamp(0, 1)
            if crop is not None:
                # cls/seg: square-crop for the downstream ConvNeXt input.
                recon_01 = resize_bicubic(recon_01, (crop, crop)).clamp(0, 1)
            else:
                # clip: upsample back to the original naflex (H, W) so SigLIP-2
                # sees the same per-image patch grid as the clean branch.
                recon_01 = resize_bicubic(recon_01, (H_orig, W_orig)).clamp(0, 1)
            x_u8 = (recon_01 * 255.0).round().clamp(0, 255)  # float [0,255]

            # Stage 2: SEA OTTER sandwich (fine-tuned fwd / proxy / inv).
            z_codec_in_u8 = self.fwd.codec_input_uint8(x_u8)
            z_codec_in_cpu = z_codec_in_u8.squeeze(0).cpu()
            jpeg_bytes = jpeg_codec.encode(z_codec_in_cpu, self._qtable_int, subsampling=0)
            z_codec_out_u8 = jpeg_codec.decode(jpeg_bytes)
            z_float = z_codec_out_u8.to(torch.float32).unsqueeze(0).to(self.device)
            rgb_hat_u8 = self.inv.output_uint8(z_float).squeeze(0)
            recons.append(rgb_hat_u8)
            storage_bytes.append(len(jpeg_bytes))
            # transmit_bytes excludes the 4-byte header (not part of the
            # actual transmission unit; same accounting as `wal`/`walsand`).
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
        z_codec_in_u8 = self.fwd.codec_input_uint8(x_u8)
        z_codec_in_cpu = z_codec_in_u8.squeeze(0).cpu()
        jpeg_bytes = jpeg_codec.encode(z_codec_in_cpu, self._qtable_int, subsampling=0)
        return jpeg_bytes

    @torch.no_grad()
    def decode_steady_state_consumer(self, jpeg_bytes: bytes) -> torch.Tensor:
        """Deployed steady-state consumer-side decode: JPEG decode + F^{-1}."""
        z_codec_out_u8 = jpeg_codec.decode(jpeg_bytes)
        z_float = z_codec_out_u8.to(torch.float32).unsqueeze(0).to(self.device)
        rgb_hat_u8 = self.inv.output_uint8(z_float)
        return rgb_hat_u8.squeeze(0)

    def config_block(self) -> dict:
        return {
            "codec": "walft",
            "q_pixel_ratio": self.q,
            "checkpoint": self.checkpoint,
            "phase2_init": self._sandwich_meta.get("phase2_init"),
            "phase2_k": self._sandwich_meta.get("phase2_k"),
            "phase2_arch": self._sandwich_meta.get("arch"),
            "walloc_latent_dim": self._info.latent_dim,
            "walloc_latent_bits": self._info.latent_bits,
        }
