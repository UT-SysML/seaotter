"""SEA OTTER pipeline — `seab` (no fine-tuning) and `seaft` (fine-tuned).

Workers run a fork-inherited CPU FRAPPE encoder → JPEG-LS pack (= transmit).
Main loop unpacks → batched FRAPPE decoder on GPU → SEA OTTER sandwich →
ConvNeXt-ready uint8 RGB. Transmit bytes are the JPEG-LS blob length;
storage bytes are the per-image JPEG sandwich length.

Fine-tuned weights (`seaft`) come from the public pipeline bundle
`seaotter_{ft_task}_n{N}` on the Hugging Face hub at the pinned
`HF_REVISION`. `ft_task` defaults to the eval task; `extras.ft_task`
selects another task's snapshot for cross-task evaluation. A local .pth
checkpoint (fwd/proxy/inv/decoder state dicts) can still be named through
`extras.checkpoint`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import PIL.Image
import torch
from torchvision.transforms.v2.functional import pil_to_tensor

from .._pipeline import (  # noqa: E402
    PHASE2_K3_PATH_DEFAULT, PHASE2_K_DEFAULT,
    hf_spec, load_pipeline_weights, load_sandwich, strip_prefix,
)
from seaotter import jpeg_codec  # noqa: E402

from ._base import Pipeline


class SeaOtterPipeline(Pipeline):
    def __init__(self, *, short: str, op: dict, task: str, device: torch.device):
        super().__init__(op=op, task=task, device=device)
        if short not in ("seab", "seaft"):
            raise ValueError(short)
        self.short = short
        self.n_ch = int(op["value"])

        from compressors.frappe.model import load_from_hub, load_progressive_model
        cfg, weights, n_trained = load_from_hub()
        if not 1 <= self.n_ch <= n_trained:
            raise ValueError(f"n_ch must be in [1, {n_trained}], got {self.n_ch}")
        self._linear_input = bool(getattr(cfg, "linear_input", False))

        self._frappe_dev = load_progressive_model(weights, cfg, self.n_ch, device).eval()
        self._frappe_cpu = load_progressive_model(
            weights, cfg, self.n_ch, torch.device("cpu"),
        ).eval()
        for p in self._frappe_dev.parameters():
            p.requires_grad_(False)
        for p in self._frappe_cpu.parameters():
            p.requires_grad_(False)

        fwd, proxy, inv, meta = load_sandwich(
            PHASE2_K3_PATH_DEFAULT, PHASE2_K_DEFAULT, device,
        )
        self.fwd = fwd; self.proxy = proxy; self.inv = inv; self._sandwich_meta = meta

        self.checkpoint: str | None = None
        self.ft_task: str | None = None
        extras = op.get("extras", {})
        if short == "seaft" and "checkpoint" not in extras:
            self.ft_task = extras.get("ft_task", task)
            if self.ft_task not in ("cls", "seg", "clip"):
                raise ValueError(f"ft_task must be cls, seg or clip, got {self.ft_task!r}")
            subdir = f"seaotter_{self.ft_task}_n{self.n_ch}"
            self.checkpoint = hf_spec(subdir)
            cfg, w = load_pipeline_weights(subdir)
            if cfg["codec"] != "frappe" or int(cfg["frappe_n_ch"]) != self.n_ch:
                raise ValueError(f"{subdir}: codec={cfg['codec']} n_ch={cfg['frappe_n_ch']}")
            self.fwd.load_state_dict(strip_prefix(w, "fwd."))
            self.proxy.load_state_dict(strip_prefix(w, "proxy."))
            self.inv.load_state_dict(strip_prefix(w, "inv."))
            self._frappe_dev.decoder.load_state_dict(strip_prefix(w, "decoder."))
            # Workers still use the base encoder (encoder always frozen).
        elif short == "seaft":
            # `extras.checkpoint`: a local .pth checkpoint.
            self.checkpoint = extras["checkpoint"]
            ckpt = torch.load(self.checkpoint, map_location=device, weights_only=False)
            if "fwd_state_dict" in ckpt:
                self.fwd.load_state_dict(ckpt["fwd_state_dict"])
            if "proxy_state_dict" in ckpt:
                self.proxy.load_state_dict(ckpt["proxy_state_dict"])
            if "inv_state_dict" in ckpt:
                self.inv.load_state_dict(ckpt["inv_state_dict"])
            if "decoder_state_dict" in ckpt and ckpt["decoder_state_dict"] is not None:
                self._frappe_dev.decoder.load_state_dict(ckpt["decoder_state_dict"])
                # Workers still use the base encoder (encoder always frozen).

        self.fwd.eval(); self.proxy.eval(); self.inv.eval()
        self._qtable_int = (
            torch.clamp(self.proxy.qtable().detach().round(), 1, 255)
            .to(torch.int32).cpu()
        )

    def _frappe_input(self, x_float_m1_1: torch.Tensor) -> torch.Tensor:
        from compressors.frappe.quantize import srgb_to_linear
        return srgb_to_linear(x_float_m1_1) if self._linear_input else x_float_m1_1

    # ---- accuracy path -------------------------------------------------

    @torch.no_grad()
    def collate_encode(self, pil_img: PIL.Image.Image) -> bytes:
        from compressors.frappe.entropy_coding import arrange_latents, encode_latents
        x_u8 = pil_to_tensor(pil_img.convert("RGB")).unsqueeze(0)
        x = x_u8.to(torch.float32) / 127.5 - 1.0
        x_in = self._frappe_input(x)
        latents = self._frappe_cpu.encode(x_in)
        latents = [z.round().clamp(-127, 127).to(torch.int8) for z in latents]
        return encode_latents(arrange_latents(latents))

    @torch.no_grad()
    def decode_blobs_batch(self, blobs):
        from compressors.frappe.entropy_coding import decode_latents, unarrange_latents
        scale_groups = self._frappe_dev.scale_groups
        per_image_latents = []
        for blob in blobs:
            arranged = decode_latents(blob)
            per_image_latents.append(unarrange_latents(arranged, scale_groups))
        S = len(per_image_latents[0])
        stacked = []
        for s in range(S):
            cat = torch.cat([per_image_latents[i][s] for i in range(len(blobs))], dim=0)
            stacked.append(cat.to(self.device).to(torch.float32))
        rgb_decoded = self._frappe_dev.decode(stacked).clamp(-1.0, 1.0)
        rgb_codec_in = (rgb_decoded + 1.0) * 127.5

        z_codec_in_u8 = self.fwd.codec_input_uint8(rgb_codec_in)
        z_codec_in_cpu = z_codec_in_u8.cpu()
        z_decoded_cpu = torch.empty_like(z_codec_in_cpu)
        storage_bytes: list[int] = []
        for i in range(len(blobs)):
            jpeg_bytes = jpeg_codec.encode(
                z_codec_in_cpu[i], self._qtable_int, subsampling=0,
            )
            z_decoded_cpu[i] = jpeg_codec.decode(jpeg_bytes)
            storage_bytes.append(len(jpeg_bytes))
        z_decoded_float = z_decoded_cpu.to(self.device).to(torch.float32)
        rgb_hat_u8 = self.inv.output_uint8(z_decoded_float)
        return rgb_hat_u8, [len(b) for b in blobs], storage_bytes

    @torch.no_grad()
    def decode_only_consumer(self, blob_bytes: bytes) -> torch.Tensor:
        from compressors.frappe.entropy_coding import decode_latents, unarrange_latents
        arranged = decode_latents(blob_bytes)
        latents_int8 = unarrange_latents(arranged, self._frappe_dev.scale_groups)
        latents = [z.to(torch.float32).to(self.device) for z in latents_int8]
        rgb_decoded = self._frappe_dev.decode(latents).clamp(-1.0, 1.0)
        rgb_codec_in = (rgb_decoded + 1.0) * 127.5
        z_codec_in_u8 = self.fwd.codec_input_uint8(rgb_codec_in).squeeze(0).cpu()
        jpeg_bytes = jpeg_codec.encode(z_codec_in_u8, self._qtable_int, subsampling=0)
        z_codec_out_u8 = jpeg_codec.decode(jpeg_bytes)
        z_codec_out_float = z_codec_out_u8.to(torch.float32).unsqueeze(0).to(self.device)
        rgb_hat_u8 = self.inv.output_uint8(z_codec_out_float)
        return rgb_hat_u8.squeeze(0)

    @torch.no_grad()
    def transcode_to_storage_blob(self, sensor_blob: bytes) -> bytes:
        """Cloud-side one-time transcode: FRAPPE-LS latents → JPEG file bytes.
        Runs once per image at the cloud per the encode-once / decode-many
        architecture; not paid by the downstream consumer."""
        from compressors.frappe.entropy_coding import decode_latents, unarrange_latents
        arranged = decode_latents(sensor_blob)
        latents_int8 = unarrange_latents(arranged, self._frappe_dev.scale_groups)
        latents = [z.to(torch.float32).to(self.device) for z in latents_int8]
        rgb_decoded = self._frappe_dev.decode(latents).clamp(-1.0, 1.0)
        rgb_codec_in = (rgb_decoded + 1.0) * 127.5
        z_codec_in_u8 = self.fwd.codec_input_uint8(rgb_codec_in).squeeze(0).cpu()
        jpeg_bytes = jpeg_codec.encode(z_codec_in_u8, self._qtable_int, subsampling=0)
        return jpeg_bytes

    @torch.no_grad()
    def decode_steady_state_consumer(self, jpeg_bytes: bytes) -> torch.Tensor:
        """Deployed steady-state consumer-side decode: vanilla JPEG decode +
        F^{-1} (3x3 depthwise conv + companding + affine). Input is the
        on-disk JPEG bytes produced by ``transcode_to_storage_blob``."""
        z_codec_out_u8 = jpeg_codec.decode(jpeg_bytes)
        z_codec_out_float = z_codec_out_u8.to(torch.float32).unsqueeze(0).to(self.device)
        rgb_hat_u8 = self.inv.output_uint8(z_codec_out_float)
        return rgb_hat_u8.squeeze(0)

    def config_block(self) -> dict:
        return {
            "codec": "seaotter",
            "n_ch": self.n_ch,
            "fine_tuned": self.short == "seaft",
            "ft_task": self.ft_task,
            "checkpoint": self.checkpoint,
            "phase2_init": self._sandwich_meta.get("phase2_init"),
            "phase2_k": self._sandwich_meta.get("phase2_k"),
            "phase2_arch": self._sandwich_meta.get("arch"),
        }
