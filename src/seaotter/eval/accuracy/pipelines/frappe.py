"""FRAPPE-only (no transcode) — `frp` short, "S3" in iter-3.

Workers run a fork-inherited CPU FRAPPE encoder to produce JPEG-LS
latent blobs. Main loop unpacks them, stacks per-scale latents, and
runs the FRAPPE decoder batched on GPU.

Knob: op.value = n_ch (FRAPPE rate tier; iter-3/4 used {3, 6, 9, 12, 15}).
"""

from __future__ import annotations

import PIL.Image
import torch
from torchvision.transforms.v2.functional import pil_to_tensor

from ._base import Pipeline


class FrappePipeline(Pipeline):
    short = "frp"

    def __init__(self, *, op: dict, task: str, device: torch.device):
        super().__init__(op=op, task=task, device=device)
        self.n_ch = int(op["value"])
        from compressors.frappe.model import load_from_hub, load_progressive_model
        cfg, weights, n_trained = load_from_hub()
        if not 1 <= self.n_ch <= n_trained:
            raise ValueError(f"n_ch must be in [1, {n_trained}], got {self.n_ch}")
        self._cfg = cfg
        self._linear_input = bool(getattr(cfg, "linear_input", False))

        self._model_dev = load_progressive_model(weights, cfg, self.n_ch, device).eval()
        for p in self._model_dev.parameters():
            p.requires_grad_(False)
        # CPU model is fork-inherited by DataLoader workers.
        self._model_cpu = load_progressive_model(
            weights, cfg, self.n_ch, torch.device("cpu"),
        ).eval()
        for p in self._model_cpu.parameters():
            p.requires_grad_(False)

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
        latents = self._model_cpu.encode(x_in)
        latents = [z.round().clamp(-127, 127).to(torch.int8) for z in latents]
        return encode_latents(arrange_latents(latents))

    @torch.no_grad()
    def decode_blobs_batch(self, blobs):
        from compressors.frappe.entropy_coding import decode_latents, unarrange_latents
        scale_groups = self._model_dev.scale_groups
        # Per-blob unpack on CPU, then stack per-scale to batch on GPU.
        per_image_latents: list[list[torch.Tensor]] = []
        for blob in blobs:
            arranged = decode_latents(blob)
            latents_int8 = unarrange_latents(arranged, scale_groups)
            per_image_latents.append(latents_int8)
        # Transpose B × S → S × B and stack each scale.
        S = len(per_image_latents[0])
        stacked = []
        for s in range(S):
            batch_s = torch.cat([per_image_latents[i][s] for i in range(len(blobs))], dim=0)
            stacked.append(batch_s.to(self.device).to(torch.float32))
        rgb = self._model_dev.decode(stacked).clamp(-1.0, 1.0)
        rgb_u8 = ((rgb + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
        return rgb_u8, [len(b) for b in blobs], None

    @torch.no_grad()
    def decode_only_consumer(self, blob_bytes: bytes) -> torch.Tensor:
        from compressors.frappe.entropy_coding import decode_latents, unarrange_latents
        arranged = decode_latents(blob_bytes)
        latents_int8 = unarrange_latents(arranged, self._model_dev.scale_groups)
        latents = [z.to(torch.float32).to(self.device) for z in latents_int8]
        rgb = self._model_dev.decode(latents).clamp(-1.0, 1.0)
        return ((rgb + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8).squeeze(0)

    def config_block(self) -> dict:
        return {
            "codec": "frappe",
            "n_ch": self.n_ch,
            "linear_input": self._linear_input,
        }
