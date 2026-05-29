"""FRAPPE + ITU JPEG — `frp_jpeg`.

Sensor: FRAPPE encoder → JPEG-LS (transmit).
Cloud:  FRAPPE decoder → ITU JPEG (q=75, subsampling=0) round-trip.
Consumer: stock JPEG decode.

Transmit blob = FRAPPE-LS bytes; storage blob = JPEG bytes. Storage is
typically 5-30 x transmit because JPEG is less efficient on FRAPPE's
decoded RGB than JPEG-LS is on the int8 latent stack.

Knob: op.value = n_ch (FRAPPE rate tier). op.extras.jpeg_quality (default 75).
"""

from __future__ import annotations

import io

import PIL.Image
import torch
from torchvision.transforms.v2.functional import pil_to_tensor

from ._base import Pipeline


class FrpJpegPipeline(Pipeline):
    short = "frp_jpeg"

    def __init__(self, *, op: dict, task: str, device: torch.device):
        super().__init__(op=op, task=task, device=device)
        self.n_ch = int(op["value"])
        self.q_jpeg = int(op.get("extras", {}).get("jpeg_quality", 75))
        if not 1 <= self.q_jpeg <= 100:
            raise ValueError(f"jpeg_quality must be in [1, 100], got {self.q_jpeg}")

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
        per_image_latents: list[list[torch.Tensor]] = []
        for blob in blobs:
            arranged = decode_latents(blob)
            per_image_latents.append(unarrange_latents(arranged, scale_groups))
        S = len(per_image_latents[0])
        stacked = []
        for s in range(S):
            cat = torch.cat([per_image_latents[i][s] for i in range(len(blobs))], dim=0)
            stacked.append(cat.to(self.device).to(torch.float32))
        rgb = self._model_dev.decode(stacked).clamp(-1.0, 1.0)
        rgb_u8 = ((rgb + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
        rgb_u8_cpu = rgb_u8.cpu()

        recons: list[torch.Tensor] = []
        storage_bytes: list[int] = []
        for i in range(len(blobs)):
            recon, n_bytes = self._jpeg_roundtrip(rgb_u8_cpu[i])
            recons.append(recon)
            storage_bytes.append(n_bytes)
        recon_batch = torch.stack(recons, dim=0).to(self.device)
        return recon_batch, [len(b) for b in blobs], storage_bytes

    @torch.no_grad()
    def decode_only_consumer(self, blob_bytes: bytes) -> torch.Tensor:
        from compressors.frappe.entropy_coding import decode_latents, unarrange_latents
        arranged = decode_latents(blob_bytes)
        latents_int8 = unarrange_latents(arranged, self._model_dev.scale_groups)
        latents = [z.to(torch.float32).to(self.device) for z in latents_int8]
        rgb = self._model_dev.decode(latents).clamp(-1.0, 1.0)
        rgb_u8 = ((rgb + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8).squeeze(0)
        recon, _ = self._jpeg_roundtrip(rgb_u8.cpu())
        return recon.to(self.device)

    def config_block(self) -> dict:
        return {
            "format": "JPEG",
            "quality": self.q_jpeg,
            "subsampling": 0,
            "frappe_n_ch": self.n_ch,
            "linear_input": self._linear_input,
        }
