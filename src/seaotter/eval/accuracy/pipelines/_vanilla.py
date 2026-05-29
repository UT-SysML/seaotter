"""Shared vanilla-codec base: PIL save → bytes, PIL open → uint8 tensor.

Subclasses override `_save_kwargs()` (passed to `PIL.Image.save`).
"""

from __future__ import annotations

import io

import PIL.Image
import torch
from torchvision.transforms.v2.functional import pil_to_tensor

from ._base import Pipeline


class VanillaPilPipeline(Pipeline):
    """For codecs where PIL is the whole pipeline (JPEG / WebP / JP2 / AVIF)."""

    def _save_kwargs(self) -> dict:
        raise NotImplementedError

    @torch.no_grad()
    def collate_encode(self, pil_img: PIL.Image.Image) -> bytes:
        buf = io.BytesIO()
        pil_img.save(buf, **self._save_kwargs())
        return buf.getvalue()

    def _decode_to_tensor(self, blob: bytes) -> torch.Tensor:
        dec = PIL.Image.open(io.BytesIO(blob))
        dec.load()
        return pil_to_tensor(dec.convert("RGB"))

    @torch.no_grad()
    def decode_blobs_batch(self, blobs):
        recons = [self._decode_to_tensor(b) for b in blobs]
        recon = torch.stack(recons, dim=0).to(self.device)
        return recon, [len(b) for b in blobs], None

    @torch.no_grad()
    def decode_only_consumer(self, blob_bytes: bytes) -> torch.Tensor:
        return self._decode_to_tensor(blob_bytes).to(self.device)
