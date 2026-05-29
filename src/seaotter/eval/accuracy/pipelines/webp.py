"""Vanilla WebP (lossy). Knob: op.value = quality."""

from __future__ import annotations

import torch

from ._vanilla import VanillaPilPipeline


class WebpPipeline(VanillaPilPipeline):
    short = "webp"

    def __init__(self, *, op: dict, task: str, device: torch.device):
        super().__init__(op=op, task=task, device=device)
        self.quality = int(op["value"])

    def _save_kwargs(self) -> dict:
        return {"format": "WEBP", "quality": self.quality}

    def config_block(self) -> dict:
        return {"format": "WEBP", "quality": self.quality}
