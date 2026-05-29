"""Vanilla JPEG (subsampling=0, no `optimize=True`). Knob: op.value = quality."""

from __future__ import annotations

import torch

from ._vanilla import VanillaPilPipeline


class JpegPipeline(VanillaPilPipeline):
    short = "jpeg"

    def __init__(self, *, op: dict, task: str, device: torch.device):
        super().__init__(op=op, task=task, device=device)
        self.quality = int(op["value"])

    def _save_kwargs(self) -> dict:
        return {"format": "JPEG", "quality": self.quality, "subsampling": 0}

    def config_block(self) -> dict:
        return {"format": "JPEG", "quality": self.quality, "subsampling": 0}
