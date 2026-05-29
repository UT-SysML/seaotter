"""Vanilla JPEG 2000. Knob: op.value = compression rate target."""

from __future__ import annotations

import torch

from ._vanilla import VanillaPilPipeline


class Jp2Pipeline(VanillaPilPipeline):
    short = "jp2"

    def __init__(self, *, op: dict, task: str, device: torch.device):
        super().__init__(op=op, task=task, device=device)
        self.rate = float(op["value"])

    def _save_kwargs(self) -> dict:
        return {
            "format": "JPEG2000",
            "quality_mode": "rates",
            "quality_layers": [self.rate],
        }

    def config_block(self) -> dict:
        return {"format": "JPEG2000", "rate": self.rate}
