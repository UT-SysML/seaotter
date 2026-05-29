"""Vanilla AVIF.

Two pipeline shorts:
  - `avif`  : libavif default speed (~6).
  - `avifx` : libavif speed=10 (fastest profile).

Knob: op.value = quality. For `avifx`, op.extras.avif_speed = 10.
"""

from __future__ import annotations

import torch

from ._vanilla import VanillaPilPipeline


class AvifPipeline(VanillaPilPipeline):
    def __init__(self, *, short: str, op: dict, task: str, device: torch.device):
        super().__init__(op=op, task=task, device=device)
        if short not in ("avif", "avifx"):
            raise ValueError(short)
        self.short = short
        self.quality = int(op["value"])
        self.avif_speed = (
            int(op["extras"]["avif_speed"]) if short == "avifx" else None
        )

    def _save_kwargs(self) -> dict:
        kw = {"format": "AVIF", "quality": self.quality}
        if self.avif_speed is not None:
            kw["speed"] = self.avif_speed
        return kw

    def config_block(self) -> dict:
        return {"format": "AVIF", "quality": self.quality, "avif_speed": self.avif_speed}
