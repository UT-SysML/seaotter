"""Lossless reference pipeline.

A bit-exact uint8 round-trip via PNG. The reconstructed batch is
identical to what was fed into the encoder, so this pipeline reports
the teacher's accuracy ceiling on the chosen preprocessing
(squash 384x384 for cls, squash 512x512 for seg) with no codec loss
of any kind.

Op type is ``none`` with a single operating point ``value=0``.
"""

from __future__ import annotations

import torch

from ._vanilla import VanillaPilPipeline


class RawPipeline(VanillaPilPipeline):
    short = "raw"

    def __init__(self, *, op: dict, task: str, device: torch.device):
        super().__init__(op=op, task=task, device=device)

    def _save_kwargs(self) -> dict:
        return {"format": "PNG", "optimize": False, "compress_level": 1}

    def config_block(self) -> dict:
        return {"format": "PNG", "lossless": True}
