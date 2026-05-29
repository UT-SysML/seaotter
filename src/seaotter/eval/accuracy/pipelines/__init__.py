"""Pipeline registry. Lazy-imports per-pipeline modules so missing
optional deps (e.g. compressors.frappe) don't break unrelated pipelines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._base import Pipeline


def make_pipeline(short: str, op: dict, task: str, device) -> "Pipeline":
    if short == "avif" or short == "avifx":
        from .avif import AvifPipeline
        return AvifPipeline(short=short, op=op, task=task, device=device)
    if short == "raw":
        from .raw import RawPipeline
        return RawPipeline(op=op, task=task, device=device)
    if short == "jpeg":
        from .jpeg import JpegPipeline
        return JpegPipeline(op=op, task=task, device=device)
    if short == "jp2":
        from .jp2 import Jp2Pipeline
        return Jp2Pipeline(op=op, task=task, device=device)
    if short == "webp":
        from .webp import WebpPipeline
        return WebpPipeline(op=op, task=task, device=device)
    if short == "wal":
        from .walloc import WallocPipeline
        return WallocPipeline(op=op, task=task, device=device)
    if short == "frp":
        from .frappe import FrappePipeline
        return FrappePipeline(op=op, task=task, device=device)
    if short in ("seab", "seaft"):
        from .seaotter import SeaOtterPipeline
        return SeaOtterPipeline(short=short, op=op, task=task, device=device)
    if short == "sandbox":
        from .sandbox import SandboxPipeline
        return SandboxPipeline(op=op, task=task, device=device)
    if short == "walsand":
        from .sandbox import WalSandwichPipeline
        return WalSandwichPipeline(op=op, task=task, device=device)
    if short == "walft":
        from .walloc_ft import WalftPipeline
        return WalftPipeline(op=op, task=task, device=device)
    if short == "frp_jpeg":
        from .frp_jpeg import FrpJpegPipeline
        return FrpJpegPipeline(op=op, task=task, device=device)
    if short == "wal_jpeg":
        from .wal_jpeg import WalJpegPipeline
        return WalJpegPipeline(op=op, task=task, device=device)
    raise ValueError(f"unknown pipeline short-code {short!r}")
