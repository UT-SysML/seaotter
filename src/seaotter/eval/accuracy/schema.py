"""JSON envelope + per-pipeline operating-point validation.

Every output JSON (accuracy or throughput) shares one envelope so the
aggregator can index across pipelines uniformly. See the iter-6 prompt
"Standardized JSON envelope" section.
"""

from __future__ import annotations

import json
from typing import Any

from . import HARNESS_VERSION

# Pipeline short-codes registered for iter-6.
PIPELINES = ("raw", "jpeg", "jp2", "webp", "avif", "avifx", "wal", "frp", "seab", "seaft",
             "sandbox", "walsand", "walft", "frp_jpeg", "wal_jpeg")

PIPELINE_LABEL = {
    "raw":      "Lossless reference (no codec)",
    "jpeg":     "Vanilla JPEG (subsampling=0)",
    "jp2":      "Vanilla JPEG 2000",
    "webp":     "Vanilla WebP",
    "avif":     "Vanilla AVIF (libavif default speed)",
    "avifx":    "Vanilla AVIF (libavif speed=10)",
    "wal":      "WaLLoC RGB_16x",
    "frp":      "FRAPPE-only (no transcode)",
    "seab":     "SEA OTTER (no fine-tuning)",
    "seaft":    "SEA OTTER (fine-tuned)",
    "sandbox":  "Sandwich-only (raw input → SEA OTTER sandwich)",
    "walsand":  "WaLLoC + SEA OTTER sandwich",
    "walft":    "WaLLoC + SEA OTTER (fine-tuned)",
    "frp_jpeg": "FRAPPE + ITU JPEG (subsampling=0, q=75)",
    "wal_jpeg": "WaLLoC + ITU JPEG (subsampling=0, q=75)",
}

# op-point type expected per pipeline. avifx pins extras.avif_speed=10.
OP_TYPE = {
    "raw":      "none",
    "jpeg":     "quality",
    "jp2":      "rate",
    "webp":     "quality",
    "avif":     "quality",
    "avifx":    "quality",
    "wal":      "pixel_ratio",
    "frp":      "n_ch",
    "seab":     "n_ch",
    "seaft":    "n_ch",
    "sandbox":  "phase2_k",
    "walsand":  "pixel_ratio",
    "walft":    "pixel_ratio",
    "frp_jpeg": "n_ch",
    "wal_jpeg": "pixel_ratio",
}

VAL_DS = {
    "cls": ("timm/imagenet-1k-wds", "validation"),
    "seg": ("danjacobellis/scene_parse_150", "validation"),
    "clip": ("timm/imagenet-1k-wds", "validation"),
}

PREPROCESSING = {
    "cls": "squash 384x384",
    "seg": "squash 512x512",
    "clip": "naflex max_num_patches=256 snap=32",
}
# Sentinel: clip has variable per-image (H', W') determined by naflex_resize.
# Any code that reads CROP_FOR_TASK[task] for clip must dispatch through
# preprocessing.naflex_resize() and NOT assume a fixed crop.
CROP_FOR_TASK = {"cls": 384, "seg": 512, "clip": None}

# Patch budget for naflex (per-image upper bound on (H'/16)*(W'/16)).
# Recorded in the envelope so eval JSONs are unambiguous.
CLIP_NAFLEX_MAX_PATCHES = 256
CLIP_NAFLEX_PATCH_SIZE = 16
CLIP_NAFLEX_SNAP = 32  # FRAPPE max_ps=32 forces multiples-of-32 codec shapes.


def parse_op(op_json: str) -> dict:
    op = json.loads(op_json) if isinstance(op_json, str) else op_json
    if "type" not in op or "value" not in op:
        raise ValueError(f"op must have type+value, got {op}")
    op.setdefault("extras", {})
    return op


def validate_op_for_pipeline(pipeline: str, op: dict) -> None:
    if pipeline not in PIPELINES:
        raise ValueError(f"unknown pipeline {pipeline!r}; expected one of {PIPELINES}")
    expected = OP_TYPE[pipeline]
    if op["type"] != expected:
        raise ValueError(
            f"pipeline={pipeline} expects op.type={expected!r}, got {op['type']!r}"
        )
    if pipeline == "avifx":
        if op["extras"].get("avif_speed") != 10:
            raise ValueError("avifx requires extras.avif_speed=10")


def op_id(pipeline: str, op: dict) -> str:
    """Short filename slug for an operating point."""
    v = op["value"]
    if pipeline == "raw":
        return "ref"
    if pipeline == "jp2":
        return f"r{v}"
    if pipeline in ("wal", "walsand", "walft", "wal_jpeg"):
        return f"p{v}"
    if pipeline in ("frp", "seab", "seaft", "frp_jpeg"):
        return f"n{v}"
    if pipeline == "sandbox":
        return f"k{v}"
    if pipeline == "avifx":
        return f"q{v}_s{op['extras'].get('avif_speed', 10)}"
    return f"q{v}"


def envelope_skeleton(
    *, pipeline: str, task: str, op: dict, kind: str
) -> dict[str, Any]:
    """Construct an empty envelope; the caller fills metrics / throughput."""
    if kind not in ("accuracy", "throughput"):
        raise ValueError(kind)
    ds_name, ds_split = VAL_DS[task]
    env: dict[str, Any] = {
        "harness_version": HARNESS_VERSION,
        "pipeline": pipeline,
        "pipeline_label": PIPELINE_LABEL[pipeline],
        "task": task,
        "val_ds": ds_name,
        "val_split": ds_split,
        "preprocessing": PREPROCESSING[task],
        "operating_point": op,
        "transmit_bpp_mean": None,
        "storage_bpp_mean": None,
        "n_eval": None,
        "n_throughput_images": None,
        "metrics": None,
        "throughput": None,
        "config": {},
    }
    if task == "clip":
        env["clip_naflex_max_patches"] = CLIP_NAFLEX_MAX_PATCHES
        env["clip_naflex_patch_size"] = CLIP_NAFLEX_PATCH_SIZE
        env["clip_naflex_snap"] = CLIP_NAFLEX_SNAP
    return env
