"""iter-11: CPU-only consumer-side throughput eval.

Re-measures the consumer-side cost of every (pipeline, op) cell with
**everything pinned to CPU** (no GPU at all -- consumer codec, teacher
forward, latent reshapes, all on CPU). Reuses the iter-6 pipeline
implementations as-is (with the WaLLoC header bug fixed) but:

  - forces ``device = torch.device("cpu")`` everywhere;
  - times the codec-only decode and the teacher-only forward
    separately, instead of conflating them as the iter-6 harness does;
  - keeps the iter-6 envelope schema so downstream figures /
    tables don't need to learn a new format.

Why a CPU-only re-measurement:

iter-6's consumer time ran the teacher on ``cuda:0`` while the
classical codecs (JPEG/AVIF/WebP/JP2) decoded on CPU and the neural
codecs (FRAPPE / WaLLoC / SEAOTTER) decoded on a workstation GPU
(RTX PRO 6000). On that asymmetric setup the 57M-param FRAPPE
decoder beat libavif software decode wall-clock, which is real but
hardware-dependent and misleading as a single "consumer cost"
number. Pinning everything to CPU gives a deployment-realistic
baseline for "no-GPU consumer" scenarios (browsers, IoT, edge
inference).

Usage::

    python eval_cpu_throughput.py \\
        --pipeline avif --task cls \\
        --op '{"type":"quality","value":25}' \\
        --out_json production/throughput_avif_cls_q25.json \\
        --n_images 32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import PIL.Image
import numpy as np
import torch


from .pipelines import make_pipeline  # noqa: E402
from .preprocessing import load_val, task_crop_pil  # noqa: E402
from .schema import (  # noqa: E402
    envelope_skeleton, parse_op, validate_op_for_pipeline,
)
from .teacher import load_teacher, teacher_logits_from_uint8  # noqa: E402

HARNESS_VERSION = "iter11-cpu-2"  # iter11-cpu-2: steady-state consumer split
                                   # (transcode untimed; decode_steady_state_consumer timed)

VAL_IMG_KEY = {"cls": "jpg", "seg": "image"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", required=True)
    p.add_argument("--task", choices=("cls", "seg"), required=True)
    p.add_argument("--op", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--n_images", type=int, default=32,
                   help="Per-cell image count (default 32; CPU-only "
                        "neural decode is ~1-3s/image so the default "
                        "trades distribution width for tractable wall-clock).")
    p.add_argument("--warmup", type=int, default=2,
                   help="Per-cell warm-up iterations before timing starts.")
    return p.parse_args()


def _stats(values: list[float]) -> dict:
    arr = np.asarray(values)
    return {
        "n": int(arr.size),
        "median_ms": float(np.median(arr) * 1000),
        "mean_ms": float(np.mean(arr) * 1000),
        "p25_ms": float(np.percentile(arr, 25) * 1000),
        "p75_ms": float(np.percentile(arr, 75) * 1000),
    }


def _load_images(task: str, n_images: int) -> list[PIL.Image.Image]:
    val = load_val(task)
    n = min(n_images, val.num_rows)
    sub = val.select(range(n))
    out = []
    for i in range(n):
        sample = sub[i]
        out.append(task_crop_pil(sample[VAL_IMG_KEY[task]], task))
    return out


@torch.no_grad()
def time_encodes(
    pipe, images: list[PIL.Image.Image], warmup: int,
) -> tuple[list[float], list[bytes]]:
    # warm-up (results discarded).
    for img in images[:warmup]:
        _ = pipe.encode_only_cpu(img)

    blobs = []
    times = []
    for img in images:
        t0 = time.perf_counter()
        blob = pipe.encode_only_cpu(img)
        times.append(time.perf_counter() - t0)
        blobs.append(blob)
    return times, blobs


@torch.no_grad()
def cloud_transcode(pipe, sensor_blobs: list[bytes]) -> list[bytes]:
    """Cloud-side one-time transcode (untimed). Maps each sensor uplink
    blob to its on-disk storage blob. For non-transcode pipelines this is
    identity (storage blob = sensor blob)."""
    return [pipe.transcode_to_storage_blob(b) for b in sensor_blobs]


@torch.no_grad()
def time_decodes_only(
    pipe, storage_blobs: list[bytes], warmup: int,
) -> tuple[list[float], list[torch.Tensor]]:
    """Time steady-state consumer-side decode only — no teacher forward.
    For SEAOTTER-family pipelines this is JPEG decode + F^{-1} (vanilla
    consumer-side cost). For other pipelines this is the standard
    decode_only_consumer path."""
    for blob in storage_blobs[:warmup]:
        _ = pipe.decode_steady_state_consumer(blob).unsqueeze(0)

    recons = []
    times = []
    for blob in storage_blobs:
        t0 = time.perf_counter()
        x_u8 = pipe.decode_steady_state_consumer(blob).unsqueeze(0)
        times.append(time.perf_counter() - t0)
        recons.append(x_u8)
    return times, recons


@torch.no_grad()
def time_teacher_only(
    teacher, recons: list[torch.Tensor], warmup: int,
) -> list[float]:
    """Time teacher forward only — given precomputed decoded tensors."""
    for x_u8 in recons[:warmup]:
        _ = teacher_logits_from_uint8(teacher, x_u8)

    times = []
    for x_u8 in recons:
        t0 = time.perf_counter()
        _ = teacher_logits_from_uint8(teacher, x_u8)
        times.append(time.perf_counter() - t0)
    return times


@torch.no_grad()
def time_consumer_combined(
    pipe, teacher, storage_blobs: list[bytes], warmup: int,
) -> list[float]:
    """Time codec-decode + teacher-forward as a single block, matching the
    iter-6 'consumer' field for backward-compat with downstream tooling.
    Uses the steady-state consumer path."""
    for blob in storage_blobs[:warmup]:
        x_u8 = pipe.decode_steady_state_consumer(blob).unsqueeze(0)
        _ = teacher_logits_from_uint8(teacher, x_u8)

    times = []
    for blob in storage_blobs:
        t0 = time.perf_counter()
        x_u8 = pipe.decode_steady_state_consumer(blob).unsqueeze(0)
        _ = teacher_logits_from_uint8(teacher, x_u8)
        times.append(time.perf_counter() - t0)
    return times


def main() -> None:
    args = parse_args()
    op = parse_op(args.op)
    validate_op_for_pipeline(args.pipeline, op)
    device = torch.device("cpu")  # hard-pinned for iter-11.

    env = envelope_skeleton(
        pipeline=args.pipeline, task=args.task, op=op, kind="throughput",
    )
    env["harness_version"] = HARNESS_VERSION

    print(f"[cpu-bench] pipeline={args.pipeline} task={args.task} "
          f"n={args.n_images} device={device}", flush=True)
    images = _load_images(args.task, args.n_images)
    pipe = make_pipeline(args.pipeline, op, args.task, device)
    teacher = load_teacher(args.task, device)

    print(f"  [encode] sensor encode (CPU) ...", flush=True)
    enc_times, sensor_blobs = time_encodes(pipe, images, args.warmup)
    enc_stats = _stats(enc_times)
    print(f"    encode    median={enc_stats['median_ms']:.2f} ms", flush=True)

    print(f"  [transcode] cloud-side transcode (untimed; ", flush=True)
    print(f"             identity for non-sandwich pipelines) ...", flush=True)
    storage_blobs = cloud_transcode(pipe, sensor_blobs)

    print(f"  [decode-only] codec steady-state consumer decode (CPU) ...",
          flush=True)
    dec_times, recons = time_decodes_only(pipe, storage_blobs, args.warmup)
    dec_stats = _stats(dec_times)
    print(f"    decode    median={dec_stats['median_ms']:.2f} ms", flush=True)

    print(f"  [teacher-only] teacher forward (CPU) ...", flush=True)
    teacher_times = time_teacher_only(teacher, recons, args.warmup)
    teacher_stats = _stats(teacher_times)
    print(f"    teacher   median={teacher_stats['median_ms']:.2f} ms", flush=True)

    print(f"  [consumer-combined] codec + teacher (CPU, single block) ...",
          flush=True)
    combined_times = time_consumer_combined(pipe, teacher, storage_blobs, args.warmup)
    combined_stats = _stats(combined_times)
    print(f"    combined  median={combined_stats['median_ms']:.2f} ms",
          flush=True)

    env["n_throughput_images"] = len(images)
    env["throughput"] = {
        "encode": enc_stats,
        "consumer": combined_stats,
        "consumer_decode_only": dec_stats,
        "teacher_only": teacher_stats,
    }
    env["config"] = {
        **pipe.config_block(),
        "cpu_model": "AMD EPYC 9354 32-Core Processor",
        "gpu_model": None,
        "device": "cpu",
        "threading": "natural (library defaults; no OMP/MKL/torch caps)",
        "warmup_iters": args.warmup,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(env, indent=2))
    print(f"[cpu-bench] WROTE {out_path}", flush=True)


if __name__ == "__main__":
    main()
