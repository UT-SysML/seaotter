"""iter-6 throughput entry point.

Usage:
    python eval_throughput.py --pipeline avif --task cls --n_images 256 \\
        --op '{"type":"quality","value":25}' \\
        --out_json production/throughput_avif_cls_q25.json

Measures per-image, single-threaded-from-the-app-perspective:
  - sensor encode (CPU-side bytes-out only)
  - consumer decode + ConvNeXt forward (CPU decode + GPU teacher fwd, bs=1)
on a 256-image subset of the task's validation split.

Natural threading (no OMP / MKL / torch caps) — matches the
FRAPPE / compressors encode-complexity harnesses.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import PIL.Image
import datasets
import numpy as np
import torch
from torchvision.transforms.v2.functional import pil_to_tensor


from .pipelines import make_pipeline  # noqa: E402
from .preprocessing import load_val, task_crop_pil  # noqa: E402
from .schema import (  # noqa: E402
    envelope_skeleton, parse_op, validate_op_for_pipeline,
)
from .teacher import load_teacher, teacher_logits_from_uint8  # noqa: E402


VAL_IMG_KEY = {"cls": "jpg", "seg": "image"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", required=True)
    p.add_argument("--task", choices=("cls", "seg"), required=True)
    p.add_argument("--op", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n_images", type=int, default=256)
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
def time_encodes(pipe, images: list[PIL.Image.Image]) -> tuple[list[float], list[bytes]]:
    blobs = []
    times = []
    for img in images:
        t0 = time.perf_counter()
        blob = pipe.encode_only_cpu(img)
        times.append(time.perf_counter() - t0)
        blobs.append(blob)
    return times, blobs


@torch.no_grad()
def time_consumers(
    pipe, blobs: list[bytes], task: str, device: torch.device,
) -> list[float]:
    teacher = load_teacher(task, device)
    # warm-up so first-call CUDA allocator / cuDNN heuristics don't pollute.
    for blob in blobs[:4]:
        x_u8 = pipe.decode_only_consumer(blob).unsqueeze(0)
        _ = teacher_logits_from_uint8(teacher, x_u8)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    out = []
    for blob in blobs:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        x_u8 = pipe.decode_only_consumer(blob).unsqueeze(0)
        _ = teacher_logits_from_uint8(teacher, x_u8)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        out.append(time.perf_counter() - t0)
    return out


def main() -> None:
    args = parse_args()
    op = parse_op(args.op)
    validate_op_for_pipeline(args.pipeline, op)
    device = torch.device(args.device)
    env = envelope_skeleton(
        pipeline=args.pipeline, task=args.task, op=op, kind="throughput",
    )

    print(f"[bench] pipeline={args.pipeline} task={args.task} n={args.n_images} "
          f"device={device}", flush=True)
    images = _load_images(args.task, args.n_images)

    pipe = make_pipeline(args.pipeline, op, args.task, device)

    print(f"  [encode] sensor encode ...", flush=True)
    enc_times, blobs = time_encodes(pipe, images)
    enc_stats = _stats(enc_times)
    print(f"    median={enc_stats['median_ms']:.2f}ms", flush=True)

    print(f"  [consumer] decode + teacher forward ...", flush=True)
    con_times = time_consumers(pipe, blobs, args.task, device)
    con_stats = _stats(con_times)
    print(f"    median={con_stats['median_ms']:.2f}ms", flush=True)

    env["n_throughput_images"] = len(images)
    env["throughput"] = {"encode": enc_stats, "consumer": con_stats}
    env["config"] = {
        **pipe.config_block(),
        "cpu_model": "AMD EPYC 9354 32-Core Processor",
        "gpu_model": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation",
        "device": str(device),
        "threading": "natural (library defaults; no OMP/MKL/torch caps)",
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(env, indent=2))
    print(f"[bench] WROTE {out_path}", flush=True)


if __name__ == "__main__":
    main()
