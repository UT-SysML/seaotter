"""Standalone-codec ImageNet cls eval (codec-kodak-cls-v1).

Companion to `run_eval.py` (Kodak-native). Same 17 codec operating points
evaluated on ImageNet val 50k under the cls protocol:
ConvNeXt-Tiny `convnext_tiny.in12k_ft_in1k_384` top-1/top-5 + piq distortion
metrics + bpp. Resize: PIL bicubic to (384, 384). bpp denominator pinned to
384*384 = 147456. Single-GPU.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import datasets
import numpy as np
import piq
import torch
from PIL import Image
from torchvision.transforms.v2.functional import pil_to_tensor

# Workaround: torch 2.12+cu130 reports cuDNN 9.2 but the runtime emits
# CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH on conv2d. Disable cuDNN globally
# for inference; ConvNeXt-Tiny on the native cuDNN-free path is fast enough.
torch.backends.cudnn.enabled = False

# Reuse iter-6 harness helpers.
HARNESS_DIR = Path(
    "/home/dgj335/danjacobellis/seaotter/pre_trained_convnext/"
    "experiments/iter6_extra_codec_baselines"
)
sys.path.insert(0, str(HARNESS_DIR))
from harness.metrics import DistortionAccumulator, db_from_metric  # noqa: E402
from harness.preprocessing import squash_resize  # noqa: E402
# Skip harness.teacher: it pulls in pipeline.py → compressors (not installed).
# Load the cls teacher inline with timm directly.
import timm  # noqa: E402
from gigatorch import normalize as gt_normalize  # noqa: E402

from seaotter import load_from_hub  # noqa: E402


def load_cls_teacher(device: torch.device) -> torch.nn.Module:
    m = timm.create_model("convnext_tiny.in12k_ft_in1k_384", pretrained=True)
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()
    m.to(device)
    return m


def teacher_logits_from_uint8(teacher: torch.nn.Module, x_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 [0,255] (B,3,H,W) → in1k-timm-normalized → teacher logits."""
    x_m1_1 = x_uint8.to(torch.float32) / 127.5 - 1.0
    return teacher(gt_normalize.in1k_timm(x_m1_1))

# Local helpers (encode/decode pillow JPEG) — reused from run_eval.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eval import encode_pillow_jpeg, decode_pillow_jpeg  # noqa: E402


HARNESS_VERSION = "codec-kodak-cls-v1"
CROP = 384
BPP_DENOM = CROP * CROP  # 147456
TEACHER_NAME = "convnext_tiny.in12k_ft_in1k_384"
VAL_DS = "timm/imagenet-1k-wds"
VAL_SPLIT = "validation"
N_EVAL = 50_000

SUMMARY_PATH = Path(
    "/home/dgj335/danjacobellis/seaotter/experiments/"
    "codec_kodak_eval/results/summary.json"
)

# Module-global SEA OTTER bundle; populated lazily in main() and inherited
# by DataLoader workers via fork (Linux default start method).
_BUNDLE = None
# Module-global current cell spec for the per-worker collate.
_CELL = None


# ---------------------------------------------------------------------------
# Cell definitions
# ---------------------------------------------------------------------------

def _q_ladder() -> dict[str, int]:
    with open(SUMMARY_PATH) as f:
        s = json.load(f)
    return s["q_ladder"]


def build_cells() -> list[dict]:
    """17 cells in order: jpeg (7) → jpeg_sub0 (7) → seaotter (3).

    Each cell is a self-contained spec: pipeline, op, filename suffix.
    """
    qs = _q_ladder()
    q_names = ["q0p5", "q1", "q1p5", "q2", "q2p5", "q3", "q3p5"]
    cells: list[dict] = []
    for q_name in q_names:
        cells.append({
            "pipeline": "jpeg",
            "pipeline_label": "Pillow JPEG (subsampling=2, 4:2:0)",
            "q_name": q_name,
            "quality": int(qs[q_name]),
            "subsampling": 2,
            "filename": f"eval_jpeg_cls_kodak_{q_name}.json",
        })
    for q_name in q_names:
        cells.append({
            "pipeline": "jpeg_sub0",
            "pipeline_label": "Pillow JPEG (subsampling=0, 4:4:4)",
            "q_name": q_name,
            "quality": int(qs[q_name]),
            "subsampling": 0,
            "filename": f"eval_jpeg_sub0_cls_kodak_{q_name}.json",
        })
    for k in range(3):
        cells.append({
            "pipeline": "seaotter",
            "pipeline_label": f"SEA OTTER S3 K=3, k={k}",
            "seaotter_k": k,
            "filename": f"eval_seaotter_cls_kodak_k{k}.json",
        })
    assert len(cells) == 17, len(cells)
    return cells


# ---------------------------------------------------------------------------
# Per-cell DataLoader plumbing
# ---------------------------------------------------------------------------

def _codec_roundtrip(pil_384: Image.Image, x_uint8: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Return (recon_uint8 [3,384,384], n_bytes) for the active cell."""
    cell = _CELL
    if cell["pipeline"] in ("jpeg", "jpeg_sub0"):
        blob = encode_pillow_jpeg(pil_384, cell["quality"], cell["subsampling"])
        recon = decode_pillow_jpeg(blob)
    else:
        # seaotter
        k = cell["seaotter_k"]
        blob = _BUNDLE.encode(x_uint8, k=k)
        recon = _BUNDLE.decode(blob)
    return recon, len(blob)


def _collate_cell(batch):
    """Worker-side: PIL → squash-384 → uint8 → codec round-trip per image."""
    refs, recons, blob_lens, labels = [], [], [], []
    for sample in batch:
        pil = squash_resize(sample["jpg"], CROP)
        x = pil_to_tensor(pil)  # (3, 384, 384) uint8
        recon, nb = _codec_roundtrip(pil, x)
        refs.append(x)
        recons.append(recon)
        blob_lens.append(nb)
        labels.append(int(sample["cls"]))
    return (
        torch.stack(refs, 0),
        torch.stack(recons, 0),
        torch.tensor(blob_lens, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def _collate_anchor(batch):
    """Anchor pass: no codec — return refs and labels."""
    refs, labels = [], []
    for sample in batch:
        pil = squash_resize(sample["jpg"], CROP)
        refs.append(pil_to_tensor(pil))
        labels.append(int(sample["cls"]))
    return torch.stack(refs, 0), torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------------------
# Atomic JSON write
# ---------------------------------------------------------------------------

def write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Eval drivers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_anchor(args, val_ds, device) -> dict:
    teacher = _TEACHER_CACHE[device]
    n = min(args.n_eval, val_ds.num_rows)
    sub = val_ds.select(range(n))
    loader = torch.utils.data.DataLoader(
        sub, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, collate_fn=_collate_anchor,
        persistent_workers=(args.num_workers > 0),
    )
    correct1 = correct5 = seen = 0
    t0 = time.time()
    n_batches = (n + args.batch_size - 1) // args.batch_size
    for bi, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = teacher_logits_from_uint8(teacher, x)
        top1 = logits.argmax(dim=1)
        top5 = logits.topk(5, dim=1).indices
        correct1 += int((top1 == y).sum().item())
        correct5 += int((top5 == y.unsqueeze(1)).any(dim=1).sum().item())
        seen += y.numel()
        if (bi + 1) % 100 == 0 or bi == n_batches - 1:
            elapsed = time.time() - t0
            print(
                f"  anchor {seen}/{n} top1={correct1/seen*100:.3f}% "
                f"({elapsed:.0f}s)", flush=True,
            )
    return {
        "top1": correct1 / seen,
        "top5": correct5 / seen,
        "n_seen": seen,
        "elapsed_s": time.time() - t0,
    }


@torch.no_grad()
def evaluate_cell(args, cell, val_ds, device) -> dict:
    """Stream the val set through the codec round-trip + teacher + piq metrics.

    Reuses one teacher / LPIPS / DISTS net across cells via the
    `_ACCUMULATOR_CACHE` / `_TEACHER_CACHE` globals.
    """
    teacher = _TEACHER_CACHE[device]
    dist = DistortionAccumulator(device=device)
    global _CELL
    _CELL = cell

    n = min(args.n_eval, val_ds.num_rows)
    sub = val_ds.select(range(n))
    loader = torch.utils.data.DataLoader(
        sub, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, collate_fn=_collate_cell,
        persistent_workers=(args.num_workers > 0),
    )

    correct1 = correct5 = seen = 0
    total_bytes = []
    t0 = time.time()
    n_batches = (n + args.batch_size - 1) // args.batch_size
    for bi, (ref, recon, blob_lens, y) in enumerate(loader):
        ref = ref.to(device, non_blocking=True)
        recon = recon.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = teacher_logits_from_uint8(teacher, recon)
        top1 = logits.argmax(dim=1)
        top5 = logits.topk(5, dim=1).indices
        correct1 += int((top1 == y).sum().item())
        correct5 += int((top5 == y.unsqueeze(1)).any(dim=1).sum().item())
        seen += y.numel()

        dist.update(recon, ref)
        total_bytes.extend(int(v) for v in blob_lens.tolist())

        if (bi + 1) % 100 == 0 or bi == n_batches - 1:
            elapsed = time.time() - t0
            bpp_now = float(np.mean(total_bytes)) * 8.0 / BPP_DENOM
            print(
                f"  {cell['filename']}: {seen}/{n} "
                f"top1={correct1/seen*100:.3f}% bpp={bpp_now:.4f} "
                f"({elapsed:.0f}s)", flush=True,
            )

    elapsed = time.time() - t0
    bpp_arr = np.asarray(total_bytes, dtype=np.float64) * 8.0 / BPP_DENOM
    bpp_mean = float(bpp_arr.mean())
    bpp_std = float(bpp_arr.std())
    d = dist.finalize()
    return {
        "n_seen": seen,
        "bpp_mean": bpp_mean,
        "bpp_std": bpp_std,
        "top1": correct1 / seen,
        "top5": correct5 / seen,
        "psnr_db": d["psnr_db"],
        "ssim": d["ssim"],
        "lpips_db": d["lpips_db"],
        "dists_db": d["dists_db"],
        "elapsed_s": elapsed,
    }


def envelope_for_cell(cell: dict, result: dict, device: torch.device) -> dict:
    pipeline = cell["pipeline"]
    if pipeline == "seaotter":
        op = {"type": "seaotter_k", "value": cell["seaotter_k"]}
        cfg_quality = None
        cfg_seaotter_k = cell["seaotter_k"]
        cfg_subsampling = 0
    else:
        op = {
            "type": "jpeg_q",
            "value": cell["quality"],
            "ladder_id": cell["q_name"],
        }
        cfg_quality = cell["quality"]
        cfg_seaotter_k = None
        cfg_subsampling = cell["subsampling"]

    metrics = {
        "bpp_mean": result["bpp_mean"],
        "bpp_std": result["bpp_std"],
        "top1": result["top1"],
        "top5": result["top5"],
        "psnr_db": result["psnr_db"],
        "ssim": result["ssim"],
        "lpips_db": result["lpips_db"],
        "dists_db": result["dists_db"],
        "miou": None,
        "pixel_accuracy": None,
        "elapsed_s": result["elapsed_s"],
    }
    return {
        "harness_version": HARNESS_VERSION,
        "pipeline": pipeline,
        "pipeline_label": cell["pipeline_label"],
        "task": "cls",
        "val_ds": VAL_DS,
        "val_split": VAL_SPLIT,
        "preprocessing": "squash 384x384",
        "operating_point": op,
        "n_eval": result["n_seen"],
        "transmit_bpp_mean": result["bpp_mean"],
        "storage_bpp_mean": result["bpp_mean"],
        "metrics": metrics,
        "config": {
            "codec": pipeline,
            "subsampling": cfg_subsampling,
            "quality": cfg_quality,
            "seaotter_k": cfg_seaotter_k,
            "seaotter_source": (
                "danjacobellis/seaotter @ seaotter_jpeg_s3 "
                "(load_from_hub default)"
            ) if pipeline == "seaotter" else None,
            "seaotter_lambdas": (
                [0.75, 0.4, 0.22] if pipeline == "seaotter" else None
            ),
            "teacher": TEACHER_NAME,
            "device": str(device),
            "bpp_denominator": BPP_DENOM,
            "resize": "bicubic 384x384",
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_TEACHER_CACHE: dict[torch.device, torch.nn.Module] = {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out_dir",
        default=(
            "/home/dgj335/danjacobellis/seaotter/experiments/"
            "codec_kodak_eval/cls_results"
        ),
    )
    p.add_argument(
        "--cell_indices",
        default=None,
        help="Comma-separated subset, e.g. '0,3,6'. Default = all 17.",
    )
    p.add_argument(
        "--anchor_only", action="store_true",
        help="Run raw (no-codec) cls anchor and exit. No 17-cell sweep.",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--n_eval", type=int, default=N_EVAL)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"[init] device={device} bs={args.batch_size} nw={args.num_workers}", flush=True)

    # Load SEA OTTER bundle once (CPU); inherited by workers via fork.
    global _BUNDLE
    print("[init] Loading SEA OTTER bundle from HF (default S3)...", flush=True)
    _BUNDLE = load_from_hub()
    assert _BUNDLE.lambdas == [0.75, 0.4, 0.22], _BUNDLE.lambdas
    assert _BUNDLE.K == 3

    # Load + cache the cls teacher.
    print("[init] Loading cls teacher (convnext_tiny.in12k_ft_in1k_384)...", flush=True)
    _TEACHER_CACHE[device] = load_cls_teacher(device)

    # Load val dataset once.
    print(f"[init] Loading {VAL_DS} {VAL_SPLIT}...", flush=True)
    val_ds = datasets.load_dataset(VAL_DS, split=VAL_SPLIT)
    print(f"[init] val rows = {val_ds.num_rows}", flush=True)

    # ----------------------------------------------------------------------
    # Anchor pass
    # ----------------------------------------------------------------------
    print("\n[anchor] raw squash-384 cls anchor", flush=True)
    anchor = evaluate_anchor(args, val_ds, device)
    print(
        f"[anchor] top1={anchor['top1']*100:.4f}%  "
        f"top5={anchor['top5']*100:.4f}%  n={anchor['n_seen']}  "
        f"({anchor['elapsed_s']:.0f}s)",
        flush=True,
    )
    # Persist anchor for downstream inspection.
    write_json_atomic(out_dir / "eval_raw_cls_kodak_anchor.json", {
        "harness_version": HARNESS_VERSION,
        "pipeline": "raw",
        "pipeline_label": "Lossless reference (no codec)",
        "task": "cls",
        "val_ds": VAL_DS,
        "val_split": VAL_SPLIT,
        "preprocessing": "squash 384x384",
        "n_eval": anchor["n_seen"],
        "metrics": {
            "top1": anchor["top1"],
            "top5": anchor["top5"],
            "elapsed_s": anchor["elapsed_s"],
        },
        "config": {
            "teacher": TEACHER_NAME,
            "device": str(device),
            "bpp_denominator": BPP_DENOM,
            "resize": "bicubic 384x384",
        },
    })
    # Verify the anchor reproduces ~85.13% top-1 within 0.05 pp. Only enforced
    # on the full 50k pass — small-n smoke runs hit sample-size variance.
    target = 0.8513
    if args.n_eval >= N_EVAL and abs(anchor["top1"] - target) > 5e-4:
        msg = (
            f"[anchor] top-1 {anchor['top1']*100:.4f}% deviates from expected "
            f"85.13% by > 0.05 pp; stopping before launching 17-cell sweep."
        )
        print(msg, flush=True)
        if not args.anchor_only:
            sys.exit(2)

    if args.anchor_only:
        return

    # ----------------------------------------------------------------------
    # 17-cell sweep
    # ----------------------------------------------------------------------
    cells = build_cells()
    if args.cell_indices:
        idxs = [int(s) for s in args.cell_indices.split(",")]
    else:
        idxs = list(range(len(cells)))
    print(f"\n[sweep] cells = {idxs}", flush=True)

    for ci in idxs:
        cell = cells[ci]
        print(
            f"\n[cell {ci:2d}/{len(cells)-1}] {cell['filename']}",
            flush=True,
        )
        t_cell_start = time.time()
        result = evaluate_cell(args, cell, val_ds, device)
        env = envelope_for_cell(cell, result, device)
        out_path = out_dir / cell["filename"]
        write_json_atomic(out_path, env)
        dt = time.time() - t_cell_start
        m = env["metrics"]
        print(
            f"[cell {ci:2d}] WROTE {out_path.name}  "
            f"top1={m['top1']*100:.4f}%  bpp={m['bpp_mean']:.4f}  "
            f"psnr={m['psnr_db']:.2f}  ({dt:.0f}s)",
            flush=True,
        )

    print("\n[done] all cells finished.", flush=True)


if __name__ == "__main__":
    main()
