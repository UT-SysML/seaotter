"""iter-6 accuracy + distortion entry point.

Usage:
    python eval_accuracy.py --pipeline avif --task cls \\
        --op '{"type":"quality","value":25}' \\
        --out_json production/eval_avif_cls_q25.json

Pipelines encode on the *worker* side (DataLoader collate_fn) and
decode on the *main* side (batched on GPU); this pipelines CPU codec
throughput with the GPU teacher / piq metric forwards. Accuracy
(top-1/top-5 for cls; mIoU for seg) and distortion
(PSNR/SSIM/LPIPS_dB/DISTS_dB) come out of a single pass.

The pipeline instance is created in main, then fork()ed into workers
via the DataLoader — so the CPU codec (if any) must be loaded in
`Pipeline.__init__` *before* the loader is constructed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import datasets
import fastprogress
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.v2.functional import pil_to_tensor


from .metrics import DistortionAccumulator, bpp_for_image  # noqa: E402
from .pipelines import make_pipeline  # noqa: E402
from .preprocessing import (  # noqa: E402
    ade_label_offset, load_val, naflex_resize, task_crop_pil,
)
from .schema import (  # noqa: E402
    CROP_FOR_TASK, envelope_skeleton, parse_op, validate_op_for_pipeline,
)
from .teacher import load_teacher, teacher_logits_from_uint8  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", required=True)
    p.add_argument("--task", choices=("cls", "seg", "clip"), required=True)
    p.add_argument("--op", required=True,
                   help="JSON operating point, e.g. '{\"type\":\"quality\",\"value\":25}'")
    p.add_argument("--out_json", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_images", type=int, default=None,
                   help="Cap for smoke runs; default = full val.")
    return p.parse_args()


# Module-global pipeline reference, fork-inherited by DataLoader workers.
_PIPELINE_FOR_WORKERS = None


def _set_pipeline_for_workers(pipe):
    global _PIPELINE_FOR_WORKERS
    _PIPELINE_FOR_WORKERS = pipe


# --------------------------------------------------------------------------
# CLS
# --------------------------------------------------------------------------

def _collate_cls(batch):
    """Worker-side: PIL squash → uint8 tensor stack, AND codec encode → blob list."""
    pipe = _PIPELINE_FOR_WORKERS
    task = "cls"
    x_list, y_list, blobs = [], [], []
    for sample in batch:
        img = task_crop_pil(sample["jpg"], task)
        x_list.append(pil_to_tensor(img).unsqueeze(0))
        y_list.append(int(sample["cls"]))
        blobs.append(pipe.collate_encode(img))
    x = torch.cat(x_list, dim=0)
    y = torch.tensor(y_list, dtype=torch.long)
    return x, y, blobs


@torch.no_grad()
def evaluate_cls(args, op, device, env, pipe):
    crop = CROP_FOR_TASK["cls"]
    val = load_val("cls")
    n_total = val.num_rows if args.max_images is None else min(args.max_images, val.num_rows)
    print(f"[eval] cls n={n_total} batch={args.batch_size} crop={crop}", flush=True)

    sub = val.select(range(n_total))
    _set_pipeline_for_workers(pipe)
    loader = torch.utils.data.DataLoader(
        sub, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, collate_fn=_collate_cls,
        persistent_workers=(args.num_workers > 0),
    )

    teacher = load_teacher("cls", device)
    dist = DistortionAccumulator(device=device)

    correct1 = correct5 = n_seen = 0
    transmit_bytes: list[int] = []
    storage_bytes: list[int] = []
    t0 = time.time()
    pb = fastprogress.progress_bar(
        loader, total=(n_total + args.batch_size - 1) // args.batch_size,
    )
    for batch_idx, (x_uint8, y, blobs) in enumerate(pb):
        x_uint8 = x_uint8.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        recon, b_list, st_list = pipe.decode_blobs_batch(blobs)
        transmit_bytes.extend(b_list)
        if st_list is not None:
            storage_bytes.extend(st_list)

        logits = teacher_logits_from_uint8(teacher, recon)
        top1 = logits.argmax(dim=1)
        top5 = logits.topk(5, dim=1).indices
        correct1 += (top1 == y).sum().item()
        correct5 += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
        n_seen += y.numel()

        dist.update(recon, x_uint8)

        if (batch_idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            top1_pct = correct1 / n_seen * 100
            bpp_now = float(np.mean([bpp_for_image(b, crop, crop) for b in transmit_bytes]))
            print(f"  {n_seen}/{n_total} top1={top1_pct:.3f}% bpp={bpp_now:.4f} "
                  f"({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    bpp_per_image = [bpp_for_image(b, crop, crop) for b in transmit_bytes]
    transmit_bpp = float(np.mean(bpp_per_image))
    storage_bpp = transmit_bpp if not storage_bytes else float(
        np.mean([bpp_for_image(b, crop, crop) for b in storage_bytes])
    )

    metrics = {
        "top1": correct1 / n_seen,
        "top5": correct5 / n_seen,
        "miou": None,
        "pixel_accuracy": None,
        "bpp_std": float(np.std(bpp_per_image)),
        "elapsed_s": elapsed,
        **dist.finalize(),
    }
    env["transmit_bpp_mean"] = transmit_bpp
    env["storage_bpp_mean"] = storage_bpp
    env["n_eval"] = n_seen
    env["metrics"] = metrics
    env["config"] = pipe.config_block()


# --------------------------------------------------------------------------
# SEG (per-image: GT annotations have variable native shapes)
# --------------------------------------------------------------------------

def _collate_seg(batch):
    pipe = _PIPELINE_FOR_WORKERS
    task = "seg"
    out = []
    for sample in batch:
        img_full = sample["image"].convert("RGB")
        ann = sample["annotation"]
        gt_full = pil_to_tensor(ann).squeeze(0)
        img_sq = task_crop_pil(img_full, task)
        x_u8 = pil_to_tensor(img_sq)
        blob = pipe.collate_encode(img_sq)
        out.append((x_u8, gt_full, blob))
    return out


@torch.no_grad()
def evaluate_seg(args, op, device, env, pipe):
    from torchmetrics.classification import JaccardIndex

    crop = CROP_FOR_TASK["seg"]
    val = load_val("seg")
    n_total = val.num_rows if args.max_images is None else min(args.max_images, val.num_rows)
    print(f"[eval] seg n={n_total} crop={crop}", flush=True)

    sub = val.select(range(n_total))
    _set_pipeline_for_workers(pipe)
    loader = torch.utils.data.DataLoader(
        sub, batch_size=1, num_workers=args.num_workers,
        shuffle=False, collate_fn=_collate_seg,
        persistent_workers=(args.num_workers > 0),
    )

    teacher = load_teacher("seg", device)
    dist = DistortionAccumulator(device=device)

    NUM_CLASSES = 150
    metric = JaccardIndex(
        task="multiclass", num_classes=NUM_CLASSES,
        average="macro", ignore_index=255,
    ).to(device)

    transmit_bytes: list[int] = []
    storage_bytes: list[int] = []
    pixel_correct = 0
    pixel_total = 0
    t0 = time.time()
    pb = fastprogress.progress_bar(loader, total=n_total)
    for batch_idx, samples in enumerate(pb):
        # bs=1 so samples is a list of length 1.
        x_u8, gt_full, blob = samples[0]
        gt_full = ade_label_offset(gt_full).to(device)
        x_u8 = x_u8.unsqueeze(0).to(device)

        recon, b_list, st_list = pipe.decode_blobs_batch([blob])
        transmit_bytes.extend(b_list)
        if st_list is not None:
            storage_bytes.extend(st_list)

        logits = teacher_logits_from_uint8(teacher, recon)
        pred_full = F.interpolate(
            logits, size=gt_full.shape, mode="bicubic", align_corners=False,
        ).argmax(dim=1).squeeze(0).to(torch.uint8)
        metric.update(pred_full, gt_full)
        valid = (gt_full != 255)
        pixel_correct += int((pred_full[valid] == gt_full[valid]).sum().item())
        pixel_total += int(valid.sum().item())

        dist.update(recon, x_u8)

        if (batch_idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            cur = float(metric.compute().item())
            bpp_now = float(np.mean([bpp_for_image(b, crop, crop) for b in transmit_bytes]))
            print(f"  {batch_idx+1}/{n_total} mIoU={cur:.4f} bpp={bpp_now:.4f} "
                  f"({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    bpp_per_image = [bpp_for_image(b, crop, crop) for b in transmit_bytes]
    transmit_bpp = float(np.mean(bpp_per_image))
    storage_bpp = transmit_bpp if not storage_bytes else float(
        np.mean([bpp_for_image(b, crop, crop) for b in storage_bytes])
    )
    metrics = {
        "top1": None,
        "top5": None,
        "miou": float(metric.compute().item()),
        "pixel_accuracy": (pixel_correct / pixel_total) if pixel_total else None,
        "bpp_std": float(np.std(bpp_per_image)),
        "elapsed_s": elapsed,
        **dist.finalize(),
    }
    env["transmit_bpp_mean"] = transmit_bpp
    env["storage_bpp_mean"] = storage_bpp
    env["n_eval"] = n_total
    env["metrics"] = metrics
    env["config"] = pipe.config_block()


# --------------------------------------------------------------------------
# CLIP (zero-shot ImageNet against SigLIP-2; bs=1 inline path, variable shape)
# --------------------------------------------------------------------------

def _collate_clip(batch):
    """Worker-side: PIL → naflex_resize → uint8 tensor (variable shape) + codec encode."""
    pipe = _PIPELINE_FOR_WORKERS
    out = []
    for sample in batch:
        pil_img = sample["jpg"].convert("RGB")
        cls_label = int(sample["cls"])
        pil_nat = naflex_resize(pil_img)
        x_u8 = pil_to_tensor(pil_nat)  # (3, H', W')
        blob = pipe.collate_encode(pil_nat)
        out.append((x_u8, cls_label, blob, pil_nat.size))  # size = (W', H')
    return out


@torch.no_grad()
def evaluate_clip(args, op, device, env, pipe):
    val = load_val("clip")
    n_total = val.num_rows if args.max_images is None else min(args.max_images, val.num_rows)
    print(f"[eval] clip n={n_total} bs=1 (naflex variable shape)", flush=True)

    sub = val.select(range(n_total))
    _set_pipeline_for_workers(pipe)
    loader = torch.utils.data.DataLoader(
        sub, batch_size=1, num_workers=args.num_workers,
        shuffle=False, collate_fn=_collate_clip,
        persistent_workers=(args.num_workers > 0),
    )

    teacher = load_teacher("clip", device)
    # Warm up the prototype matrix once (cached after first call).
    teacher.compute_clip_prototypes()
    dist = DistortionAccumulator(device=device)

    correct1 = correct5 = n_seen = 0
    transmit_bytes: list[int] = []
    storage_bytes: list[int] = []
    transmit_hw: list[tuple[int, int]] = []
    storage_hw: list[tuple[int, int]] = []
    t0 = time.time()
    pb = fastprogress.progress_bar(loader, total=n_total)
    for batch_idx, samples in enumerate(pb):
        x_u8, label, blob, (W_nat, H_nat) = samples[0]
        x_u8 = x_u8.unsqueeze(0).to(device, non_blocking=True)
        y = torch.tensor([label], dtype=torch.long, device=device)

        recon, b_list, st_list = pipe.decode_blobs_batch([blob])
        transmit_bytes.extend(b_list)
        transmit_hw.append((H_nat, W_nat))
        if st_list is not None:
            storage_bytes.extend(st_list)
            storage_hw.append((H_nat, W_nat))

        logits = teacher.zero_shot_logits(recon)
        top1 = logits.argmax(dim=1)
        top5 = logits.topk(5, dim=1).indices
        correct1 += int((top1 == y).item())
        correct5 += int((top5 == y.unsqueeze(1)).any(dim=1).item())
        n_seen += 1

        dist.update(recon, x_u8)

        if (batch_idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            top1_pct = correct1 / n_seen * 100
            bpp_now = float(np.mean([
                bpp_for_image(b, h, w)
                for b, (h, w) in zip(transmit_bytes, transmit_hw)
            ]))
            print(
                f"  {n_seen}/{n_total} top1={top1_pct:.3f}% bpp={bpp_now:.4f} "
                f"({elapsed:.0f}s)", flush=True,
            )

    elapsed = time.time() - t0
    bpp_per_image = [
        bpp_for_image(b, h, w)
        for b, (h, w) in zip(transmit_bytes, transmit_hw)
    ]
    transmit_bpp = float(np.mean(bpp_per_image))
    if storage_bytes:
        storage_bpp = float(np.mean([
            bpp_for_image(b, h, w)
            for b, (h, w) in zip(storage_bytes, storage_hw)
        ]))
    else:
        storage_bpp = transmit_bpp
    metrics = {
        "top1": correct1 / n_seen,
        "top5": correct5 / n_seen,
        "miou": None,
        "pixel_accuracy": None,
        "bpp_std": float(np.std(bpp_per_image)),
        "elapsed_s": elapsed,
        **dist.finalize(),
    }
    env["transmit_bpp_mean"] = transmit_bpp
    env["storage_bpp_mean"] = storage_bpp
    env["n_eval"] = n_seen
    env["metrics"] = metrics
    env["config"] = pipe.config_block()


def main() -> None:
    args = parse_args()
    op = parse_op(args.op)
    validate_op_for_pipeline(args.pipeline, op)
    device = torch.device(args.device)
    env = envelope_skeleton(
        pipeline=args.pipeline, task=args.task, op=op, kind="accuracy",
    )

    # Build pipeline in MAIN (so CPU codecs are loaded once and fork-inherited).
    pipe = make_pipeline(args.pipeline, op, args.task, device)

    if args.task == "cls":
        evaluate_cls(args, op, device, env, pipe)
    elif args.task == "seg":
        evaluate_seg(args, op, device, env, pipe)
    else:
        evaluate_clip(args, op, device, env, pipe)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(env, indent=2))
    m = env["metrics"]
    print(f"[eval] WROTE {out_path}", flush=True)
    psnr_db = m.get("psnr_db")
    ssim = m.get("ssim")
    lpips_db = m.get("lpips_db")
    dists_db = m.get("dists_db")
    print(
        f"[eval] bpp={env['transmit_bpp_mean']:.4f} "
        f"top1={m.get('top1')} miou={m.get('miou')} "
        f"psnr={psnr_db:.2f} ssim={ssim:.4f} "
        f"lpips_db={lpips_db:.2f} dists_db={dists_db:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
