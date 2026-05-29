"""Standalone-codec Kodak eval (CPU only, bs=1, native resolution).

Compares three codecs on Kodak 24-image validation at native size:

  1. Vanilla JPEG (Pillow default subsampling = 4:2:0)
  2. Vanilla JPEG sub=0 (4:4:4)
  3. SEA OTTER S3 K=3 production bundle (load_from_hub default)

Selects q-anchors (q1, q2, q3) per the smallest-q strict-dominance rule
against JPEG-sub=0, then interpolates q0.5 / q1.5 / q2.5 / q3.5.
Emits 17 per-cell JSONs + summary.json with per-image arrays + mean+median
summary stats. Distortion: piq.psnr / piq.ssim / piq.LPIPS / piq.DISTS
on [0,1] tensors; LPIPS/DISTS reported in dB via -10*log10.

Per CLAUDE.md hard rules: CPU only, no resize/crop, bs=1.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import statistics
from pathlib import Path

import datasets
import piq
import torch
from PIL import Image
from torchvision.transforms.v2.functional import pil_to_tensor

from seaotter import load_from_hub


HARNESS_VERSION = "codec-kodak-v1"
DEVICE = torch.device("cpu")


def db_from_metric(value: float) -> float:
    return -10.0 * math.log10(max(value, 1e-12))


def to_unit_batch(x_uint8: torch.Tensor) -> torch.Tensor:
    return x_uint8.to(torch.float32).unsqueeze(0) / 255.0


def per_image_metrics(
    recon_uint8: torch.Tensor,
    ref_uint8: torch.Tensor,
    lpips_net: piq.LPIPS,
    dists_net: piq.DISTS,
) -> dict[str, float]:
    recon_f = to_unit_batch(recon_uint8)
    ref_f = to_unit_batch(ref_uint8)
    with torch.no_grad():
        mse = (recon_f - ref_f).pow(2).mean().item()
        psnr_db = -10.0 * math.log10(max(mse, 1e-12))
        ssim = float(piq.ssim(recon_f, ref_f, data_range=1.0, reduction="mean").item())
        lpips_val = float(lpips_net(recon_f, ref_f).item())
        dists_val = float(dists_net(recon_f, ref_f).item())
    return {
        "psnr_db": psnr_db,
        "ssim": ssim,
        "lpips_db": db_from_metric(lpips_val),
        "dists_db": db_from_metric(dists_val),
    }


def summarize(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.fmean(values), "median": statistics.median(values)}


def encode_pillow_jpeg(pil: Image.Image, quality: int, subsampling: int) -> bytes:
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality, subsampling=subsampling)
    return buf.getvalue()


def decode_pillow_jpeg(jpeg_bytes: bytes) -> torch.Tensor:
    pil = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    return pil_to_tensor(pil)  # (3, H, W) uint8


def init_cell_arrays(n: int = 24) -> dict[str, list]:
    return {
        "kodak_index": list(range(n)),
        "image_h": [None] * n,
        "image_w": [None] * n,
        "bpp": [None] * n,
        "psnr_db": [None] * n,
        "ssim": [None] * n,
        "lpips_db": [None] * n,
        "dists_db": [None] * n,
    }


def per_image_arrays_to_json(per_image: dict[str, list]) -> dict:
    return {
        "bpp": summarize(per_image["bpp"]),
        "psnr_db": summarize(per_image["psnr_db"]),
        "ssim": summarize(per_image["ssim"]),
        "lpips_db": summarize(per_image["lpips_db"]),
        "dists_db": summarize(per_image["dists_db"]),
    }


def write_cell(
    out_dir: Path,
    filename: str,
    *,
    pipeline: str,
    pipeline_label: str,
    operating_point: dict,
    per_image: dict,
    config: dict,
) -> None:
    payload = {
        "harness_version": HARNESS_VERSION,
        "pipeline": pipeline,
        "pipeline_label": pipeline_label,
        "task": "kodak_recon",
        "val_ds": "danjacobellis/kodak",
        "val_split": "validation",
        "preprocessing": "native (no resize, no crop)",
        "operating_point": operating_point,
        "n_eval": 24,
        "per_image": per_image,
        "summary": per_image_arrays_to_json(per_image),
        "config": config,
    }
    (out_dir / filename).write_text(json.dumps(payload, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out_dir",
        type=str,
        default="/home/dgj335/UT-SysML/seaotter/results/codec_kodak",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load Kodak (24 images, native resolution).
    # ------------------------------------------------------------------
    print("[1/6] Loading Kodak validation set...")
    ds = datasets.load_dataset("danjacobellis/kodak", split="validation")
    assert len(ds) == 24, f"expected 24, got {len(ds)}"
    pils: list[Image.Image] = [ds[i]["image"].convert("RGB") for i in range(24)]
    refs_uint8: list[torch.Tensor] = [pil_to_tensor(im) for im in pils]
    hws = [(int(t.shape[1]), int(t.shape[2])) for t in refs_uint8]  # (H, W)

    # ------------------------------------------------------------------
    # 2. Load SEA OTTER bundle (S3 default).
    # ------------------------------------------------------------------
    print("[2/6] Loading SEA OTTER S3 bundle from HF hub...")
    bundle = load_from_hub()
    assert bundle.lambdas == [0.75, 0.4, 0.22], bundle.lambdas
    assert bundle.K == 3

    # ------------------------------------------------------------------
    # 3. Initialize distortion nets on CPU.
    # ------------------------------------------------------------------
    print("[3/6] Initializing piq.LPIPS / piq.DISTS on CPU...")
    lpips_net = piq.LPIPS().to(DEVICE).eval()
    dists_net = piq.DISTS().to(DEVICE).eval()

    # ------------------------------------------------------------------
    # 4. Run SEA OTTER eval at k ∈ {0, 1, 2}.
    # ------------------------------------------------------------------
    print("[4/6] SEA OTTER eval (3 ops × 24 images)...")
    sea_anchors: list[dict] = []
    for k in range(3):
        arrs = init_cell_arrays()
        for i in range(24):
            x = refs_uint8[i]
            H, W = hws[i]
            jpeg_bytes = bundle.encode(x, k=k)
            recon = bundle.decode(jpeg_bytes)
            m = per_image_metrics(recon, x, lpips_net, dists_net)
            arrs["image_h"][i] = H
            arrs["image_w"][i] = W
            arrs["bpp"][i] = 8.0 * len(jpeg_bytes) / (H * W)
            arrs["psnr_db"][i] = m["psnr_db"]
            arrs["ssim"][i] = m["ssim"]
            arrs["lpips_db"][i] = m["lpips_db"]
            arrs["dists_db"][i] = m["dists_db"]
        write_cell(
            out_dir,
            f"eval_seaotter_kodak_k{k}.json",
            pipeline="seaotter",
            pipeline_label=f"SEA OTTER S3 K=3, k={k} (λ={bundle.lambdas[k]})",
            operating_point={"type": "seaotter_k", "value": k},
            per_image=arrs,
            config={
                "codec": "seaotter",
                "subsampling": 0,
                "quality": None,
                "seaotter_k": k,
                "seaotter_source": "danjacobellis/seaotter @ seaotter_jpeg_s3 (load_from_hub default)",
                "seaotter_lambdas": bundle.lambdas,
                "device": "cpu",
            },
        )
        anchor = {
            "k": k,
            "lam": bundle.lambdas[k],
            "mean_bpp": summarize(arrs["bpp"])["mean"],
            "mean_psnr": summarize(arrs["psnr_db"])["mean"],
            "per_image_bpp": list(arrs["bpp"]),
            "per_image_psnr": list(arrs["psnr_db"]),
        }
        sea_anchors.append(anchor)
        print(
            f"  k={k}: mean_bpp={anchor['mean_bpp']:.4f}  mean_psnr={anchor['mean_psnr']:.3f}"
        )

    # ------------------------------------------------------------------
    # 5. JPEG sub=0 sweep q ∈ [1, 99] for q-anchor selection.
    # ------------------------------------------------------------------
    # Compute bpp+psnr only (cheap); skip LPIPS/DISTS for the sweep.
    print("[5/6] JPEG sub=0 sweep q∈[1,99] for anchor selection...")
    sweep_bpp: dict[int, float] = {}
    sweep_psnr: dict[int, float] = {}
    for q in range(1, 100):
        per_bpp = []
        per_psnr = []
        for i in range(24):
            H, W = hws[i]
            jpeg_bytes = encode_pillow_jpeg(pils[i], quality=q, subsampling=0)
            recon = decode_pillow_jpeg(jpeg_bytes)
            recon_f = to_unit_batch(recon)
            ref_f = to_unit_batch(refs_uint8[i])
            mse = (recon_f - ref_f).pow(2).mean().item()
            per_bpp.append(8.0 * len(jpeg_bytes) / (H * W))
            per_psnr.append(-10.0 * math.log10(max(mse, 1e-12)))
        sweep_bpp[q] = statistics.fmean(per_bpp)
        sweep_psnr[q] = statistics.fmean(per_psnr)

    # smallest-q strict-dominance rule per anchor k
    warnings: list[str] = []
    q_anchors: list[int] = []
    for k_idx, anchor in enumerate(sea_anchors):
        chosen_q = None
        for q in range(1, 100):
            # SEA OTTER@k strictly dominates JPEG-sub=0@q iff
            #   mean_psnr_jpeg < mean_psnr_sea  AND  mean_bpp_jpeg > mean_bpp_sea
            if sweep_psnr[q] < anchor["mean_psnr"] and sweep_bpp[q] > anchor["mean_bpp"]:
                chosen_q = q
                break
        if chosen_q is None:
            warnings.append(
                f"dominance check failed for k={k_idx}: no q in [1,99] is strictly dominated; using q=1 placeholder"
            )
            chosen_q = 1
            anchor["dominance_ok"] = False
        else:
            anchor["dominance_ok"] = True
        anchor["matched_q_sub0"] = chosen_q
        q_anchors.append(chosen_q)
        print(
            f"  k={k_idx} anchor q={chosen_q}  (jpeg_sub0 bpp={sweep_bpp[chosen_q]:.4f} psnr={sweep_psnr[chosen_q]:.3f})  dom_ok={anchor['dominance_ok']}"
        )

    q1, q2, q3 = q_anchors

    def clamp_q(v: int) -> int:
        return max(1, min(99, int(v)))

    q1p5 = clamp_q(round((q1 + q2) / 2))
    q2p5 = clamp_q(round((q2 + q3) / 2))
    q0p5 = clamp_q(round(q1 - (q2 - q1) / 2))
    q3p5 = clamp_q(round(q3 + (q3 - q2) / 2))

    q_ladder = {
        "q0p5": q0p5, "q1": q1, "q1p5": q1p5, "q2": q2,
        "q2p5": q2p5, "q3": q3, "q3p5": q3p5,
    }
    print("  q-ladder:", q_ladder)

    # collision check
    collisions: list[str] = []
    pairs = [
        ("q0p5", q0p5, "q1", q1),
        ("q1p5", q1p5, "q1", q1), ("q1p5", q1p5, "q2", q2),
        ("q2p5", q2p5, "q2", q2), ("q2p5", q2p5, "q3", q3),
        ("q3p5", q3p5, "q3", q3),
    ]
    for an, av, bn, bv in pairs:
        if av == bv:
            collisions.append(f"{an} == {bn} (={av})")

    # ------------------------------------------------------------------
    # 6. Run JPEG default + JPEG sub=0 at all 7 q values (full metrics).
    # ------------------------------------------------------------------
    print("[6/6] Running JPEG default + JPEG sub=0 at 7 q values...")
    q_names = ["q0p5", "q1", "q1p5", "q2", "q2p5", "q3", "q3p5"]
    q_values = [q_ladder[n] for n in q_names]

    for subsampling, prefix, pipeline_id, label_suffix in [
        (2, "eval_jpeg_kodak", "jpeg", "Pillow default subsampling (4:2:0)"),
        (0, "eval_jpeg_sub0_kodak", "jpeg_sub0", "subsampling=0 (4:4:4)"),
    ]:
        for q_name, q in zip(q_names, q_values):
            arrs = init_cell_arrays()
            for i in range(24):
                H, W = hws[i]
                jpeg_bytes = encode_pillow_jpeg(pils[i], quality=q, subsampling=subsampling)
                recon = decode_pillow_jpeg(jpeg_bytes)
                m = per_image_metrics(recon, refs_uint8[i], lpips_net, dists_net)
                arrs["image_h"][i] = H
                arrs["image_w"][i] = W
                arrs["bpp"][i] = 8.0 * len(jpeg_bytes) / (H * W)
                arrs["psnr_db"][i] = m["psnr_db"]
                arrs["ssim"][i] = m["ssim"]
                arrs["lpips_db"][i] = m["lpips_db"]
                arrs["dists_db"][i] = m["dists_db"]
            write_cell(
                out_dir,
                f"{prefix}_{q_name}.json",
                pipeline=pipeline_id,
                pipeline_label=f"Pillow JPEG quality={q}, {label_suffix}",
                operating_point={"type": "jpeg_q", "value": q, "ladder_id": q_name},
                per_image=arrs,
                config={
                    "codec": pipeline_id,
                    "subsampling": subsampling,
                    "quality": q,
                    "seaotter_k": None,
                    "seaotter_source": None,
                    "seaotter_lambdas": None,
                    "device": "cpu",
                },
            )
            print(
                f"  {pipeline_id} {q_name} (q={q}): "
                f"bpp_mean={summarize(arrs['bpp'])['mean']:.4f}  "
                f"psnr_mean={summarize(arrs['psnr_db'])['mean']:.3f}"
            )

    # ------------------------------------------------------------------
    # summary.json
    # ------------------------------------------------------------------
    summary = {
        "harness_version": HARNESS_VERSION,
        "seaotter_bundle": "S3 (load_from_hub default; danjacobellis/seaotter @ seaotter_jpeg_s3)",
        "seaotter_anchors": [
            {
                "k": a["k"],
                "lambda": a["lam"],
                "mean_bpp": a["mean_bpp"],
                "mean_psnr": a["mean_psnr"],
                "matched_q_sub0": a["matched_q_sub0"],
                "dominance_ok": a["dominance_ok"],
            }
            for a in sea_anchors
        ],
        "q_ladder": q_ladder,
        "jpeg_sub0_sweep_summary": {
            "q": list(range(1, 100)),
            "mean_bpp": [sweep_bpp[q] for q in range(1, 100)],
            "mean_psnr_db": [sweep_psnr[q] for q in range(1, 100)],
        },
        "q_selection_rule": (
            "smallest integer q in [1,99] such that mean_PSNR(JPEG sub=0 @ q) "
            "< mean_PSNR(SEA OTTER @ k) AND mean_bpp(JPEG sub=0 @ q) > mean_bpp(SEA OTTER @ k)"
        ),
        "q_interpolation_rule": (
            "q1p5 = round((q1+q2)/2); q2p5 = round((q2+q3)/2); "
            "q0p5 = round(q1 - (q2-q1)/2); q3p5 = round(q3 + (q3-q2)/2); "
            "all clamped to [1, 99]"
        ),
        "collisions": collisions,
        "warnings": warnings,
        "n_eval": 24,
        "val_ds": "danjacobellis/kodak",
        "device": "cpu",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
