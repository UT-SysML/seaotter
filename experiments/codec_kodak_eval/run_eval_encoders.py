"""Standalone-codec Kodak encode-only eval (bpp at native resolution).

Variant of `run_eval.py` for additional codecs:

  - `avif` (libavif default speed)
  - `avifx` (libavif speed=10)
  - `frp` (FRAPPE-only, op = n_ch)
  - `wal` (WaLLoC RGB_16x, op = pixel_ratio %)

Each cell encodes the 24 Kodak validation images at native resolution
(no resize, no crop; 16 768x512 + 8 512x768) and reports per-image bpp
plus summary mean+median. The schema follows `codec-kodak-v1` so
downstream readers can stack these alongside the existing
`eval_seaotter_kodak_*.json` / `eval_jpeg_kodak_*.json` files.

Bit accounting matches the iter-6 production transmit_bpp definition:
  - `frp`: length of `encode_latents(arrange_latents(latents))` (JPEG-LS
    blob) over the native H*W pixel count.
  - `wal`: length of the WaLLoC WebP-lossless latent blob *only* (the
    4-byte `<HH>` header `collate_encode` prepends is NOT counted), over
    the native H*W pixel count.
  - `avif` / `avifx`: length of the AVIF byte blob from `PIL.save(...)`
    over the native H*W pixel count.

Distortion (PSNR/SSIM/LPIPS/DISTS) is computed *only* for completeness;
the matched_rate branch consumes `summary.bpp.mean` only. PSNR/SSIM/
LPIPS/DISTS use the same piq harness as `run_eval.py`. WaLLoC has a
resize-down-resize-up roundtrip baked in; we measure distortion at
native resolution after upsampling back to (H, W) so the number is
directly comparable to the other codecs.

CLI:
  python run_eval_encoders.py --codec frp --n_ch 12
  python run_eval_encoders.py --codec wal --pixel_ratio 16
  python run_eval_encoders.py --codec avif --quality 1
  python run_eval_encoders.py --codec avifx --quality 1
"""

from __future__ import annotations

import argparse
import io
import json
import math
import statistics
import struct
import sys
from pathlib import Path

import datasets
import piq
import torch
from PIL import Image
from torchvision.transforms.v2.functional import pil_to_tensor

HARNESS_VERSION = "codec-kodak-v1"
DEVICE = torch.device("cpu")


def db_from_metric(value: float) -> float:
    return -10.0 * math.log10(max(value, 1e-12))


def to_unit_batch(x_uint8: torch.Tensor) -> torch.Tensor:
    return x_uint8.to(torch.float32).unsqueeze(0) / 255.0


def per_image_metrics(
    recon_uint8: torch.Tensor,
    ref_uint8: torch.Tensor,
    lpips_net: "piq.LPIPS",
    dists_net: "piq.DISTS",
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


def per_image_arrays_to_summary(per_image: dict[str, list]) -> dict:
    return {
        "bpp": summarize(per_image["bpp"]),
        "psnr_db": summarize(per_image["psnr_db"]),
        "ssim": summarize(per_image["ssim"]),
        "lpips_db": summarize(per_image["lpips_db"]),
        "dists_db": summarize(per_image["dists_db"]),
    }


def write_cell(
    out_path: Path,
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
        "summary": per_image_arrays_to_summary(per_image),
        "config": config,
    }
    out_path.write_text(json.dumps(payload, indent=2))


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


# -------------------- codec-specific encode/decode --------------------


def encode_decode_avif(
    pil: Image.Image, quality: int, speed: int | None,
) -> tuple[bytes, torch.Tensor]:
    buf = io.BytesIO()
    kw = {"format": "AVIF", "quality": quality}
    if speed is not None:
        kw["speed"] = speed
    pil.save(buf, **kw)
    blob = buf.getvalue()
    rec = Image.open(io.BytesIO(blob))
    rec.load()
    return blob, pil_to_tensor(rec.convert("RGB"))


def setup_frp(n_ch: int, device: torch.device):
    from compressors.frappe.model import load_from_hub, load_progressive_model
    cfg, weights, n_trained = load_from_hub()
    if not 1 <= n_ch <= n_trained:
        raise ValueError(f"n_ch must be in [1, {n_trained}], got {n_ch}")
    model = load_progressive_model(weights, cfg, n_ch, device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


def encode_decode_frp(
    pil: Image.Image, model, cfg,
) -> tuple[bytes, torch.Tensor]:
    from compressors.frappe.entropy_coding import (
        arrange_latents, encode_latents, decode_latents, unarrange_latents,
    )
    from compressors.frappe.quantize import srgb_to_linear

    linear_input = bool(getattr(cfg, "linear_input", False))
    x_u8 = pil_to_tensor(pil.convert("RGB")).unsqueeze(0)
    x = x_u8.to(torch.float32) / 127.5 - 1.0
    x_in = srgb_to_linear(x) if linear_input else x

    with torch.no_grad():
        latents = model.encode(x_in)
        latents_q = [z.round().clamp(-127, 127).to(torch.int8) for z in latents]
        blob = encode_latents(arrange_latents(latents_q))
        arranged = decode_latents(blob)
        latents_recon = unarrange_latents(arranged, model.scale_groups)
        latents_f = [z.to(torch.float32) for z in latents_recon]
        rgb = model.decode(latents_f).clamp(-1.0, 1.0)
        rgb_u8 = ((rgb + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8).squeeze(0)
    return blob, rgb_u8


def setup_wal(device: torch.device):
    from compressors.walloc._codec import load_codec
    codec, info = load_codec(device=device, torch_dtype=torch.float32)
    return codec, info


def encode_decode_wal(
    pil: Image.Image, codec, info, pixel_ratio: float, device: torch.device,
) -> tuple[bytes, torch.Tensor]:
    from compressors.walloc._codec import (
        SNAP_MULTIPLE, snap_shape, encode_to_latent, latent_to_webp_bytes,
        decode_from_blob, from_model_output, resize_bicubic, to_model_input,
    )
    x_u8 = pil_to_tensor(pil.convert("RGB")).unsqueeze(0)
    x = x_u8.to(torch.float32) / 255.0
    _, _, H, W = x.shape
    if H % SNAP_MULTIPLE or W % SNAP_MULTIPLE:
        raise ValueError(f"wal: H, W must be multiples of {SNAP_MULTIPLE}, got {H}x{W}")
    snap_h, snap_w = snap_shape(H, W, pixel_ratio)
    x_r = resize_bicubic(x, (snap_h, snap_w)).clamp(0, 1)
    x_in = to_model_input(x_r)

    with torch.no_grad():
        z_hat = encode_to_latent(codec, x_in)
        latent_bytes = latent_to_webp_bytes(z_hat, info.latent_bits)
        x_hat = decode_from_blob(
            codec, latent_bytes, info.latent_dim, info.latent_bits,
            device, torch.float32,
        )
        recon = from_model_output(x_hat).clamp(0, 1)
        recon = resize_bicubic(recon, (H, W)).clamp(0, 1)
        recon_u8 = (recon * 255.0).round().clamp(0, 255).to(torch.uint8).squeeze(0)
    # Per iter-6 transmit accounting: count only the latent bytes, NOT the
    # 4-byte <HH> header. So we report `len(latent_bytes)`.
    return latent_bytes, recon_u8


# -------------------- per-cell runner --------------------


def run_cell(
    *,
    codec: str,
    out_path: Path,
    pils: list[Image.Image],
    refs_uint8: list[torch.Tensor],
    hws: list[tuple[int, int]],
    lpips_net,
    dists_net,
    quality: int | None = None,
    avif_speed: int | None = None,
    n_ch: int | None = None,
    pixel_ratio: float | None = None,
):
    arrs = init_cell_arrays()

    if codec == "frp":
        assert n_ch is not None
        model, cfg = setup_frp(n_ch, DEVICE)
    elif codec == "wal":
        assert pixel_ratio is not None
        codec_obj, info = setup_wal(DEVICE)

    for i in range(24):
        x = refs_uint8[i]
        H, W = hws[i]
        if codec == "avif" or codec == "avifx":
            blob, recon = encode_decode_avif(pils[i], quality, avif_speed)
        elif codec == "frp":
            blob, recon = encode_decode_frp(pils[i], model, cfg)
        elif codec == "wal":
            blob, recon = encode_decode_wal(pils[i], codec_obj, info, pixel_ratio, DEVICE)
        else:
            raise ValueError(codec)
        m = per_image_metrics(recon, x, lpips_net, dists_net)
        arrs["image_h"][i] = H
        arrs["image_w"][i] = W
        arrs["bpp"][i] = 8.0 * len(blob) / (H * W)
        arrs["psnr_db"][i] = m["psnr_db"]
        arrs["ssim"][i] = m["ssim"]
        arrs["lpips_db"][i] = m["lpips_db"]
        arrs["dists_db"][i] = m["dists_db"]

    # build operating_point + config + label
    if codec == "avif":
        pipeline_label = f"libavif default speed, quality={quality}"
        operating_point = {"type": "quality", "value": quality}
        config = {
            "codec": "avif",
            "quality": quality,
            "avif_speed": None,
            "device": "cpu",
        }
        pipeline = "avif"
    elif codec == "avifx":
        pipeline_label = f"libavif speed={avif_speed}, quality={quality}"
        operating_point = {
            "type": "quality", "value": quality,
            "extras": {"avif_speed": avif_speed},
        }
        config = {
            "codec": "avifx",
            "quality": quality,
            "avif_speed": avif_speed,
            "device": "cpu",
        }
        pipeline = "avifx"
    elif codec == "frp":
        pipeline_label = f"FRAPPE-only n_ch={n_ch}"
        operating_point = {"type": "n_ch", "value": n_ch}
        config = {
            "codec": "frappe",
            "n_ch": n_ch,
            "device": "cpu",
        }
        pipeline = "frp"
    elif codec == "wal":
        pipeline_label = f"WaLLoC RGB_16x pixel_ratio={pixel_ratio}"
        operating_point = {"type": "q_pixel_ratio", "value": float(pixel_ratio)}
        config = {
            "codec": "walloc:RGB_16x",
            "q_pixel_ratio": float(pixel_ratio),
            "device": "cpu",
        }
        pipeline = "wal"

    write_cell(
        out_path,
        pipeline=pipeline,
        pipeline_label=pipeline_label,
        operating_point=operating_point,
        per_image=arrs,
        config=config,
    )
    s = per_image_arrays_to_summary(arrs)
    print(
        f"WROTE {out_path.name}  bpp_mean={s['bpp']['mean']:.5f}  "
        f"CR={24.0/s['bpp']['mean']:.2f}  psnr_db_mean={s['psnr_db']['mean']:.3f}"
    )
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out_dir",
        type=str,
        default="/home/dgj335/UT-SysML/seaotter/results/codec_kodak",
    )
    p.add_argument("--codec", required=True, choices=["avif", "avifx", "frp", "wal"])
    p.add_argument("--quality", type=int, default=None)
    p.add_argument("--avif_speed", type=int, default=None)
    p.add_argument("--n_ch", type=int, default=None)
    p.add_argument("--pixel_ratio", type=float, default=None)
    p.add_argument("--out_name", type=str, default=None,
                   help="override the output filename (without .json)")
    args = p.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # build default filename
    if args.out_name is None:
        if args.codec == "avif":
            name = f"eval_avif_kodak_q{args.quality}"
        elif args.codec == "avifx":
            name = f"eval_avifx_kodak_q{args.quality}_s{args.avif_speed}"
        elif args.codec == "frp":
            name = f"eval_frp_kodak_n{args.n_ch}"
        elif args.codec == "wal":
            # snap to int if integral
            pr = args.pixel_ratio
            label = f"{pr:.4f}".rstrip("0").rstrip(".").replace(".", "p")
            name = f"eval_wal_kodak_p{label}"
    else:
        name = args.out_name
    out_path = out_dir / (name + ".json")

    # ----- load Kodak -----
    ds = datasets.load_dataset("danjacobellis/kodak", split="validation")
    assert len(ds) == 24, f"expected 24, got {len(ds)}"
    pils = [ds[i]["image"].convert("RGB") for i in range(24)]
    refs_uint8 = [pil_to_tensor(im) for im in pils]
    hws = [(int(t.shape[1]), int(t.shape[2])) for t in refs_uint8]

    lpips_net = piq.LPIPS().to(DEVICE).eval()
    dists_net = piq.DISTS().to(DEVICE).eval()

    s = run_cell(
        codec=args.codec, out_path=out_path,
        pils=pils, refs_uint8=refs_uint8, hws=hws,
        lpips_net=lpips_net, dists_net=dists_net,
        quality=args.quality, avif_speed=args.avif_speed,
        n_ch=args.n_ch, pixel_ratio=args.pixel_ratio,
    )
    return s


if __name__ == "__main__":
    main()
