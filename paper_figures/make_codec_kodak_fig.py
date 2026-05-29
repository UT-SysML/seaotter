"""Render the WIP two-panel codec_kodak.pdf for the branched draft.

Left panel: PSNR (dB) vs Kodak bpp for three codecs (SEAOTTER, JPEG
4:2:0, JPEG 4:4:4 sub=0). Data lives in
`~/UT-SysML/seaotter/results/codec_kodak/`.

Right panel: ImageNet top-1 vs bpp on the SAME 17 cells, using the
frozen `convnext_tiny.in12k_ft_in1k_384` teacher under squash 384x384
preprocessing. Data may not yet exist; if absent we draw the three
curves with NaN y-values and a centered "[Top-1 results in progress]"
annotation.

Styling mirrors `~/danjacobellis/seaotter/experiments/codec_kodak_eval/plot_results.ipynb`:
serif font, blueviolet for SEAOTTER / black for JPEG 4:2:0 / gray for
JPEG 4:4:4, log-x with plain-number ticks at (1, 2, 3, 4, 5).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        14,
    "axes.titlesize":   16,
    "axes.labelsize":   14,
    "legend.fontsize":  11,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
})

ROOT = Path("/home/dgj335/UT-SysML/seaotter/results")
KODAK_DIR = ROOT / "codec_kodak"

# Optional ImageNet-side mirror; may not exist at draft time.
CLS_KODAK_DIR = ROOT / "codec_kodak_cls"

COLORS = {
    "seaotter":  "blueviolet",
    "jpeg":      "black",       # default 4:2:0
    "jpeg_sub0": "dimgray",     # 4:4:4
}
LABELS = {
    "seaotter":  "Learned (ours)",
    "jpeg":      "ITU T.81 4:2:0",
    "jpeg_sub0": "ITU T.81 4:4:4",
}


def _load_psnr_cells():
    """Return {codec_key: [(bpp_mean, psnr_mean), ...]} for the 17 cells."""
    out = {k: [] for k in COLORS}
    for path in sorted(KODAK_DIR.glob("eval_*.json")):
        d = json.loads(path.read_text())
        codec = d["config"]["codec"]
        if codec not in out:
            continue
        out[codec].append((d["summary"]["bpp"]["mean"],
                           d["summary"]["psnr_db"]["mean"]))
    for k in out:
        out[k].sort()
    return out


def _load_top1_cells():
    """Return {codec_key: [(bpp, top1), ...]} for the cls-side 17-cell rerun.

    The eval may not be finished. We pair each KODAK_DIR cell with the
    cls-side cell by config (`config.codec`, `config.quality`,
    `config.seaotter_k`). Returns empty lists if `codec_kodak_cls/` is
    absent.
    """
    out = {k: [] for k in COLORS}
    if not CLS_KODAK_DIR.exists():
        return out
    by_key: dict[tuple, dict] = {}
    for path in sorted(CLS_KODAK_DIR.glob("eval_*.json")):
        d = json.loads(path.read_text())
        cfg = d.get("config", {})
        codec = cfg.get("codec")
        if codec not in COLORS:
            continue
        key = (codec, cfg.get("quality"), cfg.get("seaotter_k"))
        by_key[key] = d
    # walk the kodak-PSNR side to keep the ladder aligned
    for path in sorted(KODAK_DIR.glob("eval_*.json")):
        d = json.loads(path.read_text())
        cfg = d.get("config", {})
        codec = cfg.get("codec")
        if codec not in COLORS:
            continue
        key = (codec, cfg.get("quality"), cfg.get("seaotter_k"))
        match = by_key.get(key)
        if match is None:
            continue
        bpp = match.get("transmit_bpp_mean") or d["summary"]["bpp"]["mean"]
        top1 = (match.get("metrics") or {}).get("top1")
        if top1 is None:
            continue
        out[codec].append((float(bpp), float(top1)))
    for k in out:
        out[k].sort()
    return out


def _set_log_x_with_plain_ticks(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%g"))
    ax.xaxis.set_major_locator(mtick.FixedLocator([0.5, 1, 2, 3, 4, 5]))
    ax.xaxis.set_minor_formatter(mtick.NullFormatter())


def main():
    psnr_cells = _load_psnr_cells()
    top1_cells = _load_top1_cells()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    ax_psnr, ax_top1 = axes

    # -- Left panel: PSNR vs bpp ------------------------------------------
    for codec_key in ("jpeg", "jpeg_sub0", "seaotter"):
        pts = psnr_cells[codec_key]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax_psnr.plot(
            xs, ys,
            color=COLORS[codec_key],
            label=LABELS[codec_key],
            marker="o", markersize=5, linewidth=1.8,
        )
    _set_log_x_with_plain_ticks(ax_psnr)
    ax_psnr.set_xlabel("Bits per pixel")
    ax_psnr.set_ylabel("PSNR (dB)")
    ax_psnr.set_title("(a) Kodak val (24 images, native res)")
    ax_psnr.grid(True, which="both", alpha=0.4, zorder=0)
    ax_psnr.set_axisbelow(True)
    ax_psnr.legend(loc="best", fontsize=11)

    # -- Right panel: top-1 vs bpp ----------------------------------------
    any_top1 = any(top1_cells.values())

    # Raw (no-codec) ceiling, drawn first so codec lines render on top.
    raw_path = CLS_KODAK_DIR / "eval_raw_cls_kodak_anchor.json"
    if raw_path.exists():
        raw_top1 = float(json.loads(raw_path.read_text())["metrics"]["top1"])
        ax_top1.axhline(
            raw_top1, linestyle="--", color="dimgray", linewidth=1.0,
            label=f"Raw (no codec) = {raw_top1*100:.2f}\\%",
            zorder=1,
        )

    for codec_key in ("jpeg", "jpeg_sub0", "seaotter"):
        pts = top1_cells.get(codec_key, [])
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
        else:
            # Mirror the PSNR x-axis so the legend / line colors line up; y=NaN
            # so matplotlib draws nothing.
            xs = [p[0] for p in psnr_cells[codec_key]]
            ys = [np.nan for _ in xs]
        ax_top1.plot(
            xs, ys,
            color=COLORS[codec_key],
            label=LABELS[codec_key],
            marker="o", markersize=5, linewidth=1.8,
        )
    _set_log_x_with_plain_ticks(ax_top1)
    ax_top1.set_xlabel("Bits per pixel")
    ax_top1.set_ylabel("ImageNet top-1")
    ax_top1.set_title("(b) ImageNet val (50k, squash 384²)")
    # When real data exists, zoom in to the observed range so we don't waste
    # 90% of the plot on whitespace; the raw ceiling at ~0.85 stays in view.
    if any_top1:
        ax_top1.set_ylim(0.80, 0.86)
    else:
        ax_top1.set_ylim(0.0, 1.0)
    ax_top1.grid(True, which="both", alpha=0.4, zorder=0)
    ax_top1.set_axisbelow(True)
    ax_top1.legend(loc="best", fontsize=11)
    if not any_top1:
        ax_top1.text(
            0.5, 0.5,
            "[Top-1 results in progress]",
            transform=ax_top1.transAxes,
            ha="center", va="center",
            fontsize=14, color="dimgray",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", edgecolor="dimgray", alpha=0.85),
        )

    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "codec_kodak.pdf"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=160)
    print(f"WROTE {out_path}")

    # Quick console table for the closeout.
    print("\n--- codec_kodak: PSNR-side cells ---")
    for codec_key in ("seaotter", "jpeg_sub0", "jpeg"):
        for bpp, psnr in psnr_cells[codec_key]:
            print(f"  {codec_key:>10}  bpp={bpp:6.3f}  PSNR={psnr:6.2f} dB")


if __name__ == "__main__":
    main()
