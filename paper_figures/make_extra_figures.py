"""Emit two supplementary figures referenced from the Results section.

Outputs (under `./figures/`):

- `rd_metric_panels.pdf` — 2x3 grid: top row PSNR / SSIM / DISTS (all on
  cls eval JSONs, distortion vs compression ratio); bottom row task
  accuracy: ImageNet top-1 (cls), ADE20K mIoU (seg), ImageNet zero-shot
  top-1 via SigLIP-2 (clip). Both vanilla codecs (`wal`, `frp`) and the
  SEAOTTER variants are present on the bottom row.
- `rd_storage.pdf` — 2x3 grid. Top row: cls top-1, seg mIoU, clip top-1,
  each vs *storage* bpp (the size of the on-disk JPEG byte stream
  produced by the cloud-side transcode). Bottom row: per-task plot of
  transmit CR vs consumer (storage) CR for the same five pipelines,
  with a y=x reference line; SEAOTTER variants sit well above y=x
  (transmit much more compressed than the consumer artifact). For
  non-SEAOTTER baselines, storage_bpp equals transmit_bpp; for SEAOTTER
  pipelines, the two diverge.

All data is pulled from `~/UT-SysML/seaotter/results/` (the canonical
paper-data mirror).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        13,
    "axes.titlesize":   14,
    "axes.labelsize":   13,
    "legend.fontsize":  10,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
})

RESULTS = Path("/home/dgj335/UT-SysML/seaotter/results")

# bpp -> compression ratio convention: uncompressed RGB is 24 bpp
# (3 channels x 8 bits), so CR = 24 / bpp.
def cr_from_bpp(bpp: float) -> float:
    return 24.0 / bpp if bpp and bpp > 0 else float("inf")


def load_pipeline_rate_pair(short: str, task: str,
                            *, x_bpp_key: str = "storage_bpp_mean",
                            y_bpp_key: str = "transmit_bpp_mean"):
    """Load (consumer_CR, transmit_CR) pairs for a given pipeline + task.

    One point per eval JSON / operating point. Both bpp values must be
    present; if either is missing the point is skipped.
    """
    eval_dir = RESULTS / task / "eval"
    points = []
    for p in sorted(eval_dir.glob(f"eval_{short}_{task}_*.json")):
        d = json.loads(p.read_text())
        xb = d.get(x_bpp_key)
        yb = d.get(y_bpp_key)
        if xb is None or yb is None:
            continue
        points.append({
            "x_CR": cr_from_bpp(float(xb)),
            "y_CR": cr_from_bpp(float(yb)),
        })
    points.sort(key=lambda d: d["x_CR"])
    return points


def load_pipeline(short: str, task: str, metric_key: str, *, bpp_key: str = "transmit_bpp_mean"):
    """Load (bpp/CR/metric) points for a given pipeline + task.

    metric_key is looked up in `metrics`. Points with the metric absent are
    dropped.
    """
    eval_dir = RESULTS / task / "eval"
    points = []
    for p in sorted(eval_dir.glob(f"eval_{short}_{task}_*.json")):
        d = json.loads(p.read_text())
        bpp = d.get(bpp_key)
        m = (d.get("metrics") or {}).get(metric_key)
        if bpp is None or m is None:
            continue
        points.append({
            "bpp": float(bpp),
            "CR":  cr_from_bpp(float(bpp)),
            metric_key: float(m),
        })
    points.sort(key=lambda d: d["bpp"])
    return points


# Pipeline visual style (mirrors make_figures.py's main_results.pdf).
# AVIF (default) is shown on main_results.pdf only; rd_metric_panels.pdf
# keeps AVIF (max speed) as the lone AVIF representative.
# Top-row distortion (PSNR/SSIM/DISTS, all on cls): walft excluded — its
# task-driven recon drift makes pixel-fidelity comparisons misleading.
PIPELINES_TOP_DISTORTION = [
    {"short": "avifx",   "name": "AVIF (max speed)",     "color": "black",      "linestyle": "--"},
    {"short": "wal",     "name": "WaLLoC",               "color": "goldenrod",  "linestyle": "-"},
    {"short": "frp",     "name": "FRAPPE",               "color": "dodgerblue", "linestyle": "-"},
    {"short": "walsand", "name": "WaLLoC-SEAOTTER-ZS",   "color": "orchid",     "linestyle": "--"},
    {"short": "seab",    "name": "FRAPPE-SEAOTTER-ZS",   "color": "blueviolet", "linestyle": "--"},
    {"short": "seaft",   "name": "FRAPPE-SEAOTTER-FT",   "color": "blueviolet", "linestyle": "-"},
]

# Bottom-row task accuracy (top1 cls / mIoU seg / top1 clip). Both vanilla
# no-transcode codecs (`wal`, `frp`) and the SEAOTTER variants are present;
# `walft` omitted (out of scope on seg/clip for this paper).
PIPELINES_BOTTOM_TASK = [
    {"short": "avifx",   "name": "AVIF (max speed)",     "color": "black",      "linestyle": "--"},
    {"short": "wal",     "name": "WaLLoC",               "color": "goldenrod",  "linestyle": "-"},
    {"short": "frp",     "name": "FRAPPE",               "color": "dodgerblue", "linestyle": "-"},
    {"short": "walsand", "name": "WaLLoC-SEAOTTER-ZS",   "color": "orchid",     "linestyle": "--"},
    {"short": "seab",    "name": "FRAPPE-SEAOTTER-ZS",   "color": "blueviolet", "linestyle": "--"},
    {"short": "seaft",   "name": "FRAPPE-SEAOTTER-FT",   "color": "blueviolet", "linestyle": "-"},
]

# Storage-axis figure: all lines share a `FRAPPE-or-WaLLoC + JPEG`
# architecture so the comparison is apples-to-apples on storage_bpp.
# Codec family is encoded by hue (`goldenrod` = WaLLoC, `dodgerblue` =
# FRAPPE); within a family, linestyle distinguishes `<codec>+ITU JPEG`
# (`--`) from vanilla `<codec>` (`-`, shown on fig 4 only — vanilla
# codecs have transmit==storage, so they're architecturally unfair on
# the storage-CR axis here). `walft` omitted across all 3 panels: it
# has cls data only, so showing it would imply seg/clip coverage that
# doesn't exist.
PIPELINES_STORAGE = [
    {"short": "wal_jpeg", "name": "WaLLoC + ITU JPEG",   "color": "goldenrod",  "linestyle": "--"},
    {"short": "frp_jpeg", "name": "FRAPPE + ITU JPEG",   "color": "dodgerblue", "linestyle": "--"},
    {"short": "walsand",  "name": "WaLLoC-SEAOTTER-ZS",  "color": "orchid",     "linestyle": "--"},
    {"short": "seab",     "name": "FRAPPE-SEAOTTER-ZS",  "color": "blueviolet", "linestyle": "--"},
    {"short": "seaft",    "name": "FRAPPE-SEAOTTER-FT",  "color": "blueviolet", "linestyle": "-"},
]


def draw_panel(ax, pipelines, task, metric_key, *, x_key="CR", x_label="Sensor-embedded compression ratio",
               y_label=None, title=None, log_y=False, bpp_key="transmit_bpp_mean"):
    for c in pipelines:
        pts = load_pipeline(c["short"], task, metric_key, bpp_key=bpp_key)
        if not pts:
            continue
        xs = [p[x_key] for p in pts]
        ys = [p[metric_key] for p in pts]
        ax.plot(xs, ys,
                linestyle=c["linestyle"],
                color=c["color"],
                marker="o",
                markersize=4,
                linewidth=1.7,
                alpha=0.9,
                label=c["name"])
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title)
    ax.grid(True, which="both", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)


def draw_rate_pair_panel(ax, pipelines, task, *, title=None):
    """Plot transmit CR vs consumer CR per pipeline / operating point."""
    all_x, all_y = [], []
    for c in pipelines:
        pts = load_pipeline_rate_pair(c["short"], task)
        if not pts:
            continue
        xs = [p["x_CR"] for p in pts]
        ys = [p["y_CR"] for p in pts]
        all_x.extend(xs); all_y.extend(ys)
        ax.plot(xs, ys,
                linestyle=c["linestyle"],
                color=c["color"],
                marker="o",
                markersize=4,
                linewidth=1.7,
                alpha=0.9,
                label=c["name"])
    # y=x reference line: above the line => transmit more compressed
    # (cheaper uplink) than consumer-side storage; below the line =>
    # storage smaller than transmit (rare for SEAOTTER pipelines).
    # X-axis (consumer CR) is clamped to [10, 60] to match the row-1
    # panels; y-axis (transmit CR) is fixed to [30, 3000] across all
    # three panels so the SEAOTTER family's gap to y=x is comparable.
    x_lo, x_hi = 10.0, 60.0
    y_lo, y_hi = 30.0, 3000.0
    diag_lo = min(x_lo, y_lo)
    diag_hi = max(x_hi, y_hi)
    ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi],
            color="gray", linestyle=":", linewidth=1.0,
            alpha=0.6, zorder=0, label=None)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Downstream consumer compression ratio")
    ax.set_ylabel("Sensor-embedded transmit CR")
    if title is not None:
        ax.set_title(title)
    ax.grid(True, which="both", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)


def emit_rd_metric_panels(out_dir: Path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=150)
    # Top row: pixel-fidelity distortion on ImageNet val (cls task path).
    draw_panel(axes[0][0], PIPELINES_TOP_DISTORTION, task="cls", metric_key="psnr_db",
               y_label="PSNR (dB)",                  title="(a) PSNR (ImageNet val)")
    draw_panel(axes[0][1], PIPELINES_TOP_DISTORTION, task="cls", metric_key="ssim",
               y_label="SSIM",                       title="(b) SSIM (ImageNet val)")
    draw_panel(axes[0][2], PIPELINES_TOP_DISTORTION, task="cls", metric_key="dists_db",
               y_label="DISTS (dB)",                 title="(c) DISTS (ImageNet val)")
    # Bottom row: downstream task accuracy across cls / seg / clip.
    draw_panel(axes[1][0], PIPELINES_BOTTOM_TASK,    task="cls",  metric_key="top1",
               y_label="ImageNet top-1",             title=r"(d) Classification ($384^2$ px)")
    draw_panel(axes[1][1], PIPELINES_BOTTOM_TASK,    task="seg",  metric_key="miou",
               y_label="ADE20K mIoU",                title=r"(e) Segmentation ($512^2$ px)")
    draw_panel(axes[1][2], PIPELINES_BOTTOM_TASK,    task="clip", metric_key="top1",
               y_label="ImageNet zero-shot top-1",   title="(f) SigLIP-2 (variable aspect, 256 tokens)")
    axes[0][0].legend(loc="best", fontsize=9)
    axes[1][0].legend(loc="best", fontsize=9)
    fig.tight_layout()
    out = out_dir / "rd_metric_panels.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=160)
    print(f"WROTE {out}")


def emit_rd_storage(out_dir: Path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=150)
    # Row 1: task accuracy vs consumer CR.
    draw_panel(axes[0][0], PIPELINES_STORAGE, task="cls",  metric_key="top1",
               x_label="Downstream consumer compression ratio", y_label="ImageNet top-1",
               title="(a) Global prediction (ImageNet)", bpp_key="storage_bpp_mean")
    draw_panel(axes[0][1], PIPELINES_STORAGE, task="seg",  metric_key="miou",
               x_label="Downstream consumer compression ratio", y_label="ADE20K mIoU",
               title="(b) Dense prediction (ADE20K)", bpp_key="storage_bpp_mean")
    draw_panel(axes[0][2], PIPELINES_STORAGE, task="clip", metric_key="top1",
               x_label="Downstream consumer compression ratio", y_label="ImageNet zero-shot top-1",
               title="(c) VLM/VLA-style zero-shot (SigLIP-2)", bpp_key="storage_bpp_mean")
    # Row 2: transmit CR vs consumer CR for the same three settings.
    draw_rate_pair_panel(axes[1][0], PIPELINES_STORAGE, task="cls",
                         title="(d) Global prediction (ImageNet)")
    draw_rate_pair_panel(axes[1][1], PIPELINES_STORAGE, task="seg",
                         title="(e) Dense prediction (ADE20K)")
    draw_rate_pair_panel(axes[1][2], PIPELINES_STORAGE, task="clip",
                         title="(f) VLM/VLA-style zero-shot (SigLIP-2)")
    # Plain integer ticks instead of 10^1 / 2x10^1 — the storage CR axis
    # spans less than one decade, so scientific notation is overkill.
    plain = FuncFormatter(lambda x, pos: f"{x:g}")
    for ax in axes[0]:
        ax.xaxis.set_major_formatter(plain)
        ax.xaxis.set_minor_formatter(plain)
    # Row 2: x-axis is clamped to a sub-decade (10--60) so label minor
    # ticks plain too; y-axis spans multiple decades, so only label majors.
    for ax in axes[1]:
        ax.xaxis.set_major_formatter(plain)
        ax.xaxis.set_minor_formatter(plain)
        ax.yaxis.set_major_formatter(plain)
    axes[0][0].legend(loc="best", fontsize=9)
    fig.tight_layout()
    out = out_dir / "rd_storage.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=160)
    print(f"WROTE {out}")


def main():
    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    emit_rd_metric_panels(out_dir)
    emit_rd_storage(out_dir)


if __name__ == "__main__":
    main()
