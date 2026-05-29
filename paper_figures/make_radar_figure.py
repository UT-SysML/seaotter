"""Radar/spider plot summarizing SEAOTTER vs AVIF, ITU JPEG, and FRAPPE.

Rankings (1 = worst, 2 = poor, 3 = fair, 4 = good, 5 = best) are
hand-curated by the user and baked in; no eval JSONs are read.

Two SEAOTTER variants are drawn:

- `SEAOTTER-FT` — fine-tuned for each downstream task. Solid line.
  Wins on cls / seg / clip; loses on PSNR / DISTS due to task-driven
  recon drift.
- `SEAOTTER-ZS` — zero-shot (no task fine-tuning). Dotted line, same
  blueviolet color. Keeps near-best behavior on every axis because
  the zero-shot codec is trained against an R-D loss, not a task
  loss.

Justification (per-axis):

- `trans`: AVIF doesn't extend to low bitrates; FRAPPE / SEAOTTER-*
  do. FRAPPE / SEAOTTER-FT / SEAOTTER-ZS share the same encoder, so
  they coincide on the transmit-bpp axis.
- `store`: FRAPPE wins (no transcode, neural latents are compact).
  SEAOTTER-FT and SEAOTTER-ZS take a moderate storage hit from the
  one-time JPEG transcode; AVIF's storage is mediocre because its
  default operating points don't reach low storage_bpp.
- `PSNR`:  Downstream fine-tuning wrecks SEAOTTER-FT's PSNR. The
  zero-shot variant keeps PSNR competitive.
- `DISTS`: Same fine-tuning trade-off — SEAOTTER-FT sacrifices
  perceptual quality for task accuracy; SEAOTTER-ZS and FRAPPE
  both stay strong.
- `cls` / `seg` / `clip`: SEAOTTER-FT best (fine-tuned per task);
  SEAOTTER-ZS and FRAPPE second; codec-only baselines trail.
- `enc`:   AVIF default is slow on CPU. ITU JPEG, FRAPPE, and both
  SEAOTTER variants tie — the encoder is a tiny conv plus DCT +
  JPEG entropy coding, on par with libjpeg.
- `dec`:   FRAPPE requires a full neural decoder → poor on CPU.
  AVIF and ITU decode are fast on CPU. SEAOTTER's on-disk artifact
  *is* a JPEG file, so its decode path is plain libjpeg + a tiny
  inverse conv, on par with ITU / AVIF — same for both ZS and FT
  variants.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        13,
    "axes.titlesize":   14,
    "axes.labelsize":   13,
    "legend.fontsize":  10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  11,
})

# Nine axes, clockwise from the top. Order matches the RANKINGS
# columns below.
METRICS = [
    "Sensor-embedded compression ratio",
    "Downstream consumer compression ratio",
    "PSNR",
    "DISTS",
    "Classification Acc.",
    "Segmentation mIoU",
    "SigLIP Acc.",
    "Sensor-embedded encoding",
    "Downstream consumer decoding throughput",
]

# 1 = worst, 2 = poor, 3 = fair, 4 = good, 5 = best.
#
# Data-driven version (v3): each axis is the linear normalization (or
# log for enc/dec) of a single scalar pulled from Table 1's
# matched-rate picks + the matching eval-JSON distortion / throughput
# values. See `notes/radar_rankings_data_driven.md` for the full
# derivation, the per-axis raw scalars, and the structural reasons
# this view differs from the hand-curated v2.
RANKINGS = {
    #             trans store PSNR DISTS cls seg clip enc dec
    "AVIF":        [4,   4,   5,   4,    4,  5,  4,   1,  3],
    "ITU":         [1,   2,   4,   1,    1,  1,  1,   5,  5],
    "FRAPPE":      [5,   5,   5,   5,    4,  4,  4,   3,  1],
    "SEAOTTER-ZS": [5,   1,   5,   5,    4,  4,  4,   3,  4],
    "SEAOTTER-FT": [5,   1,   1,   2,    5,  5,  5,   3,  4],
}

# Draw order is back-to-front (legend follows the same order). Both
# SEAOTTER variants are blueviolet; ZS uses a dotted line so it
# remains visually distinct from the solid-line fine-tuned variant.
# SEAOTTER-FT is the paper's hero system and sits on top.
SYSTEMS = [
    {"key": "ITU",         "label": "ITU JPEG (4:4:4)", "color": "tab:gray",    "linestyle": "-"},
    {"key": "AVIF",        "label": "AVIF (default)",   "color": "black",       "linestyle": "-"},
    {"key": "FRAPPE",      "label": "FRAPPE",           "color": "dodgerblue",  "linestyle": "-"},
    {"key": "SEAOTTER-ZS", "label": "SEAOTTER-ZS",      "color": "blueviolet",  "linestyle": ":"},
    {"key": "SEAOTTER-FT", "label": "SEAOTTER-FT",      "color": "blueviolet",  "linestyle": "-"},
]

TIER_LABELS = ["worst", "poor", "fair", "good", "best"]


def make_radar(out_dir: Path, no_text: bool = False) -> None:
    """Render the radar. When `no_text=True`, suppress axis labels,
    tier annotations, and the legend so labels can be placed manually
    in a slide. Polygon positions / fills / linestyles are identical
    to the labeled version so manual labels can be added at the same
    relative positions."""
    n = len(METRICS)
    angles = [2 * np.pi * k / n for k in range(n)]
    # Close the polygon by appending the first angle / vertex.
    angles_closed = angles + [angles[0]]

    fig = plt.figure(figsize=(11.0, 9.5), dpi=150)
    ax = fig.add_subplot(111, projection="polar")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)  # clockwise

    # Per-system polygons.
    for sys in SYSTEMS:
        ranks = RANKINGS[sys["key"]]
        values = list(ranks) + [ranks[0]]
        ax.plot(
            angles_closed,
            values,
            color=sys["color"],
            linestyle=sys["linestyle"],
            linewidth=2.0,
            alpha=0.95,
            marker="o",
            markersize=4,
            label=sys["label"],
            zorder=3,
        )
        ax.fill(
            angles_closed,
            values,
            color=sys["color"],
            alpha=0.18,
            zorder=2,
        )

    # Axis tick labels at the nine evenly-spaced angles.
    ax.set_xticks(angles)
    ax.set_xticklabels([""] * n if no_text else METRICS)

    # Radial scale: 0..5 plus a hair of headroom.
    ax.set_ylim(0, 5.3)

    # Radial gridlines at integer tiers, annotated with qualitative
    # labels at a single azimuth (180° clockwise from north, i.e.
    # halfway between the `cls` axis at 160° and `seg` at 200°).
    # cls and seg rankings coincide for every system except
    # SEAOTTER-FT (cls=5, seg=4), so most polygons run circumferentially
    # through this arc at their own integer radius and the tier labels
    # mostly land on polygon edges rather than crossing them.
    _, tier_texts = ax.set_rgrids(
        [1, 2, 3, 4, 5],
        labels=[""] * 5 if no_text else TIER_LABELS,
        angle=180,
        fontsize=10,
        color="black",
    )
    if not no_text:
        # White background bbox keeps labels readable on top of the
        # overlapping semi-transparent polygons.
        for t in tier_texts:
            t.set_bbox(dict(boxstyle="round,pad=0.18", facecolor="white",
                            edgecolor="none", alpha=0.88))
            t.set_zorder(4)

    # Light grid styling under the polygons.
    ax.yaxis.grid(True, linestyle="--", color="gray", alpha=0.30, zorder=0)
    ax.xaxis.grid(True, linestyle="--", color="gray", alpha=0.30, zorder=0)
    ax.spines["polar"].set_alpha(0.35)
    ax.set_axisbelow(True)

    # Legend outside the polar circle, lower-right (suppressed in
    # no-text mode so the slide deck can lay out its own legend).
    if not no_text:
        ax.legend(
            loc="lower right",
            bbox_to_anchor=(1.22, -0.02),
            frameon=True,
            framealpha=0.9,
        )

    fig.tight_layout()

    stem = "radar_summary_no_text" if no_text else "radar_summary"
    out_pdf = out_dir / f"{stem}.pdf"
    out_png = out_dir / f"{stem}.png"
    out_svg = out_dir / f"{stem}.svg"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=160)
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE {out_pdf}")
    print(f"WROTE {out_png}")
    print(f"WROTE {out_svg}")


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    make_radar(out_dir)
    make_radar(out_dir, no_text=True)


if __name__ == "__main__":
    main()
