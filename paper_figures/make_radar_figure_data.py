"""Data-driven sibling of `make_radar_figure.py`.

Same layout, fonts, colors, draw order, and tier-label placement as
the hand-curated figure; only the RANKINGS dict differs. Values
come from `notes/_compute_radar_rankings_v2.py` (see
`notes/radar_rankings_v2.md` for the per-axis derivation):

- RD axes (`trans`, `store`, `PSNR`, `DISTS`, `cls`, `seg`, `clip`):
  BD-rate vs ITU JPEG, normalised under the degeneracy-aware
  [D+1, 5] rule.
- Throughput axes (`enc`, `dec`): log-MPx/s normalisation (the
  primary linear-MPx/s scoring in v2 collapses the three neural
  codecs to ~2.1; log-MPx/s preserves the spread and matches the
  deployment-tier framing).
- `trans` uses the machine-perception aggregate (geomean of
  cls top-1 / seg mIoU / clip top-1, each normalised by its raw
  ceiling) so the axis is genuinely distinct from `cls`.

Caveats baked into the data (called out in the v2 report):

- `Consumer compression ratio` uses BD-rate vs `frp_jpeg` (FRAPPE
  on-device + ITU JPEG consumer-side recompression), not vanilla
  ITU JPEG. The axis story is "consumer storage assuming the image
  was already heavily compressed on the device" — comparing
  vanilla JPEG (no on-device front-end) to a consumer-side
  transcode is not apples-to-apples. AVIF and vanilla ITU JPEG are
  therefore flagged degenerate on this axis and get score 1.0.
- `dec` for the SEAOTTER variants uses a steady-state proxy: the
  JPEG consumer timing, because the harness's
  `decode_only_consumer` for the SEAOTTER pipelines runs the full
  first-receive transcode path (FRAPPE neural decode → SEAOTTER
  forward → JPEG encode/decode → SEAOTTER inverse → ConvNeXt) and
  therefore over-counts SEAOTTER's deployed cost. In production,
  the transcode runs once on the cloud and downstream readers read
  the JPEG file — so `JPEG consumer time + ~0.1 ms inverse conv`
  is the correct steady-state metric. The 0.1 ms inverse-conv
  delta is below the harness's run-to-run noise; we use the JPEG
  geomean throughput verbatim as the SEAOTTER proxy.

The output PDF is written to `figures/radar_summary_data.pdf` so it
does not overwrite the hand-curated `radar_summary.pdf`.
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

# Hand-curated values (kept for reference; the data-driven RANKINGS
# below replaces them).
#
# HARDCODED_RANKINGS = {
#     #             trans store PSNR DISTS cls seg clip enc dec
#     "AVIF":        [2,   2,   4,   4,    2,  2,  2,   1,  4],
#     "ITU":         [1,   1,   2,   2,    1,  1,  1,   5,  5],
#     "FRAPPE":      [3,   5,   3,   4,    3,  3,  3,   4,  1],
#     "SEAOTTER-ZS": [4,   3,   5,   5,    4,  4,  4,   4,  5],
#     "SEAOTTER-FT": [5,   4,   1,   1,    5,  4,  5,   4,  5],
# }
#
# Data-driven rankings (v3): each axis is a single principled scalar
# from Table 1's per-task matched-rate picks (FRAPPE n_ch=12 anchor
# on cls/seg/clip), linear-normalised to [1, 5] over the five-system
# range. Throughput axes (`enc`, `dec`) use log normalisation because
# raw MPx/s spans ~2 orders of magnitude.
#
# `enc` now reads from the encoder-only mirror at
# `~/UT-SysML/seaotter/results/encode_complexity/cls_384/encode_*.json`
# (timer excludes PIL decode / dtype cast preamble), with `seab`/`seaft`
# aliased to `frp` (shared FRAPPE encoder). `dec` still reads the
# iter-11 cls/throughput JSONs (codec-only steady-state decode). The
# RANKINGS dict is computed at module-import time by calling
# `notes/_compute_radar_rankings_v3.compute_rankings`, so any update
# to the underlying mirror data flows directly into the figure.
#
# Hand-curated fallback (kept for documentation; not used unless the
# import below fails):
#
# HARDCODED_RANKINGS_V3 = {
#     #               trans  store  PSNR  DISTS  cls   seg   clip  enc   dec
#     "AVIF":        [3.41,  3.90,  4.93, 4.18,  4.26, 4.95, 4.46, 1.00, 3.29],
#     "ITU":         [1.00,  2.23,  3.89, 1.00,  1.00, 1.00, 1.00, 5.00, 5.00],
#     "FRAPPE":      [5.00,  5.00,  5.00, 4.72,  3.82, 4.30, 4.09, 2.63, 1.00],
#     "SEAOTTER-ZS": [5.00,  1.00,  4.99, 5.00,  4.19, 4.49, 4.34, 2.63, 4.11],
#     "SEAOTTER-FT": [5.00,  1.37,  1.00, 2.27,  5.00, 5.00, 5.00, 2.63, 4.14],
# }
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent / "notes"))
from _compute_radar_rankings_v3 import compute_rankings as _compute_rankings  # noqa: E402

_RAW, RANKINGS = _compute_rankings()

# Same draw order, colors, and linestyles as the hand-curated figure.
SYSTEMS = [
    {"key": "ITU",         "label": "ITU JPEG (4:4:4)", "color": "tab:gray",    "linestyle": "-"},
    {"key": "AVIF",        "label": "AVIF (max-speed)", "color": "black",       "linestyle": "-"},
    {"key": "FRAPPE",      "label": "FRAPPE",           "color": "dodgerblue",  "linestyle": "-"},
    {"key": "SEAOTTER-ZS", "label": "SEAOTTER-ZS",      "color": "blueviolet",  "linestyle": ":"},
    {"key": "SEAOTTER-FT", "label": "SEAOTTER-FT",      "color": "blueviolet",  "linestyle": "-"},
]

TIER_LABELS = ["worst", "poor", "fair", "good", "best"]


def make_radar(out_dir: Path, no_text: bool = False) -> None:
    """Render the radar. When `no_text=True`, suppress axis labels,
    tier annotations, and the legend so labels can be placed manually
    in a slide deck."""
    n = len(METRICS)
    angles = [2 * np.pi * k / n for k in range(n)]
    angles_closed = angles + [angles[0]]

    fig = plt.figure(figsize=(11.0, 9.5), dpi=150)
    ax = fig.add_subplot(111, projection="polar")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

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

    ax.set_xticks(angles)
    ax.set_xticklabels([""] * n if no_text else METRICS)
    ax.set_ylim(0, 5.3)

    # Radial gridlines + tier labels at angle=180° (same azimuth as
    # the hand-curated figure). With fractional scores the polygons
    # no longer coincide cleanly on the cls↔seg arc, but the white-
    # bboxed labels stay readable on top of overlapping polygons.
    _, tier_texts = ax.set_rgrids(
        [1, 2, 3, 4, 5],
        labels=[""] * 5 if no_text else TIER_LABELS,
        angle=180,
        fontsize=10,
        color="black",
    )
    if not no_text:
        for t in tier_texts:
            t.set_bbox(dict(boxstyle="round,pad=0.18", facecolor="white",
                            edgecolor="none", alpha=0.88))
            t.set_zorder(4)

    ax.yaxis.grid(True, linestyle="--", color="gray", alpha=0.30, zorder=0)
    ax.xaxis.grid(True, linestyle="--", color="gray", alpha=0.30, zorder=0)
    ax.spines["polar"].set_alpha(0.35)
    ax.set_axisbelow(True)

    if not no_text:
        ax.legend(
            loc="lower right",
            bbox_to_anchor=(1.22, -0.02),
            frameon=True,
            framealpha=0.9,
        )

    fig.tight_layout()

    stem = "radar_summary_data_no_text" if no_text else "radar_summary_data"
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
