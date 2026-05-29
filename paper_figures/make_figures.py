"""Render the two-panel main_results.pdf for draft_results_v2.

Reuses the loading + plotting logic from
`../seaotter/pre_trained_convnext/experiments/iter6_extra_codec_baselines/pipelines.ipynb`
but composes the two panels into a single `figures/main_results.pdf`
via matplotlib's `subplots(1, 2)`.

Run from anywhere; outputs land in `./figures/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        14,
    "axes.titlesize":   16,
    "axes.labelsize":   14,
    "legend.fontsize":  11,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
})

RESULTS = Path("/home/dgj335/UT-SysML/seaotter/results")
EVAL_DIR = RESULTS / "cls" / "eval"
ENC_DIR = RESULTS / "encode_complexity" / "cls_384"
CROP = 384  # cls squash

# Pipeline -> encoder-only codec short. seab/seaft alias to the frozen
# FRAPPE encoder; walsand/walft alias to the frozen WaLLoC encoder.
# Mirrors `make_tables.py`'s PIPELINE_TO_ENC_CODEC.
PIPELINE_TO_ENC_CODEC = {
    "frp":     "frp",
    "seab":    "frp",
    "seaft":   "frp",
    "wal":     "wal",
    "walsand": "wal",
    "walft":   "wal",
    "avif":    "avif",
    "avifx":   "avifx",
    "jpeg":    "jpeg",
    "jp2":     "jp2",
    "webp":    "webp",
}


def _load_enc_data(codec):
    """Read every `encode_<codec>_*.json` under cls_384/, keyed by op value."""
    out = {}
    if codec is None:
        return out
    for p in sorted(ENC_DIR.glob(f"encode_{codec}_*.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        op = d.get("operating_point", {})
        out[(op.get("type"), op.get("value"))] = d
    return out


def _enc_mpx(enc_data, op_type, op_value):
    """Look up encoder-only median MPx/s for a given (type, value)."""
    for key in (
        (op_type, op_value),
        (op_type, int(op_value) if isinstance(op_value, float) and op_value.is_integer() else op_value),
        (op_type, float(op_value)),
    ):
        d = enc_data.get(key)
        if d is not None:
            return ((d.get("results") or {}).get("throughput") or {}).get("median_MPx_per_s")
    return None

AXIS_LABEL = {
    "bpp":            "Transmit bpp",
    "CR":             "On-device compression ratio",
    "throughput_MPx": "Encoding throughput [MPx/s]",
    "top1":           "ImageNet top-1",
}
TITLE_LABEL = {
    "bpp":            "bpp",
    "CR":             "on-device compression ratio",
    "throughput_MPx": "Throughput",
    "top1":           "Downstream accuracy",
}
LOG_KEYS = ("bpp", "CR", "throughput_MPx")


def load_pipeline(short, task="cls"):
    """Eval data from the UT-SysML mirror; encoder-only throughput from
    the encode_complexity mirror (FRAPPE-paper methodology: inputs
    pre-staged in encoder-native form). SEAOTTER-family pipelines
    alias to their shared encoder (`seab`/`seaft` -> `frp`,
    `walsand`/`walft` -> `wal`).
    """
    if task != "cls":
        raise NotImplementedError(f"task={task!r}: only cls is wired here")
    enc_codec = PIPELINE_TO_ENC_CODEC.get(short)
    enc_data = _load_enc_data(enc_codec)
    points = []
    for p in sorted(EVAL_DIR.glob(f"eval_{short}_{task}_*.json")):
        prefix = f"eval_{short}_{task}_"
        opid = p.stem[len(prefix):]
        acc = json.loads(p.read_text())
        op = acc.get("operating_point") or {}
        mpx = _enc_mpx(enc_data, op.get("type"), op.get("value"))
        if mpx is None or mpx <= 0:
            # No encoder-only measurement for this op; skip rather than
            # silently fall back to a different methodology.
            continue
        bpp = float(acc["transmit_bpp_mean"])
        points.append({
            "opid":           opid,
            "bpp":            bpp,
            # Standard compression-ratio convention: uncompressed RGB is
            # 24 bpp (3 channels x 8 bits), so CR = 24 / bpp.
            "CR":             24.0 / bpp if bpp > 0 else float("inf"),
            "throughput_MPx": mpx,
            "top1":           (acc.get("metrics") or {}).get("top1"),
        })
    points.sort(key=lambda d: d["bpp"])
    return points


def col(pts, key):
    return [p[key] for p in pts]


# Deployment-tier thresholds reused across panels for the viability glyphs.
# (CR_min, throughput_min) — CR = 24 / bpp, throughput in MPx/s.
DEPLOYMENT_TIERS = [
    (288, 12),  # BLE, 480p, 30 fps
    (133, 28),  # 5G, 720p, 30 fps
    ( 60, 62),  # Wi-Fi, 1080p, 30 fps
]


def viability_count(cr, tp_mpx):
    return sum(1 for cr_min, tp_min in DEPLOYMENT_TIERS
               if cr >= cr_min and tp_mpx >= tp_min)


def viability_marker(count):
    """Returns (marker, size) for the per-point glyph.

    - 0 regions cleared: square (stop glyph)
    - 1 or 2 cleared:    single right-chevron ">"
    - all 3 cleared:     math much-greater-than "$\\gg$"
    """
    if count == 0:
        return ("s", 7)
    if count == 3:
        return (r"$\gg$", 14)
    return (">", 9)


def draw_panel(ax, group, y_key, x_key, *, legend=False):
    if x_key in LOG_KEYS:
        ax.set_xscale("log")
    if y_key in LOG_KEYS:
        ax.set_yscale("log")
    for c in group:
        if not c["data"]:
            continue
        xs = col(c["data"], x_key)
        ys = col(c["data"], y_key)
        # Connecting line (transparent enough that the deployment-tier
        # shading on panel (a) reads through where they overlap).
        ax.plot(xs, ys,
                linestyle=c.get("linestyle", "-"),
                color=c.get("color"),
                alpha=0.9,
                linewidth=1.8,
                label=c["name"])
        # Per-point viability glyphs — same shape on both panels, computed
        # once per operating point from (CR, throughput_MPx).
        for p in c["data"]:
            cnt = viability_count(p["CR"], p["throughput_MPx"])
            m, ms = viability_marker(cnt)
            ax.plot(p[x_key], p[y_key],
                    marker=m,
                    markersize=ms,
                    color=c.get("color"),
                    alpha=0.85,
                    linestyle="none",
                    zorder=3)
    ax.set_xlabel(AXIS_LABEL[x_key])
    ax.set_ylabel(AXIS_LABEL[y_key])
    ax.set_title(f"{TITLE_LABEL[y_key]} vs {TITLE_LABEL[x_key]}")
    if legend:
        # Render the legend below the plotted data lines/markers but above
        # the gridlines: legend zorder sits between grid (z=0) and lines (z=2).
        leg = ax.legend(loc="best", fontsize=11)
        leg.set_zorder(1.5)
    log_either = (x_key in LOG_KEYS) or (y_key in LOG_KEYS)
    ax.grid(True, which="both" if log_either else "major", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)


def main():
    # Plot pipelines. Colors follow the FRAPPE paper convention with
    # SEAOTTER (the proposed fine-tuned method) split out as its own
    # color (purple) to distinguish it from the FRAPPE family. Linestyle
    # ranking within a family follows accuracy at matched bpp:
    # solid (higher) > dashed (lower). JPEG is omitted from the plot,
    # tabulated in Table~\ref{tab:results}.
    # Color/style scheme (requested 2026-05-22):
    #   - AVIF stays black (FRAPPE-paper convention); default solid, max-speed dashed.
    #   - FRAPPE-codec stays dodgerblue, WaLLoC-codec is gold (FRAPPE-paper convention).
    #   - All four SEAOTTER variants share a pink/purple family:
    #       FRAPPE-side  -> blueviolet     (more purple-blue)
    #       WaLLoC-side  -> orchid   (more pink/magenta)
    #     Line style discriminates training regime:
    #       zero-shot (frozen codec + trained aligner) -> dotted (":")
    #       fine-tuned (codec updated w/ task loss)    -> solid  ("-")
    # AVIF default/max-speed: drop the second-lowest-bpp point on the plot
    # (the two lowest points sit ~on top of each other; the lowest already
    # marks the high-CR/low-acc corner clearly enough).
    def drop_second_lowest(pts):
        return [pts[0]] + pts[2:] if len(pts) >= 2 else pts

    PIPELINES = [
        {"name": "AVIF (default)",       "data": drop_second_lowest(load_pipeline("avif")),    "color": "black",      "linestyle": "-",  "marker": "."},
        {"name": "AVIF (max speed)",     "data": drop_second_lowest(load_pipeline("avifx")),   "color": "black",      "linestyle": "--", "marker": "."},
        {"name": "WaLLoC",               "data": load_pipeline("wal"),                          "color": "goldenrod",  "linestyle": "-",  "marker": "."},
        {"name": "FRAPPE",               "data": load_pipeline("frp"),                          "color": "dodgerblue", "linestyle": "-",  "marker": "."},
        {"name": "WaLLoC-SEAOTTER-ZS",   "data": load_pipeline("walsand"),                      "color": "orchid",     "linestyle": "--", "marker": "."},
        {"name": "FRAPPE-SEAOTTER-ZS",   "data": load_pipeline("seab"),                         "color": "blueviolet", "linestyle": "--", "marker": "."},
        {"name": "WaLLoC-SEAOTTER-FT",   "data": load_pipeline("walft"),                        "color": "orchid",     "linestyle": "-",  "marker": "."},
        {"name": "FRAPPE-SEAOTTER-FT",   "data": load_pipeline("seaft"),                        "color": "blueviolet", "linestyle": "-",  "marker": "."},
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    # Panel (a): downstream accuracy vs CR (task-accuracy R-D). Legend lives
    # here so it doesn't overlap the shaded deployment-tier regions on (b).
    draw_panel(axes[0], PIPELINES, "top1", "CR", legend=True)
    # Panel (b): throughput vs CR (deployment plane, with tier shading).
    draw_panel(axes[1], PIPELINES, "throughput_MPx", "CR", legend=False)

    # Shaded deployment-tier regions on panel (b) only. Each tier is the
    # operating region a sensor needs to clear to stream 30 fps video over
    # the named uplink:
    #   BLE      480p, 30fps: CR >= 288, throughput >= 12 MPx/s
    #   5G       720p, 30fps: CR >= 133, throughput >= 28 MPx/s
    #   Wi-Fi   1080p, 30fps: CR >=  60, throughput >= 62 MPx/s
    # (CR = 24 / bpp; throughput from sensor-side encode at 384 squash.)
    ax_thr = axes[1]
    # Lock the panel-a axis limits so the rectangles + labels lay out cleanly
    # regardless of data range.
    ax_thr.set_xlim(15, 3000)
    ax_thr.set_ylim(2, 700)
    xlim = ax_thr.get_xlim()
    ylim = ax_thr.get_ylim()
    REGIONS = [
        ("BLE, 480p",     288, 12, "tab:cyan"),
        ("5G, 720p",      133, 28, "tab:green"),
        ("Wi-Fi, 1080p",   60, 62, "tab:pink"),
    ]
    for label, cr_min, tp_min, color in REGIONS:
        ax_thr.fill_between(
            [cr_min, xlim[1]], tp_min, ylim[1],
            color=color, alpha=0.15, zorder=0, linewidth=0,
        )
        # Mark the lower-left corner of each region with a small dotted "L"
        # and a label nudged inward.
        ax_thr.plot([cr_min, cr_min], [tp_min, ylim[1]],
                    color=color, linestyle=":", linewidth=1.0, zorder=0.5)
        ax_thr.plot([cr_min, xlim[1]], [tp_min, tp_min],
                    color=color, linestyle=":", linewidth=1.0, zorder=0.5)
        ax_thr.text(cr_min * 1.08, tp_min * 1.08, label,
                    color=color, fontsize=10, ha="left", va="bottom",
                    zorder=4)

    # Shaded "impractically high bitrates" region on panel (a): CR < 200.
    ax_acc = axes[0]
    ax_acc.set_xlim(15, 3000)
    xlim_a = ax_acc.get_xlim()
    ylim_a = ax_acc.get_ylim()
    NV_COLOR = "tab:red"
    CR_BOUND = 200
    ax_acc.fill_between(
        [xlim_a[0], CR_BOUND], ylim_a[0], ylim_a[1],
        color=NV_COLOR, alpha=0.15, zorder=0, linewidth=0,
    )
    ax_acc.plot([CR_BOUND, CR_BOUND], [ylim_a[0], ylim_a[1]],
                color=NV_COLOR, linestyle=":", linewidth=1.0, zorder=0.5)
    # Horizontal two-line label, centered (geometric midpoint on log axis)
    # just above the 0.5 top-1 gridline.
    ax_acc.text((xlim_a[0] * CR_BOUND) ** 0.5, 0.52,
                "Bitrates not feasible for\nreal-time wireless streaming",
                color=NV_COLOR, fontsize=10,
                ha="center", va="bottom", rotation=0, zorder=4)
    ax_acc.set_ylim(ylim_a)

    # Subplot labels (a) / (b).
    axes[0].set_title("(a) Downstream accuracy")
    axes[1].set_title("(b) CPU Encoding throughput")

    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "main_results.pdf"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=160)
    print(f"WROTE {out_path}")

    # Also print a quick summary table for the closeout notes.
    print("\n--- pipeline summary (cls only) ---")
    for c in PIPELINES:
        if not c["data"]:
            print(f"{c['name']:>22}: (no data)")
            continue
        bpps = col(c["data"], "bpp")
        crs = col(c["data"], "CR")
        tps = col(c["data"], "throughput_MPx")
        t1s = [p["top1"] for p in c["data"] if p["top1"] is not None]
        print(
            f"{c['name']:>22}: {len(c['data'])} pts, "
            f"bpp [{min(bpps):.4f}, {max(bpps):.4f}], "
            f"CR [{min(crs):.2f}, {max(crs):.2f}], "
            f"thr [{min(tps):.2f}, {max(tps):.2f}] MPx/s, "
            f"top1 [{min(t1s)*100:.2f}%, {max(t1s)*100:.2f}%]"
        )


if __name__ == "__main__":
    main()
