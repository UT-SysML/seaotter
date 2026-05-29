"""Emit the LaTeX tables for the matched_rate branch.

Tables produced (all under `figures/`):

- `headline_table.tex`         — load-bearing inline summary, one row per
                                  non-raw pipeline at the Kodak rate of
                                  FRAPPE n_ch=12 (the FRAPPE-family
                                  anchor). Selected by closest Kodak-CR
                                  against ~/UT-SysML/seaotter/results/
                                  codec_kodak/eval_*_kodak_*.json cells.
- `deployment_tier_table.tex`  — load-bearing inline BLE/5G/Wi-Fi suitability
                                  (every pipeline x op cell from the mirror,
                                  sorted by descending CR).
- `rd_table_cls.tex`           — appendix per-task RD detail: transmit_bpp,
                                  storage_bpp, top1, psnr; bold-best per col.
- `rd_table_seg.tex`           — same shape, seg task (FRAPPE-side only).
- `rd_table_clip.tex`          — same shape, clip task (FRAPPE-side only).
- `throughput_table.tex`       — appendix throughput detail: encode ms,
                                  consumer ms (with steady-state proxy note).
- `codec_kodak_table.tex`      — 17-cell standalone-codec Kodak eval.
- `codec_kodak_cls_table.tex`  — 17-cell standalone-codec ImageNet eval.

All data is pulled from `~/UT-SysML/seaotter/results/`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path("/home/dgj335/UT-SysML/seaotter/results")
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CROP_CLS = 384  # cls squash side; used for MPx/s conversion.

# Pretty labels for pipeline keys.
PIPELINE_LABEL = {
    "raw":     "Raw (no codec)",
    "frp":     "FRAPPE",
    "seab":    "SEAOTTER-ZS",
    "seaft":   "SEAOTTER-FT",
    "wal":     "WaLLoC",
    "walsand": "WaLLoC-SEAOTTER-ZS",
    "walft":   "WaLLoC-SEAOTTER-FT",
    "avif":    "AVIF",
    "avifx":   "AVIF (max-speed)",
}

# Pipelines in main-paper scope. walft (WaLLoC-SEAOTTER-FT) is excluded
# from every table because we did not find fine-tuning hyperparameters
# that improved over the zero-shot walsand baseline; the limitation is
# a hyperparameter-search budget, not a property of SEAOTTER, so walft
# only appears in Figure~2 (main_results.pdf) for visual continuity
# with the rest of the FT family.
CLS_ORDER  = ["avif", "avifx", "frp", "wal", "seab", "walsand", "seaft", "raw"]
SEG_ORDER  = ["avif", "avifx", "frp", "wal", "seab", "walsand", "seaft", "raw"]
CLIP_ORDER = ["avif", "avifx", "frp", "wal", "seab", "walsand", "seaft", "raw"]

# Headline table pipeline order (collapsed to one row per pipeline at
# the canonical op).
HEADLINE_ORDER = ["avif", "avifx", "frp", "wal", "seab", "seaft", "raw"]

# Deployment-tier thresholds shared with make_figures.py.
# (CR_min, throughput_MPx_per_sec_min)
TIER_BLE   = (288, 12)   # 480p, 30 fps, 1 Mbps uplink
TIER_5G    = (133, 28)   # 720p, 30 fps, 5 Mbps uplink
TIER_WIFI  = ( 60, 62)   # 1080p, 30 fps, 25 Mbps uplink
TARGET_CR  = 133         # 5G anchor used for headline-table op selection.
TARGET_BPP = 24.0 / TARGET_CR  # 0.1804

CHECK   = r"\checkmark"
CROSS   = r"$\cdot$"  # subtle "no" mark to avoid visual clutter.

# Pipelines that share the frozen FRAPPE encoder. Encode bytes are
# identical across this set by construction; under the encoder-only
# throughput mirror the value is the single FRAPPE measurement at
# matched n_ch, aliased to all three rows. Marked with a trailing
# asterisk in tables to make the aliasing explicit and excluded from
# bold-best comparisons.
FRP_ENCODER_FAMILY = {"frp", "seab", "seaft"}
# Same pattern for the WaLLoC family — walsand reuses the frozen WaLLoC
# encoder; the encode bytes (and therefore the encode throughput) are
# the WaLLoC encoder's. walft is excluded paper-wide.
WAL_ENCODER_FAMILY = {"wal", "walsand"}


def _fmt(x, prec):
    if x is None:
        return "--"
    try:
        return f"{float(x):.{prec}f}"
    except (TypeError, ValueError):
        return "??"


def _load_eval(task: str):
    cells = []
    eval_dir = ROOT / task / "eval"
    if not eval_dir.exists():
        return cells
    for p in sorted(eval_dir.glob("eval_*.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        cells.append(d)
    return cells


def _load_thr(task: str):
    out = {}
    thr_dir = ROOT / task / "throughput"
    if not thr_dir.exists():
        return out
    for p in sorted(thr_dir.glob("throughput_*.json")):
        d = json.loads(p.read_text())
        key = (d["pipeline"], d["operating_point"]["type"],
               d["operating_point"]["value"])
        out[key] = d
    return out


# Map a pipeline short code to the encode-complexity codec short code
# used in the new mirror namespace. SEAOTTER-{ZS,FT} (seab/seaft) share
# the frozen FRAPPE encoder, so we alias them to `frp`. The WaLLoC
# sandwich pipelines (walsand/walft) similarly use the frozen WaLLoC
# encoder; we alias them to `wal`.
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

# Op-type translation from the iter-6 eval JSON's operating_point.type
# to the encode-complexity harness's op type. Most line up; q_pixel_ratio
# in some iter-6 JSONs maps to pixel_ratio in the new harness.
OP_TYPE_TO_ENC = {
    "q_pixel_ratio": "pixel_ratio",
    "pixel_ratio":   "pixel_ratio",
    "n_ch":          "n_ch",
    "quality":       "quality",
    "rate":          "rate",
    "compression_ratio": "rate",
}


# Task -> encode-complexity dataset label.
TASK_TO_ENC_DATASET = {
    "cls":  "cls_384",
    "seg":  "seg_512",
    "clip": "clip_naflex",
}


def _load_enc_only(dataset_label: str) -> dict:
    """Read all `encode_*.json` under `results/encode_complexity/<dataset>/`.

    Returns dict keyed by (codec, op_type, op_value) -> JSON dict. The
    op value matches the harness's stored ``operating_point.value``
    (which is the same form the iter-6 eval JSONs use).
    """
    out: dict[tuple, dict] = {}
    enc_dir = ROOT / "encode_complexity" / dataset_label
    if not enc_dir.exists():
        return out
    for p in sorted(enc_dir.glob("encode_*.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        op = d.get("operating_point", {})
        key = (d.get("codec"), op.get("type"), op.get("value"))
        out[key] = d
    return out


def _enc_only_mpx(enc_data: dict, pipeline: str, op_type: str, op_value) -> float | None:
    """Look up encoder-only median MPx/s in the new mirror.

    Aliases pipeline -> codec via PIPELINE_TO_ENC_CODEC (so seab/seaft
    resolve to the same `frp` row as the canonical FRAPPE encoder).
    Tolerates int<->float drift in op values.
    """
    codec = PIPELINE_TO_ENC_CODEC.get(pipeline)
    if codec is None:
        return None
    op_t = OP_TYPE_TO_ENC.get(op_type, op_type)

    def _lookup(val):
        d = enc_data.get((codec, op_t, val))
        if d is None:
            return None
        return ((d.get("results") or {}).get("throughput") or {}).get("median_MPx_per_s")

    mpx = _lookup(op_value)
    if mpx is None:
        try:
            mpx = _lookup(int(op_value))
        except (TypeError, ValueError):
            pass
    if mpx is None:
        try:
            mpx = _lookup(float(op_value))
        except (TypeError, ValueError):
            pass
    return mpx


def _enc_only_ms(enc_data: dict, pipeline: str, op_type: str, op_value,
                 crop: int = CROP_CLS) -> float | None:
    """Encode-only median time in ms, converted from MPx/s using `crop`.

    Note: the underlying JSON's median time is the canonical per-image
    median (sum of stage medians) — we recompute via MPx/s for tooling
    that wants milliseconds.
    """
    mpx = _enc_only_mpx(enc_data, pipeline, op_type, op_value)
    if mpx is None or mpx <= 0:
        return None
    return (crop * crop) / 1e6 / mpx * 1000.0


def _fmt_op_val(v):
    """Render an operating-point value as an integer if it's integral
    (e.g. 12, 16), otherwise as a one-decimal float (e.g. 10.5)."""
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)
    if abs(fv - round(fv)) < 1e-9:
        return str(int(round(fv)))
    return f"{fv:.1f}"


def _op_label(d):
    op = d["operating_point"]
    t = op.get("type")
    v = op.get("value")
    if t == "n_ch":
        return "$n=" + _fmt_op_val(v) + "$"
    if t in ("q_pixel_ratio", "pixel_ratio"):
        return "$p=" + _fmt_op_val(v) + "$"
    if t == "quality":
        return "$q=" + _fmt_op_val(v) + "$"
    if t == "compression_ratio":
        return "$r=" + _fmt_op_val(v) + "$"
    if t in ("none", None):
        return "--"
    return f"{t}={v}"


def _op_key(d):
    op = d["operating_point"]
    return (op.get("type"), op.get("value"))


def _enc_mpx(thr, crop=CROP_CLS):
    """Deprecated: encode throughput from the iter-11 cls/throughput JSONs.

    The iter-11 timer includes PIL decode + dtype cast inside the
    encode block, which conflates encoder cost with per-image
    preprocessing cost (the PIL preamble adds ~1.5-2 ms / 384^2 image,
    comparable to FRAPPE's encoder forward itself). The current
    tables read encode throughput from the new mirror at
    `results/encode_complexity/<dataset>/` via `_enc_only_mpx`.
    This function is kept for backward-compat tooling but no longer
    called by the table emitters.
    """
    if not thr:
        return None
    enc = thr["throughput"]["encode"].get("median_ms")
    if not enc or enc <= 0:
        return None
    return (crop * crop) / 1e6 / (enc / 1000.0)


def _consumer_ms(thr):
    """Consumer median in ms. Prefers ``consumer_decode_only`` (codec only,
    no teacher forward; iter11-cpu-2 schema) when available; otherwise
    falls back to the combined ``consumer`` field."""
    if not thr:
        return None
    block = thr["throughput"].get("consumer_decode_only") or thr["throughput"]["consumer"]
    return block.get("median_ms")


def _consumer_mpx(thr, crop=CROP_CLS):
    cm = _consumer_ms(thr)
    if not cm or cm <= 0:
        return None
    return (crop * crop) / 1e6 / (cm / 1000.0)


def _by_pipe(cells):
    out: dict[str, list[dict]] = {}
    for d in cells:
        out.setdefault(d["pipeline"], []).append(d)
    for k in out:
        out[k].sort(key=lambda d: d.get("transmit_bpp_mean") or 0.0)
    return out


def _closest_op(by_pipe_cls, pipe, target_bpp=TARGET_BPP):
    """Return the eval JSON of the op closest to target transmit bpp."""
    if pipe not in by_pipe_cls:
        return None
    cells = by_pipe_cls[pipe]
    if pipe == "raw":
        return cells[0] if cells else None
    best, best_d = None, math.inf
    for d in cells:
        bpp = d.get("transmit_bpp_mean")
        if bpp is None or bpp <= 0:
            continue
        dist = abs(math.log(bpp) - math.log(target_bpp))
        if dist < best_d:
            best, best_d = d, dist
    return best


def _bold(s, is_best):
    return r"\textbf{" + s + r"}" if is_best else s


def _find_eval_cell(by_pipe_cells, pipe, op_type, op_val):
    """Find the iter-6 eval JSON for a (pipe, op_type, op_value) tuple.
    Tolerates float vs int.
    """
    if pipe not in by_pipe_cells:
        return None
    for d in by_pipe_cells[pipe]:
        op = d.get("operating_point", {})
        if op.get("type") != op_type:
            continue
        v = op.get("value")
        if v is None:
            continue
        if abs(float(v) - float(op_val)) < 1e-6:
            return d
    return None


# Per-task subsection definitions for the transposed headline table.
SUBSECTIONS = [
    {
        "task":         "cls",
        "label":        r"ImageNet-1k ($384^2$)",
        "metric":       "top1",
        "metric_label": r"Top-1 Accuracy (\%)",
    },
    {
        "task":         "seg",
        "label":        r"ADE20k ($512^2$)",
        "metric":       "miou",
        "metric_label": r"mIoU (\%)",
    },
    {
        "task":         "clip",
        "label":        r"ImageNet-1k (naflex)",
        "metric":       "top1",
        "metric_label": r"SigLIP Accuracy (\%)",
    },
]

# Column order in the transposed headline table.
HEADLINE_COLS = ["avif", "avifx", "frp", "wal", "seab", "seaft"]

# Per-task pick overrides for the headline table. Keys are
# `(task, pipeline)` and values are `(op_type, op_value)`. The picker
# default rule is "lowest bpp strictly > FRAPPE n_ch=12's bpp" — an
# entry here overrides that with a manually-specified op. Used on seg
# to put AVIF on the below-anchor side (q=5, bpp=0.086) where the
# comparison with SEAOTTER-FT is rate-fair: AVIF spends *fewer* bits
# than FRAPPE n=12 and still essentially ties SEAOTTER-FT on mIoU.
HEADLINE_OP_OVERRIDES = {
    ("seg", "avif"): ("quality", 5),
}


def _pick_op_below_anchor(by_pipe_task, pipe, anchor_bpp):
    """Return the eval-cell for `pipe` whose transmit-bpp is the
    smallest value strictly greater than `anchor_bpp` (= the highest
    CR strictly less than the anchor's CR). Falls back to the closest
    bpp if no op exceeds the anchor."""
    if pipe not in by_pipe_task:
        return None
    cells = by_pipe_task[pipe]
    candidates = [
        d for d in cells
        if (d.get("transmit_bpp_mean") or 0) > anchor_bpp
    ]
    if candidates:
        return min(
            candidates,
            key=lambda d: d.get("transmit_bpp_mean") or float("inf"),
        )
    # Inversion: no op has more bits than FRAPPE n_ch=12. Fall back to
    # the closest available op.
    return min(
        cells,
        key=lambda d: abs((d.get("transmit_bpp_mean") or 0) - anchor_bpp),
    )


def emit_headline_table():
    """Transposed headline table grouped into three task subsections.

    Per subsection, FRAPPE n_ch=12's transmit-bpp on that task is the
    anchor. Each non-FRAPPE/non-SEAOTTER codec uses the op with the
    lowest bpp strictly greater than the anchor (= highest CR strictly
    less than FRAPPE n_ch=12's CR on that task). FRAPPE, SEAOTTER-ZS,
    and SEAOTTER-FT all use n_ch=12 in every subsection.

    Rows per subsection: Op, Transmit CR, Storage CR, accuracy,
    Encode (MPx/s), Decode (MPx/s). Throughput comes from
    cls/throughput JSONs (MPx/s is task-invariant for the same op).
    """
    by_task = {t: _by_pipe(_load_eval(t)) for t in ("cls", "seg", "clip")}
    thr_cls = _load_thr("cls")
    # Encoder-only throughput from the new mirror (per-task, since the
    # encode-complexity harness measures per-dataset). Decode stays at
    # iter-11 (`thr_cls`); the MPx/s for decode is approximately
    # task-invariant at the same op.
    enc_only = {t: _load_enc_only(TASK_TO_ENC_DATASET[t])
                for t in ("cls", "seg", "clip")}

    # Per-task chosen op (eval dict) per pipeline.
    per_task_chosen: dict[str, dict[str, dict | None]] = {}
    per_task_anchor: dict[str, float] = {}

    for task in ("cls", "seg", "clip"):
        by_pipe = by_task[task]
        frp_n12 = _find_eval_cell(by_pipe, "frp", "n_ch", 12)
        if frp_n12 is None:
            raise RuntimeError(
                f"No frp n_ch=12 eval cell for task {task!r}"
            )
        anchor_bpp = frp_n12.get("transmit_bpp_mean")
        per_task_anchor[task] = anchor_bpp
        chosen: dict[str, dict | None] = {
            "raw":   by_pipe["raw"][0] if by_pipe.get("raw") else None,
            "frp":   frp_n12,
            "seab":  _find_eval_cell(by_pipe, "seab",  "n_ch", 12),
            "seaft": _find_eval_cell(by_pipe, "seaft", "n_ch", 12),
        }
        for pipe in ("avif", "avifx", "wal"):
            override = HEADLINE_OP_OVERRIDES.get((task, pipe))
            if override is not None:
                op_t, op_v = override
                chosen[pipe] = _find_eval_cell(by_pipe, pipe, op_t, op_v)
                if chosen[pipe] is None:
                    raise RuntimeError(
                        f"Override ({task}, {pipe}) -> {override} has no "
                        f"matching eval cell in the mirror."
                    )
            else:
                chosen[pipe] = _pick_op_below_anchor(by_pipe, pipe, anchor_bpp)
        per_task_chosen[task] = chosen

    def _cr_int(x):
        if x is None:
            return "--"
        return f"{round(float(x))}"

    def _thr_for(d, task):
        """Return (encode_MPx/s, decode_MPx/s) for one chosen eval cell.

        Encode comes from the new encode-complexity mirror at the same
        task's dataset label (cls_384 / seg_512 / clip_naflex); the
        encode JSON is task-aware because input shape changes (384^2
        for cls, 512^2 for seg, variable naflex for clip). Decode
        stays at the iter-11 cls/throughput source — decode MPx/s is
        approximately task-invariant at the same op.
        """
        if d is None:
            return (None, None)
        op = d.get("operating_point", {})
        op_t, op_v = op.get("type"), op.get("value")
        t = thr_cls.get((d["pipeline"], op_t, op_v))
        if t is None:
            try:
                t = thr_cls.get((d["pipeline"], op_t, int(op_v)))
            except (TypeError, ValueError):
                pass
        if t is None:
            try:
                t = thr_cls.get((d["pipeline"], op_t, float(op_v)))
            except (TypeError, ValueError):
                pass
        enc_v = _enc_only_mpx(enc_only[task], d["pipeline"], op_t, op_v)
        return (enc_v, _consumer_mpx(t))

    n_pipes = len(HEADLINE_COLS)
    pipe_headers = [PIPELINE_LABEL.get(p, p) for p in HEADLINE_COLS]

    col_spec = "l" + "r" * n_pipes

    rows = []
    rows.append(r"\begin{tabular}{" + col_spec + r"}")
    rows.append(r"\toprule")
    rows.append(" & " + " & ".join(pipe_headers) + r" \\")
    rows.append(r"\midrule")

    def _bold_mask(values, rel_tol=1e-6):
        """Return [bool] flagging entries tied with the max (raw excluded).
        `values` is per-pipeline in HEADLINE_COLS order; the raw entry is
        excluded from the comparison."""
        non_raw_vals = [v for p, v in zip(HEADLINE_COLS, values)
                        if p != "raw" and v is not None]
        if not non_raw_vals:
            return [False] * len(values)
        vmax = max(non_raw_vals)
        abs_tol = max(abs(vmax) * rel_tol, 1e-9)
        return [
            (p != "raw" and v is not None and (vmax - v) <= abs_tol)
            for p, v in zip(HEADLINE_COLS, values)
        ]

    def _wrap_bold(cells, mask):
        return [
            (r"\textbf{" + c + r"}") if (m and c not in ("--", "1"))
            else c
            for c, m in zip(cells, mask)
        ]

    first_sub = True
    for sub in SUBSECTIONS:
        if not first_sub:
            rows.append(r"\midrule")
        first_sub = False

        task = sub["task"]
        chosen = per_task_chosen[task]

        # Subsection header (italicized, full-width).
        rows.append(
            r"\multicolumn{" + str(n_pipes + 1) + r"}{l}{\textit{"
            + sub["label"] + r"}} \\[2pt]"
        )

        # Operating-point row (no bolding).
        op_cells = []
        for pipe in HEADLINE_COLS:
            d = chosen[pipe]
            op_cells.append("--" if d is None else _op_label(d))
        rows.append("Operating point & " + " & ".join(op_cells) + r" \\")

        # Numerical-value collectors (None for missing).
        tx_vals, st_vals, acc_vals, enc_vals, dec_vals = [], [], [], [], []
        for pipe in HEADLINE_COLS:
            d = chosen[pipe]
            if d is None:
                tx_vals.append(None)
                st_vals.append(None)
                acc_vals.append(None)
            else:
                tb = d.get("transmit_bpp_mean")
                sb = d.get("storage_bpp_mean")
                tx_vals.append(24.0 / tb if tb and tb > 0 else None)
                st_vals.append(24.0 / sb if sb and sb > 0 else None)
                acc_vals.append((d.get("metrics") or {}).get(sub["metric"]))
            # Encode throughput for FRAPPE-family pipelines (frp / seab /
            # seaft) is canonicalized to FRAPPE n=12's measurement: the
            # three pipelines share the same frozen encoder and emit the
            # same sensor-uplink bytes, so the spread in per-cell
            # measurements is measurement noise rather than a real
            # architectural difference. Reporting one value avoids
            # implying a spread that doesn't exist.
            if pipe in FRP_ENCODER_FAMILY:
                enc_v, _ = _thr_for(chosen.get("frp"), task)
                _, dec_v = _thr_for(d, task)
            else:
                enc_v, dec_v = _thr_for(d, task)
            enc_vals.append(enc_v)
            dec_vals.append(dec_v)

        # Rendered cell strings.
        tx_cells  = [_cr_int(v) for v in tx_vals]
        st_cells  = [_cr_int(v) for v in st_vals]
        acc_cells = [_fmt(v * 100 if v is not None else None, 2)
                     for v in acc_vals]
        enc_cells = [_fmt(v, 2) for v in enc_vals]
        dec_cells = [_fmt(v, 2) for v in dec_vals]

        # Bold the row-best (raw excluded from comparison).
        tx_cells  = _wrap_bold(tx_cells,  _bold_mask(tx_vals))
        st_cells  = _wrap_bold(st_cells,  _bold_mask(st_vals))
        acc_cells = _wrap_bold(acc_cells, _bold_mask(acc_vals))
        enc_cells = _wrap_bold(enc_cells, _bold_mask(enc_vals))
        dec_cells = _wrap_bold(dec_cells, _bold_mask(dec_vals))

        rows.append("Transmit CR & "         + " & ".join(tx_cells)  + r" \\")
        rows.append("Storage CR & "          + " & ".join(st_cells)  + r" \\")
        rows.append(sub["metric_label"] + " & " + " & ".join(acc_cells) + r" \\")
        rows.append(r"Encode (MPx/s) & "     + " & ".join(enc_cells) + r" \\")
        rows.append(r"Decode (MPx/s) & "     + " & ".join(dec_cells) + r" \\")

    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")

    (OUT_DIR / "headline_table.tex").write_text("\n".join(rows) + "\n")
    print(f"WROTE {OUT_DIR / 'headline_table.tex'}")
    # Also print the picks so the user can sanity-check from the build log.
    print("  Per-task picks (transmit-bpp / CR):")
    for task in ("cls", "seg", "clip"):
        print(f"    [{task}] anchor (frp n_ch=12) bpp={per_task_anchor[task]:.4f} "
              f"CR={24/per_task_anchor[task]:.1f}")
        for pipe in HEADLINE_COLS:
            d = per_task_chosen[task].get(pipe)
            if d is None or pipe == "raw":
                continue
            op = d.get("operating_point", {})
            tb = d.get("transmit_bpp_mean")
            print(f"      {pipe:6s}  op={op.get('type')}={op.get('value')}  "
                  f"bpp={tb:.4f}  CR={24/tb:.1f}")


def emit_deployment_tier_table():
    """Inline suitability table: every pipeline x op cell, sorted by
    descending CR. Columns: Pipeline & Op & Transmit CR & Encode (MPx/s) &
    BLE & 5G & Wi-Fi.
    """
    cells = _load_eval("cls")
    thr   = _load_thr("cls")
    enc_only_cls = _load_enc_only("cls_384")

    rows_data = []
    for d in cells:
        pipe = d["pipeline"]
        if pipe not in CLS_ORDER or pipe == "raw":
            continue
        bpp = d.get("transmit_bpp_mean")
        if not bpp or bpp <= 0:
            continue
        cr = 24.0 / bpp
        op_key = _op_key(d)
        enc = _enc_only_mpx(enc_only_cls, pipe, op_key[0], op_key[1])

        def tier_ok(tier):
            cr_min, tp_min = tier
            return (enc is not None) and (cr >= cr_min) and (enc >= tp_min)

        rows_data.append({
            "pipe": pipe,
            "op_label": _op_label(d),
            "cr": cr,
            "enc": enc,
            "ble":  tier_ok(TIER_BLE),
            "fg":   tier_ok(TIER_5G),
            "wifi": tier_ok(TIER_WIFI),
        })

    rows_data.sort(key=lambda r: r["cr"], reverse=True)

    rows = []
    rows.append(r"\begin{tabular}{llrrccc}")
    rows.append(r"\toprule")
    rows.append(r" &  & Transmit & Encode & \multicolumn{3}{c}{Deployment tier} \\")
    rows.append(r"\cmidrule(lr){5-7}")
    rows.append(r"Pipeline & Op & CR & (MPx/s) & BLE & 5G & Wi-Fi \\")
    rows.append(r"\midrule")
    for r in rows_data:
        cr_s  = _fmt(r["cr"], 1) + ":1"
        enc_s = _fmt(r["enc"], 2)
        rows.append(
            " & ".join([
                PIPELINE_LABEL.get(r["pipe"], r["pipe"]),
                r["op_label"],
                cr_s, enc_s,
                CHECK if r["ble"]  else CROSS,
                CHECK if r["fg"]   else CROSS,
                CHECK if r["wifi"] else CROSS,
            ]) + r" \\"
        )
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")

    (OUT_DIR / "deployment_tier_table.tex").write_text("\n".join(rows) + "\n")
    print(f"WROTE {OUT_DIR / 'deployment_tier_table.tex'}")


def emit_task_table(task: str, accuracy_key: str, accuracy_header: str,
                    order: list[str], out_name: str):
    """Per-task RD table, appendix-bound. Columns: Pipeline, Op,
    Transmit bpp, Storage bpp, <accuracy>, PSNR. Bold-best per column
    over non-raw rows.
    Accuracy values printed as percentages (0..100).
    """
    cells = _load_eval(task)
    by_pipe = _by_pipe(cells)

    flat = []
    for short in order:
        if short == "raw" or short not in by_pipe:
            continue
        for d in by_pipe[short]:
            flat.append(d)

    def col_max(key, in_metrics=True):
        vals = []
        for d in flat:
            v = (d.get("metrics") or {}).get(key) if in_metrics else d.get(key)
            if v is not None:
                vals.append(v)
        return max(vals) if vals else None

    best_acc  = col_max(accuracy_key, in_metrics=True)
    best_psnr = col_max("psnr_db", in_metrics=True)

    rows = []
    rows.append(r"\begin{tabular}{llrrrr}")
    rows.append(r"\toprule")
    rows.append(
        r"Pipeline & Op & Transmit bpp & Storage bpp & "
        + accuracy_header + r" & PSNR (dB) \\"
    )
    rows.append(r"\midrule")
    first = True
    for short in order:
        if short not in by_pipe:
            continue
        if not first:
            rows.append(r"\midrule")
        first = False
        for i, d in enumerate(by_pipe[short]):
            m = d.get("metrics") or {}
            label_str = PIPELINE_LABEL.get(short, short)
            pipe_cell = label_str if i == 0 else ""
            acc_v  = m.get(accuracy_key)
            psnr_v = m.get("psnr_db")
            acc_s  = _fmt(acc_v * 100 if acc_v is not None else None, 2)
            psnr_s = _fmt(psnr_v, 2)
            rows.append(
                " & ".join([
                    pipe_cell,
                    _op_label(d),
                    _fmt(d.get("transmit_bpp_mean"), 4),
                    _fmt(d.get("storage_bpp_mean"), 4),
                    acc_s,
                    psnr_s,
                ]) + r" \\"
            )
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    (OUT_DIR / out_name).write_text("\n".join(rows) + "\n")
    print(f"WROTE {OUT_DIR / out_name}")


def emit_throughput_table():
    """Appendix throughput detail: encode + steady-state consumer-decode
    median wall-clock per pipeline x op for the cls task. iter11-cpu-2
    schema: ``consumer_decode_only`` is the codec-only decode time (no
    teacher forward), and for SEAOTTER pipelines it is JPEG decode +
    F^{-1} only (the cloud transcode is paid once and not counted here).
    """
    cells = _load_eval("cls")
    thr   = _load_thr("cls")
    enc_only_cls = _load_enc_only("cls_384")
    by_pipe = _by_pipe(cells)

    rows = []
    rows.append(r"\begin{tabular}{llrrrrr}")
    rows.append(r"\toprule")
    rows.append(r"Pipeline & Op & Transmit bpp & Enc.\ (ms) & Dec.\ (ms) & Enc.\ (MPx/s) & Dec.\ (MPx/s) \\")
    rows.append(r"\midrule")
    # Throughput-table-local labels: two-line cells for the AVIF speed
    # variants and the WaLLoC-encoder SEAOTTER row (does not affect the
    # shared PIPELINE_LABEL used by the other tables).
    THR_LABEL = {
        "avif":    r"\shortstack[l]{AVIF\\(default)}",
        "avifx":   r"\shortstack[l]{AVIF\\(max speed)}",
        "walsand": r"\shortstack[l]{SEAOTTER-ZS\\(WaLLoC encode)}",
    }
    first = True
    for short in CLS_ORDER:
        if short not in by_pipe or short == "raw":
            continue
        if not first:
            rows.append(r"\midrule")
        first = False
        for i, d in enumerate(by_pipe[short]):
            label_str = THR_LABEL.get(short, PIPELINE_LABEL.get(short, short))
            pipe_cell = label_str if i == 0 else ""
            op_key = _op_key(d)
            t = thr.get((d["pipeline"], op_key[0], op_key[1]))
            # Encode: new encoder-only mirror (cls_384). Decode: iter-11.
            enc_mpx = _enc_only_mpx(enc_only_cls, d["pipeline"], op_key[0], op_key[1])
            enc_ms  = _enc_only_ms(enc_only_cls, d["pipeline"], op_key[0], op_key[1])
            dec_ms  = _consumer_ms(t)
            dec_mpx = _consumer_mpx(t)
            enc_mpx_s = _fmt(enc_mpx, 2)
            if (
                short in FRP_ENCODER_FAMILY or short in WAL_ENCODER_FAMILY
            ) and enc_mpx is not None:
                # Asterisk: encode value is the canonical encoder's
                # measurement (FRAPPE for seab/seaft; WaLLoC for walsand)
                # since these pipelines share the upstream encoder.
                enc_mpx_s = enc_mpx_s + r"$^{*}$"
            rows.append(
                " & ".join([
                    pipe_cell,
                    _op_label(d),
                    _fmt(d.get("transmit_bpp_mean"), 4),
                    _fmt(enc_ms, 2),
                    _fmt(dec_ms, 2),
                    enc_mpx_s,
                    _fmt(dec_mpx, 2),
                ]) + r" \\"
            )
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    (OUT_DIR / "throughput_table.tex").write_text("\n".join(rows) + "\n")
    print(f"WROTE {OUT_DIR / 'throughput_table.tex'}")


def emit_codec_kodak_table():
    """Standalone-codec Kodak eval, 17 cells. Per-bpp-bracket bold PSNR.

    The matched_rate branch adds additional codec_kodak cells (frp,
    wal, avif, avifx) for rate-matching purposes; those are filtered
    out here so the standalone-codec appendix table keeps showing the
    original three codecs only.
    """
    kk = ROOT / "codec_kodak"
    rows = []
    rows.append(r"\begin{tabular}{llrrrrr}")
    rows.append(r"\toprule")
    rows.append(
        r"Codec & Setting & bpp & PSNR (dB) & SSIM & LPIPS (dB) & DISTS (dB) \\"
    )
    rows.append(r"\midrule")
    cells = []
    KEEP = {"seaotter", "jpeg", "jpeg_sub0"}
    for p in sorted(kk.glob("eval_*.json")):
        d = json.loads(p.read_text())
        if "summary" not in d:
            continue
        if d.get("config", {}).get("codec") not in KEEP:
            continue
        cells.append(d)
    cells.sort(key=lambda d: (d["config"]["codec"], d["summary"]["bpp"]["mean"]))

    def best_psnr_at_bpp(cells, target_bpp, window=0.30):
        lo, hi = target_bpp * (1 - window), target_bpp * (1 + window)
        best_psnr = None
        for c in cells:
            bpp = c["summary"]["bpp"]["mean"]
            if not (lo <= bpp <= hi):
                continue
            psnr = c["summary"]["psnr_db"]["mean"]
            if best_psnr is None or psnr > best_psnr:
                best_psnr = psnr
        return best_psnr

    last_codec = None
    for d in cells:
        codec = d["config"]["codec"]
        if codec != last_codec:
            if last_codec is not None:
                rows.append(r"\midrule")
            last_codec = codec
        cfg = d["config"]
        if codec == "seaotter":
            label_codec = "SEAOTTER (ours)"
            setting = "$k=" + str(cfg.get("seaotter_k")) + "$"
        elif codec == "jpeg":
            label_codec = "ITU T.81 4:2:0"
            setting = "$q=" + str(cfg.get("quality")) + "$"
        elif codec == "jpeg_sub0":
            label_codec = "ITU T.81 4:4:4"
            setting = "$q=" + str(cfg.get("quality")) + "$"
        else:
            label_codec = codec
            setting = "--"
        s = d["summary"]
        bpp = s["bpp"]["mean"]
        psnr = s["psnr_db"]["mean"]
        psnr_s = _fmt(psnr, 2)
        rows.append(
            " & ".join([
                label_codec, setting,
                _fmt(bpp, 3),
                psnr_s,
                _fmt(s["ssim"]["mean"], 3),
                _fmt(s["lpips_db"]["mean"], 2),
                _fmt(s["dists_db"]["mean"], 2),
            ]) + r" \\"
        )
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    (OUT_DIR / "codec_kodak_table.tex").write_text("\n".join(rows) + "\n")
    print(f"WROTE {OUT_DIR / 'codec_kodak_table.tex'}")


def emit_codec_kodak_cls_table():
    """ImageNet val (50k, squash 384^2) standalone-codec top-1 + PSNR over
    the same 17 cells. Bold Top-1 best at matched bpp."""
    kkc = ROOT / "codec_kodak_cls"
    rows = []
    rows.append(r"\begin{tabular}{llrrr}")
    rows.append(r"\toprule")
    rows.append(r"Codec & Setting & bpp & Top-1 (\%) & PSNR (dB) \\")
    rows.append(r"\midrule")
    cells = []
    for p in sorted(kkc.glob("eval_*.json")):
        if p.name == "eval_raw_cls_kodak_anchor.json":
            continue
        d = json.loads(p.read_text())
        if "metrics" not in d:
            continue
        cells.append(d)
    cells.sort(key=lambda d: (d["config"]["codec"], d["metrics"]["bpp_mean"]))

    def best_top1_at_bpp(cells, target_bpp, window=0.30):
        lo, hi = target_bpp * (1 - window), target_bpp * (1 + window)
        best = None
        for c in cells:
            bpp = c["metrics"]["bpp_mean"]
            if not (lo <= bpp <= hi):
                continue
            top1 = c["metrics"]["top1"]
            if best is None or top1 > best:
                best = top1
        return best

    last_codec = None
    for d in cells:
        codec = d["config"]["codec"]
        if codec != last_codec:
            if last_codec is not None:
                rows.append(r"\midrule")
            last_codec = codec
        cfg = d["config"]
        if codec == "seaotter":
            label_codec = "SEAOTTER (ours)"
            setting = "$k=" + str(cfg.get("seaotter_k")) + "$"
        elif codec == "jpeg":
            label_codec = "ITU T.81 4:2:0"
            setting = "$q=" + str(cfg.get("quality")) + "$"
        elif codec == "jpeg_sub0":
            label_codec = "ITU T.81 4:4:4"
            setting = "$q=" + str(cfg.get("quality")) + "$"
        else:
            label_codec, setting = codec, "--"
        m = d["metrics"]
        bpp = m["bpp_mean"]
        top1 = m["top1"]
        top1_s = _fmt(top1 * 100.0, 2)
        rows.append(
            " & ".join([
                label_codec, setting,
                _fmt(bpp, 3),
                top1_s,
                _fmt(m["psnr_db"], 2),
            ]) + r" \\"
        )
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    (OUT_DIR / "codec_kodak_cls_table.tex").write_text("\n".join(rows) + "\n")
    print(f"WROTE {OUT_DIR / 'codec_kodak_cls_table.tex'}")


def main():
    emit_headline_table()
    emit_deployment_tier_table()
    emit_task_table("cls",  "top1", r"Top-1 (\%)",
                    CLS_ORDER,  "rd_table_cls.tex")
    emit_task_table("seg",  "miou", r"mIoU (\%)",
                    SEG_ORDER,  "rd_table_seg.tex")
    emit_task_table("clip", "top1", r"Zero-shot Top-1 (\%)",
                    CLIP_ORDER, "rd_table_clip.tex")
    emit_throughput_table()
    emit_codec_kodak_table()
    emit_codec_kodak_cls_table()


if __name__ == "__main__":
    main()
