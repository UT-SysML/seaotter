"""Compute data-driven radar rankings from Table 1 (matched-rate, v3).

For each radar axis, pull one principled scalar per system from
Table 1's per-task matched-rate picks (FRAPPE n=12 anchor on
cls/seg/clip; for ITU, the lowest-q op closest to the anchor
neighborhood) plus the matching eval-JSON distortion / throughput
values. Linear-normalize the five-system range onto [1, 5];
throughput axes use log normalization because their raw range
spans ~2 orders of magnitude.

Encoder-only throughput (`enc` axis) reads from the new
`results/encode_complexity/<dataset>/encode_<codec>_<op>.json`
mirror so the timer excludes PIL decode / dtype-cast preamble. The
decode axis (`dec`) still reads from the iter-11 CPU throughput
JSONs (codec-only steady-state decode time).

Produces a fractional RANKINGS dict suitable for
`make_radar_figure_data.py`. The discrete (rounded) variant lives
in `make_radar_figure.py` via `notes/radar_rankings_data_driven.md`.

Output: pretty-printed dict to stdout, plus a markdown table of raw
scalars and scores for cross-referencing the notes report.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

RESULTS = Path("/home/dgj335/UT-SysML/seaotter/results")
ENC_COMPLEXITY = RESULTS / "encode_complexity"

# Pipeline -> encode-complexity codec alias. SEAOTTER-{ZS,FT} share the
# frozen FRAPPE encoder, so we alias both to `frp`.
PIPE_TO_ENC_CODEC = {
    "avifx":  "avifx",
    "avif":   "avif",
    "jpeg":   "jpeg",
    "frp":    "frp",
    "seab":   "frp",
    "seaft":  "frp",
    "wal":    "wal",
}

# Pipeline-short codes used in the eval JSONs. "AVIF" in the radar
# maps to the max-speed variant (`avifx`, libavif speed=10); this is
# the deployment-realistic configuration for any real-time sensor-
# side encoder and yields a less pessimistic encode-throughput score
# than the default-speed variant (5.16 vs 23.87 MPx/s).
SHORT = {
    "AVIF":        "avifx",
    "ITU":         "jpeg",
    "FRAPPE":      "frp",
    "SEAOTTER-ZS": "seab",
    "SEAOTTER-FT": "seaft",
}

SYSTEMS = list(SHORT.keys())

# Per-task matched-rate operating point per system (op_type, op_value).
# AVIF (max-speed) / WaLLoC: matched-rate picks per the headline_table
# builder in make_tables.py (lowest bpp strictly > FRAPPE n=12). ITU:
# lowest-quality available op (q=1), which is the closest ITU can get
# to the FRAPPE-anchor neighborhood.
PICKS = {
    "cls":  {
        "AVIF":        ("quality", 1),
        "ITU":         ("quality", 1),
        "FRAPPE":      ("n_ch", 12),
        "SEAOTTER-ZS": ("n_ch", 12),
        "SEAOTTER-FT": ("n_ch", 12),
    },
    "seg":  {
        "AVIF":        ("quality", 6),
        "ITU":         ("quality", 1),
        "FRAPPE":      ("n_ch", 12),
        "SEAOTTER-ZS": ("n_ch", 12),
        "SEAOTTER-FT": ("n_ch", 12),
    },
    "clip": {
        "AVIF":        ("quality", 1),
        "ITU":         ("quality", 1),
        "FRAPPE":      ("n_ch", 12),
        "SEAOTTER-ZS": ("n_ch", 12),
        "SEAOTTER-FT": ("n_ch", 12),
    },
}

METRIC_KEY = {"cls": "top1", "seg": "miou", "clip": "top1"}
CROP_CLS = 384  # for converting cls throughput ms to MPx/s


def _find_eval(task: str, pipe_short: str, op_type, op_value):
    for p in sorted((RESULTS / task / "eval").glob(f"eval_{pipe_short}_{task}_*.json")):
        d = json.loads(p.read_text())
        op = d["operating_point"]
        if op.get("type") != op_type:
            continue
        v = op.get("value")
        if v is None:
            continue
        if abs(float(v) - float(op_value)) < 1e-6:
            return d
    raise FileNotFoundError(f"No eval for {task} {pipe_short} {op_type}={op_value}")


def _find_thr(pipe_short: str, op_type, op_value):
    # iter-11 CPU throughput JSONs are in cls/throughput; throughput at
    # the same op is approximately task-invariant in MPx/s.
    for p in sorted((RESULTS / "cls" / "throughput").glob(f"throughput_{pipe_short}_cls_*.json")):
        d = json.loads(p.read_text())
        op = d["operating_point"]
        if op.get("type") != op_type:
            continue
        v = op.get("value")
        if v is None:
            continue
        if abs(float(v) - float(op_value)) < 1e-6:
            return d
    return None


def _normalize_linear(values: dict[str, float]) -> dict[str, float]:
    """Linear normalise to [1, 5]. Assumes higher raw == better."""
    vmin, vmax = min(values.values()), max(values.values())
    if vmax - vmin < 1e-9:
        return {k: 5.0 for k in values}
    return {k: 1.0 + (v - vmin) / (vmax - vmin) * 4.0 for k, v in values.items()}


def _normalize_log(values: dict[str, float]) -> dict[str, float]:
    """Log normalise to [1, 5]. Assumes higher raw == better."""
    logs = {k: math.log10(v) for k, v in values.items()}
    return _normalize_linear(logs)


def gather_raw():
    """Return dict {axis: {system: raw_scalar}} from Table 1 picks."""
    out: dict[str, dict[str, float]] = {axis: {} for axis in (
        "trans", "store", "PSNR", "DISTS", "cls", "seg", "clip", "enc", "dec",
    )}

    # Per-system, per-task pulls (CRs, downstream accuracy, PSNR, DISTS_dB).
    per_task_tx_cr, per_task_st_cr = {}, {}
    per_task_psnr, per_task_dists = {}, {}

    for sys in SYSTEMS:
        per_task_tx_cr[sys] = {}
        per_task_st_cr[sys] = {}
        per_task_psnr[sys] = {}
        per_task_dists[sys] = {}
        for task in ("cls", "seg", "clip"):
            op_t, op_v = PICKS[task][sys]
            d = _find_eval(task, SHORT[sys], op_t, op_v)
            tb = d["transmit_bpp_mean"]
            sb = d["storage_bpp_mean"]
            m = d.get("metrics") or {}
            per_task_tx_cr[sys][task] = 24.0 / tb if tb else None
            per_task_st_cr[sys][task] = 24.0 / sb if sb else None
            per_task_psnr[sys][task] = m.get("psnr_db")
            per_task_dists[sys][task] = m.get("dists_db")
            if task == METRIC_KEY[task]:  # tautology guard
                pass
            metric = m.get(METRIC_KEY[task])
            out[task][sys] = metric * 100.0 if metric is not None else None

    # Means across tasks for the aggregate axes.
    for sys in SYSTEMS:
        out["trans"][sys] = sum(per_task_tx_cr[sys].values()) / 3.0
        out["store"][sys] = sum(per_task_st_cr[sys].values()) / 3.0
        out["PSNR"][sys]  = sum(per_task_psnr[sys].values()) / 3.0
        out["DISTS"][sys] = sum(per_task_dists[sys].values()) / 3.0

    # Decode throughput: still from iter-11 cls/throughput JSONs
    # (consumer_decode_only / consumer median_ms). Decode rate-context
    # is the matched-rate pick (FRAPPE n_ch=12 anchor), keeping the
    # decode workload fixed at the rate the paper headlines.
    for sys in SYSTEMS:
        op_t, op_v = PICKS["cls"][sys]
        t_match = _find_thr(SHORT[sys], op_t, op_v)
        if t_match is None:
            raise RuntimeError(
                f"Missing matched-rate throughput for {sys} cls {op_t}={op_v}"
            )
        dec_ms = (t_match["throughput"].get("consumer_decode_only")
                  or t_match["throughput"]["consumer"])["median_ms"]
        out["dec"][sys] = (CROP_CLS * CROP_CLS) / 1e6 / (dec_ms / 1000.0)

    # Encoder-only throughput from the new mirror (cls_384 dataset).
    # `enc` is each codec's best-case (highest-compression) operating
    # point — FRAPPE's encoder reaches its fastest at n_ch=3, not n_ch=12,
    # so pinning to the matched-rate op would understate the peak.
    enc_dir = ENC_COMPLEXITY / "cls_384"
    if not enc_dir.exists():
        raise FileNotFoundError(
            f"Encoder-only mirror missing: {enc_dir}. Run "
            f"`python -m seaotter.eval.launch_load_bearing` first."
        )
    for sys in SYSTEMS:
        enc_codec = PIPE_TO_ENC_CODEC[SHORT[sys]]
        best_enc = 0.0
        for p in sorted(enc_dir.glob(f"encode_{enc_codec}_*.json")):
            d = json.loads(p.read_text())
            tp = (d.get("results") or {}).get("throughput") or {}
            mpx = tp.get("median_MPx_per_s") or 0.0
            if mpx > best_enc:
                best_enc = mpx
        if best_enc <= 0:
            raise RuntimeError(
                f"No encode-only data for {sys} ({enc_codec}) under {enc_dir}"
            )
        out["enc"][sys] = best_enc

    # SEAOTTER-{ZS,FT} share the FRAPPE encoder bytes; canonicalize
    # their enc value to FRAPPE's, matching what Table 1 reports.
    out["enc"]["SEAOTTER-ZS"] = out["enc"]["FRAPPE"]
    out["enc"]["SEAOTTER-FT"] = out["enc"]["FRAPPE"]

    return out


AXES = ["trans", "store", "PSNR", "DISTS", "cls", "seg", "clip", "enc", "dec"]
LOG_AXES = {"enc", "dec"}


def compute_rankings() -> tuple[dict, dict]:
    """Return (raw_per_axis, scored_per_system).

    ``raw_per_axis[axis][system]`` is the raw scalar; ``scored_per_system[system]``
    is the [1, 5]-normalised radar vector ordered by ``AXES``.
    """
    raw = gather_raw()
    scored: dict[str, dict[str, float]] = {}
    for axis, vals in raw.items():
        if axis in LOG_AXES:
            scored[axis] = _normalize_log(vals)
        else:
            scored[axis] = _normalize_linear(vals)
    out: dict[str, list[float]] = {}
    for sys in SYSTEMS:
        out[sys] = [scored[a][sys] for a in AXES]
    return raw, out


def main():
    raw, ranked = compute_rankings()

    scored: dict[str, dict[str, float]] = {}
    for axis, vals in raw.items():
        if axis in LOG_AXES:
            scored[axis] = _normalize_log(vals)
        else:
            scored[axis] = _normalize_linear(vals)

    # Pretty-print raw + score per axis.
    for axis in AXES:
        print(f"\n## {axis}")
        print(f"{'system':<14s}  {'raw':>10s}  {'score':>6s}")
        for sys in SYSTEMS:
            r = raw[axis][sys]
            s = scored[axis][sys]
            print(f"{sys:<14s}  {r:>10.3f}  {s:>6.3f}")

    # Emit RANKINGS dict ready to paste into make_radar_figure_data.py.
    print("\n## RANKINGS dict")
    print("RANKINGS = {")
    for sys in SYSTEMS:
        row = ", ".join(f"{ranked[sys][i]:5.2f}" for i in range(len(AXES)))
        print(f"    {sys!r:<15s}: [{row}],")
    print("}")


if __name__ == "__main__":
    main()
