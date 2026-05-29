"""Validate the 10 production cells against per-cell acceptance criteria.

Run after all 10 JSONs land. Prints a per-cell pass/fail table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROD = Path("/home/dgj335/danjacobellis/seaotter/experiments/wal_seg_clip/production")

EXPECTED = {
    ("wal", task, f"p{p}"): {"op_type": "pixel_ratio", "op_value": p, "q_pixel_ratio": float(p)}
    for task in ("seg", "clip") for p in (4, 16, 36, 80, 100)
}


def primary_metric(task: str, metrics: dict) -> tuple[str, float | None]:
    if task == "seg": return ("miou", metrics.get("miou"))
    if task == "clip": return ("top1", metrics.get("top1"))
    return ("?", None)


def check(name: str, d: dict, expected: dict) -> list[str]:
    errs: list[str] = []
    if d["pipeline"] != "wal":
        errs.append(f"unexpected pipeline {d['pipeline']!r}")
    if d.get("pipeline_label") != "WaLLoC RGB_16x":
        errs.append(f"pipeline_label {d.get('pipeline_label')!r} != 'WaLLoC RGB_16x'")
    op = d["operating_point"]
    if op["type"] != expected["op_type"]:
        errs.append(f"op.type {op['type']!r} != {expected['op_type']!r}")
    if op["value"] != expected["op_value"]:
        errs.append(f"op.value {op['value']} != {expected['op_value']}")
    cfg = d["config"]
    if cfg.get("codec") != "walloc:RGB_16x":
        errs.append(f"config.codec {cfg.get('codec')!r} != 'walloc:RGB_16x'")
    if cfg.get("q_pixel_ratio") != expected["q_pixel_ratio"]:
        errs.append(f"config.q_pixel_ratio {cfg.get('q_pixel_ratio')} != {expected['q_pixel_ratio']}")
    tx, st = d["transmit_bpp_mean"], d["storage_bpp_mean"]
    if tx is None or st is None:
        errs.append(f"bpp None (tx={tx}, st={st})")
    elif abs(tx - st) > 1e-9:
        errs.append(f"transmit_bpp {tx:.6f} != storage_bpp {st:.6f} (expected equal for vanilla wal)")
    metric_key, m = primary_metric(d["task"], d["metrics"])
    if m is None:
        errs.append(f"primary metric {metric_key} is None")
    elif not (0.0 <= m <= 1.0):
        errs.append(f"{metric_key}={m} out of [0,1]")
    return errs


def main() -> int:
    rows: list[tuple[str, str, str, str, list[str], dict]] = []
    missing: list[str] = []
    for key, expected in EXPECTED.items():
        pipeline, task, op_short = key
        path = PROD / f"eval_{pipeline}_{task}_{op_short}.json"
        if not path.exists():
            missing.append(path.name)
            continue
        d = json.loads(path.read_text())
        errs = check(path.name, d, expected)
        rows.append((pipeline, task, op_short, path.name, errs, d))

    print(f"Found {len(rows)}/{len(EXPECTED)} cells. Missing: {len(missing)}.")
    if missing:
        for m in missing:
            print(f"  MISSING {m}")

    print(f"\n{'pipeline':<6} {'task':<5} {'op':<6} {'tx_bpp':>8} {'st_bpp':>8} {'metric':>9} {'psnr':>6} {'status'}")
    n_ok = n_fail = 0
    for pipeline, task, op_short, name, errs, d in rows:
        m_key, m = primary_metric(task, d["metrics"])
        m_str = f"{m:.4f}" if m is not None else "  None"
        psnr = d["metrics"].get("psnr_db") or 0.0
        status = "OK" if not errs else f"FAIL: {'; '.join(errs)}"
        n_ok += int(not errs)
        n_fail += int(bool(errs))
        print(f"{pipeline:<6} {task:<5} {op_short:<6} "
              f"{d['transmit_bpp_mean']:8.4f} {d['storage_bpp_mean']:8.4f} "
              f"{m_str:>9} {psnr:6.2f}  {status}")
    print(f"\nSummary: {n_ok} OK, {n_fail} FAIL, {len(missing)} MISSING (of {len(EXPECTED)} expected)")
    return 0 if (n_fail == 0 and not missing) else 1


if __name__ == "__main__":
    sys.exit(main())
