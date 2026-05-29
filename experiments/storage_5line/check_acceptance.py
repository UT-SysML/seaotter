"""Validate the 40 production cells against the per-cell acceptance criteria.

Run after all 40 JSONs land. Prints a per-cell pass/fail table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROD = Path("/home/dgj335/danjacobellis/seaotter/experiments/storage_5line/production")

EXPECTED = {
    # 15 frp_jpeg cells
    **{("frp_jpeg", task, f"n{n}"): {"op_type": "n_ch", "op_value": n, "frappe_n_ch": n}
       for task in ("cls", "seg", "clip") for n in (3, 6, 9, 12, 15)},
    # 15 wal_jpeg cells
    **{("wal_jpeg", task, f"p{p}"): {"op_type": "pixel_ratio", "op_value": p, "walloc_pixel_ratio": float(p)}
       for task in ("cls", "seg", "clip") for p in (4, 16, 36, 80, 100)},
    # 10 walsand cells
    **{("walsand", task, f"p{p}"): {"op_type": "pixel_ratio", "op_value": p, "q_pixel_ratio": float(p)}
       for task in ("seg", "clip") for p in (4, 16, 36, 80, 100)},
}


def primary_metric(task: str, metrics: dict) -> tuple[str, float | None]:
    if task == "cls": return ("top1", metrics.get("top1"))
    if task == "seg": return ("miou", metrics.get("miou"))
    if task == "clip": return ("top1", metrics.get("top1"))
    return ("?", None)


def check(name: str, d: dict, expected: dict) -> list[str]:
    errs: list[str] = []
    if d["pipeline"] not in ("frp_jpeg", "wal_jpeg", "walsand"):
        errs.append(f"unexpected pipeline {d['pipeline']!r}")
    op = d["operating_point"]
    if op["type"] != expected["op_type"]:
        errs.append(f"op.type {op['type']!r} != {expected['op_type']!r}")
    if op["value"] != expected["op_value"]:
        errs.append(f"op.value {op['value']} != {expected['op_value']}")
    if d["pipeline"] in ("frp_jpeg", "wal_jpeg"):
        cfg = d["config"]
        if cfg.get("format") != "JPEG":
            errs.append(f"config.format != 'JPEG' (got {cfg.get('format')})")
        if cfg.get("subsampling") != 0:
            errs.append(f"config.subsampling != 0 (got {cfg.get('subsampling')})")
        if cfg.get("quality") != 75:
            errs.append(f"config.quality != 75 (got {cfg.get('quality')})")
        check_key = "frappe_n_ch" if d["pipeline"] == "frp_jpeg" else "walloc_pixel_ratio"
        if cfg.get(check_key) != expected[check_key]:
            errs.append(f"config.{check_key} != {expected[check_key]} (got {cfg.get(check_key)})")
        if d["transmit_bpp_mean"] >= d["storage_bpp_mean"]:
            errs.append(f"transmit_bpp {d['transmit_bpp_mean']:.4f} not < storage_bpp {d['storage_bpp_mean']:.4f}")
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

    print(f"\n{'pipeline':<10} {'task':<5} {'op':<6} {'tx_bpp':>8} {'st_bpp':>8} {'metric':>9} {'psnr':>6} {'status'}")
    n_ok = n_fail = 0
    for pipeline, task, op_short, name, errs, d in rows:
        m_key, m = primary_metric(task, d["metrics"])
        m_str = f"{m:.4f}" if m is not None else "  None"
        psnr = d["metrics"].get("psnr_db") or 0.0
        status = "OK" if not errs else f"FAIL: {'; '.join(errs)}"
        n_ok += int(not errs)
        n_fail += int(bool(errs))
        print(f"{pipeline:<10} {task:<5} {op_short:<6} "
              f"{d['transmit_bpp_mean']:8.4f} {d['storage_bpp_mean']:8.4f} "
              f"{m_str:>9} {psnr:6.2f}  {status}")
    print(f"\nSummary: {n_ok} OK, {n_fail} FAIL, {len(missing)} MISSING (of {len(EXPECTED)} expected)")
    return 0 if (n_fail == 0 and not missing) else 1


if __name__ == "__main__":
    sys.exit(main())
