"""Launch every load-bearing (codec, op, dataset) cell for the CoRL paper.

Runs serially on CPU (no parallel cells — cache contention skews
throughput on the EPYC 9354). Outputs land at
``~/UT-SysML/seaotter/results/encode_complexity/<dataset>/encode_<codec>_<op>.json``.

Codecs in scope (from the prompt):
  - avif, avifx, jpeg, jpeg_sub0, webp, jp2  (Pillow family)
  - frp                                       (FRAPPE)
  - wal                                       (WaLLoC)
  - seaotter_jpeg                             (standalone learned JPEG)

For the SEAOTTER family (`seab`, `seaft`), encoder bytes are identical
to `frp` at matched `n_ch`, so we measure `frp` once and the table
code aliases the result. We do not re-time the same encoder under
three pipeline labels.

Usage::

    python -m seaotter.eval.launch_load_bearing

Optional flags::

    --datasets cls_384,seg_512,clip_naflex,kodak_native
    --codecs   frp,wal,avif,...
    --dry_run
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

OUT_ROOT = Path("/home/dgj335/UT-SysML/seaotter/results/encode_complexity")


# ----- load-bearing matrix -----

# Codec -> list of (op_dict, op_slug). Slug must match what the codec
# adapter's `op_slug()` returns; we don't actually use it here, just
# the op_dict.

CODEC_OPS = {
    "avif":     [{"type": "quality", "value": q} for q in (1, 5, 10, 25, 50)],
    "avifx":    [{"type": "quality", "value": q, "extras": {"avif_speed": 10}}
                 for q in (1, 3, 5, 6, 10, 25, 50)],
    "jpeg":     [{"type": "quality", "value": q} for q in (1, 5, 10, 25, 50)],
    "jpeg_sub0":[{"type": "quality", "value": q} for q in (1, 5, 10, 25, 50)],
    "webp":     [{"type": "quality", "value": q} for q in (1, 5, 10, 25, 50)],
    "jp2":      [{"type": "rate",    "value": r} for r in (12, 25, 50, 100, 200)],
    "frp":      [{"type": "n_ch",    "value": n} for n in (3, 6, 9, 12, 15)],
    # WaLLoC: 4, 8, 11, 16, 36, 80, 100 cover the matched-rate picks
    # in the paper plus the rate-curve endpoints (p4 / p100).
    "wal":      [{"type": "pixel_ratio", "value": p} for p in (4, 8, 11, 16, 36, 80, 100)],
    "seaotter_jpeg": [{"type": "k", "value": k} for k in (0, 1, 2)],
}

ALL_DATASETS = ("cls_384", "seg_512", "clip_naflex", "kodak_native")


def _slug_for(codec: str, op: dict) -> str:
    """Mirror `CodecAdapter.op_slug` for filename construction."""
    t = op["type"]; v = op["value"]
    if codec == "frp":
        return f"n{int(v)}"
    if codec == "wal":
        if abs(float(v) - round(float(v))) < 1e-9:
            return f"p{int(round(float(v)))}"
        return f"p{str(v).replace('.', 'p')}"
    if codec == "seaotter_jpeg":
        return f"k{int(v)}"
    if codec == "jp2":
        return f"r{int(v) if abs(float(v) - int(float(v))) < 1e-9 else v}"
    if codec == "avifx":
        speed = int(op.get("extras", {}).get("avif_speed", 10))
        return f"q{int(v)}_s{speed}"
    # avif / jpeg / jpeg_sub0 / webp
    return f"q{int(v)}"


def cells_to_run(
    codecs: list[str],
    datasets: list[str],
    skip_existing: bool = True,
) -> list[tuple[str, dict, str, Path]]:
    """Enumerate all (codec, op, dataset, out_path) tuples."""
    out = []
    for ds in datasets:
        for codec in codecs:
            if codec not in CODEC_OPS:
                raise ValueError(f"unknown codec {codec!r}")
            for op in CODEC_OPS[codec]:
                slug = _slug_for(codec, op)
                p = OUT_ROOT / ds / f"encode_{codec}_{slug}.json"
                out.append((codec, op, ds, p))
    return out


def run_all(
    *,
    codecs: list[str],
    datasets: list[str],
    n_warmup: int = 1,
    n_measurement: int = 5,
    n_images: int = 256,
    dry_run: bool = False,
    skip_existing: bool = True,
    verbose: bool = True,
):
    from seaotter.eval.encode_complexity import run_codec_x_dataset_with_inputs
    from seaotter.eval.datasets import load_inputs

    cells = cells_to_run(codecs, datasets, skip_existing=skip_existing)
    print(f"Total cells: {len(cells)}", flush=True)
    if dry_run:
        for codec, op, ds, p in cells:
            print(f"  [DRY] {codec} {op} -> {p}", flush=True)
        return

    # Pre-load each dataset once and reuse across cells. The encoder's
    # native-input pre-staging happens inside the adapter for each cell
    # since different codecs need different native forms.
    needed_datasets = sorted({ds for _, _, ds, _ in cells})
    loaded_imgs: dict[str, list] = {}
    for ds in needed_datasets:
        ds_n = 24 if ds == "kodak_native" else n_images
        if verbose:
            print(f"\n[load] {ds} ({ds_n} images) ...", flush=True)
        loaded_imgs[ds] = load_inputs(ds, ds_n)
        if verbose:
            print(f"  loaded {len(loaded_imgs[ds])} images", flush=True)

    t0 = time.time()
    n_done = 0
    n_skipped = 0
    n_failed = 0
    failures = []
    for codec, op, ds, p in cells:
        if skip_existing and p.exists():
            n_skipped += 1
            if verbose:
                print(f"[SKIP existing] {p}", flush=True)
            continue
        try:
            run_codec_x_dataset_with_inputs(
                codec=codec,
                op=op,
                dataset=ds,
                imgs=loaded_imgs[ds],
                n_warmup=n_warmup,
                n_measurement=n_measurement,
                device="cpu",
                out_json=str(p),
                verbose=verbose,
            )
            n_done += 1
        except Exception as e:
            n_failed += 1
            failures.append((codec, op, ds, str(e)))
            print(f"[FAIL] {codec} {op} @ {ds}: {e}", flush=True)
    print(
        f"\n[summary] done={n_done} skipped={n_skipped} failed={n_failed} "
        f"elapsed={time.time()-t0:.1f}s",
        flush=True,
    )
    if failures:
        print("\nFailures:")
        for codec, op, ds, msg in failures:
            print(f"  {codec} {op} @ {ds}: {msg}")


def _build_argparser():
    p = argparse.ArgumentParser(
        prog="python -m seaotter.eval.launch_load_bearing",
        description="Launch every load-bearing encode-complexity cell.",
    )
    p.add_argument(
        "--datasets",
        default=",".join(ALL_DATASETS),
        help="Comma-separated dataset labels. Default: every dataset.",
    )
    p.add_argument(
        "--codecs",
        default=",".join(CODEC_OPS),
        help="Comma-separated codec shorts. Default: every codec.",
    )
    p.add_argument("--n_warmup", type=int, default=1)
    p.add_argument("--n_measurement", type=int, default=5)
    p.add_argument("--n_images", type=int, default=256)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument(
        "--rerun_existing", action="store_true",
        help="If set, overwrite existing JSONs instead of skipping.",
    )
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None):
    args = _build_argparser().parse_args(argv)
    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    codecs = [s.strip() for s in args.codecs.split(",") if s.strip()]
    run_all(
        codecs=codecs,
        datasets=datasets,
        n_warmup=args.n_warmup,
        n_measurement=args.n_measurement,
        n_images=args.n_images,
        dry_run=args.dry_run,
        skip_existing=not args.rerun_existing,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
