"""Core timing loop + CLI for the encoder-only complexity harness.

Reproduces the FRAPPE `evaluate_encode_complexity` methodology for every
codec in the CoRL 2026 SEAOTTER paper:

  1. Pre-stage every input in the encoder's native form (no decode /
     conversion inside the timer).
  2. Run `n_warmup` untimed passes over the entire input list.
  3. Run `n_measurement` timed passes, recording per-stage timings.
  4. Median across all (image x pass) per-stage timings; sum per-stage
     medians for the per-image total; report MPx/s.

Example::

    python -m seaotter.eval.encode_complexity \\
        --codec frp \\
        --op '{"type":"n_ch","value":3}' \\
        --dataset kodak_native \\
        --n_warmup 1 \\
        --n_measurement 5 \\
        --device cpu \\
        --out_json /tmp/encode_frp_n3_kodak.json

Output schema documented in the prompt and in the JSON `task` field
(``encode_complexity``).
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
import time
import uuid
from pathlib import Path
from statistics import mean, median

import PIL
import torch

from .codecs import CodecAdapter, make_adapter
from .datasets import (
    CROP_FOR_DATASET,
    load_inputs,
    shape_distribution,
)


DEFAULT_N_WARMUP = 1
DEFAULT_N_MEASUREMENT = 5
DEFAULT_N_IMAGES = 256

CPU_FALLBACK = "AMD EPYC 9354 32-Core Processor"


# ----------------------------------------------------------------------
# Testbed metadata
# ----------------------------------------------------------------------

def _cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _testbed_info(device: str, n_warmup: int, n_measurement: int) -> dict:
    info = {
        "hostname": socket.gethostname(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "pil_version": PIL.__version__,
        "device": device,
        "n_warmup": int(n_warmup),
        "n_measurement": int(n_measurement),
    }
    cpu = _cpu_model() or CPU_FALLBACK
    info["cpu_model"] = cpu
    info["gpu_model"] = None
    return info


# ----------------------------------------------------------------------
# Stats helpers
# ----------------------------------------------------------------------

def _stats(values: list[float]) -> dict:
    """Median + mean (in seconds) of a flat float list."""
    if not values:
        return {"median": 0.0, "mean": 0.0}
    return {
        "median": float(median(values)),
        "mean": float(mean(values)),
    }


# ----------------------------------------------------------------------
# Timing loop
# ----------------------------------------------------------------------

def _device_kind(device: str) -> str:
    s = str(device).lower()
    for kind in ("cuda", "xpu"):
        if s.startswith(kind):
            return kind
    return "cpu"


def run_codec_x_dataset(
    *,
    codec: str,
    op: dict,
    dataset: str,
    n_warmup: int = DEFAULT_N_WARMUP,
    n_measurement: int = DEFAULT_N_MEASUREMENT,
    n_images: int = DEFAULT_N_IMAGES,
    device: str = "cpu",
    out_json: str | None = None,
    verbose: bool = True,
) -> dict:
    """Run one (codec, op, dataset) cell and return the result dict.

    The result dict has the schema described in the prompt; if
    ``out_json`` is provided the dict is also written to disk.
    """
    if codec == "frp" and dataset == "clip_naflex":
        # Naflex shapes are multiples of 32 = FRAPPE max_ps; OK.
        pass

    adapter = make_adapter(codec, op, device=device)

    if verbose:
        print(
            f"[{adapter.name} {adapter.op_slug()} @ {dataset}] loading inputs ...",
            flush=True,
        )
    imgs = load_inputs(dataset, n_images)
    return _run_with_loaded_imgs(
        adapter=adapter,
        codec=codec, op=op, dataset=dataset, imgs=imgs,
        n_warmup=n_warmup, n_measurement=n_measurement,
        device=device, out_json=out_json, verbose=verbose,
    )


def run_codec_x_dataset_with_inputs(
    *,
    codec: str,
    op: dict,
    dataset: str,
    imgs: list,
    n_warmup: int = DEFAULT_N_WARMUP,
    n_measurement: int = DEFAULT_N_MEASUREMENT,
    device: str = "cpu",
    out_json: str | None = None,
    verbose: bool = True,
) -> dict:
    """Same as :func:`run_codec_x_dataset` but with pre-loaded PIL inputs.

    Used by the launcher to avoid re-loading the dataset for every cell.
    """
    adapter = make_adapter(codec, op, device=device)
    if verbose:
        print(
            f"[{adapter.name} {adapter.op_slug()} @ {dataset}] "
            f"using {len(imgs)} pre-loaded inputs ...",
            flush=True,
        )
    return _run_with_loaded_imgs(
        adapter=adapter,
        codec=codec, op=op, dataset=dataset, imgs=imgs,
        n_warmup=n_warmup, n_measurement=n_measurement,
        device=device, out_json=out_json, verbose=verbose,
    )


def _run_with_loaded_imgs(
    *,
    adapter,
    codec: str,
    op: dict,
    dataset: str,
    imgs: list,
    n_warmup: int,
    n_measurement: int,
    device: str,
    out_json: str | None,
    verbose: bool,
) -> dict:
    if verbose:
        print(f"  {len(imgs)} images; pre-staging ...", flush=True)

    # Pre-stage every input in the encoder's native form. Excluded
    # from the timer (mirrors FRAPPE harness's `inputs_on_device`).
    inputs: list = []
    for img in imgs:
        inputs.append(adapter.prepare(img))

    # Configure the wallclock singleton. `device` kind drives CUDA / XPU
    # sync hooks inside the wallclock context manager; on CPU it's a no-op.
    from throughput.image import wallclock
    wallclock.device = _device_kind(device)

    # Per-image pixel count. For variable-shape (naflex) inputs we use a
    # *per-image* pixel count list, then aggregate to a geometric-mean
    # throughput at the end.
    crop = CROP_FOR_DATASET[dataset]
    if crop is None:
        # variable shape — figure out per image
        per_img_pixels = [img.height * img.width for img in imgs]
    else:
        H, W = crop
        per_img_pixels = [H * W] * len(imgs)

    # ----------- warm-up + measurement loops -----------
    if verbose:
        print(
            f"  warmup ({n_warmup} pass{'es' if n_warmup != 1 else ''}) "
            f"+ measurement ({n_measurement} passes) ...",
            flush=True,
        )

    # Warmup: don't time.
    for _ in range(n_warmup):
        for native_input in inputs:
            _ = adapter.run_stages_untimed(native_input)

    wallclock.reset()

    # Measurement: per-stage timings collected via wallclock.
    n_inferences = 0
    t_start = time.perf_counter()
    for _ in range(n_measurement):
        for native_input in inputs:
            _ = adapter.run_stages(wallclock, native_input)
            n_inferences += 1
    t_total = time.perf_counter() - t_start

    # ----------- aggregate -----------
    stage_timings = {
        stage: list(wallclock.timings[stage]) for stage in adapter.stages
    }
    stage_stats = {stage: _stats(stage_timings[stage]) for stage in adapter.stages}

    # Per-image total time = sum of stage times image-by-image.
    n_total = len(stage_timings[adapter.stages[0]])
    per_image_totals = []
    for i in range(n_total):
        per_image_totals.append(
            sum(stage_timings[stage][i] for stage in adapter.stages)
        )
    total_stats = _stats(per_image_totals)

    # Throughput. For fixed-shape datasets, median total time / pixel-count
    # is a single MPx/s number. For naflex (variable shape) we report a
    # geometric mean across the per-image (pixels / total_time) values.
    if crop is None:
        # Per-image MPx/s, then geometric mean.
        # The timings span n_measurement passes; per_image_totals is a
        # list of length n_images*n_measurement, in the same order as
        # the input list cycled n_measurement times.
        per_img_mpxs = []
        for i, t in enumerate(per_image_totals):
            if t <= 0:
                continue
            px = per_img_pixels[i % len(per_img_pixels)]
            per_img_mpxs.append(px / 1e6 / t)
        if per_img_mpxs:
            # Geometric mean: 10**(mean(log10(x))).
            log_sum = sum(__import__("math").log(v) for v in per_img_mpxs)
            geom_mean = __import__("math").exp(log_sum / len(per_img_mpxs))
            arith_mean = sum(per_img_mpxs) / len(per_img_mpxs)
        else:
            geom_mean = 0.0
            arith_mean = 0.0
        throughput_block = {
            "median_MPx_per_s": geom_mean,        # primary metric (geometric mean)
            "mean_MPx_per_s": arith_mean,         # arithmetic mean of per-image MPx/s
            "total_per_image": total_stats,
            "aggregation": "geometric_mean_over_per_image_MPx_per_s",
        }
        median_total = total_stats["median"]
    else:
        H, W = crop
        median_total = total_stats["median"]
        mean_total = total_stats["mean"]
        throughput_block = {
            "median_MPx_per_s": (
                (H * W) * 1e-6 / median_total
                if median_total > 0 else float("inf")
            ),
            "mean_MPx_per_s": (
                (H * W) * 1e-6 / mean_total
                if mean_total > 0 else float("inf")
            ),
            "total_per_image": total_stats,
        }

    # ----------- build output -----------
    if crop is None:
        # naflex: report shape distribution too.
        input_res = [None, None]
        naflex_block = {
            "shape_distribution": shape_distribution(imgs),
        }
    else:
        input_res = [crop[0], crop[1]]
        naflex_block = {}

    result = {
        "id": uuid.uuid4().hex,
        "codec": adapter.name,
        "task": "encode_complexity",
        "dataset": dataset,
        "input_resolution": input_res,
        "n_images": len(imgs),
        "operating_point": {
            "type": op["type"],
            "value": op["value"],
            "extras": dict(op.get("extras", {})),
        },
        "stages": list(adapter.stages),
        "config": adapter.config_block(),
        "testbed": _testbed_info(device, n_warmup, n_measurement),
        "results": {
            "throughput": throughput_block,
            "stages": stage_stats,
        },
        "harness": {
            "module": "seaotter.eval.encode_complexity",
            "n_warmup": int(n_warmup),
            "n_measurement": int(n_measurement),
            "n_total_inferences": n_inferences,
            "wallclock_runtime_s": float(t_total),
        },
        **naflex_block,
    }

    if out_json is not None:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        if verbose:
            print(f"  WROTE {out_path}", flush=True)

    if verbose:
        ms = {k: stage_stats[k]["median"] * 1000 for k in adapter.stages}
        stages_str = "  ".join(f"{k}={v:7.3f}ms" for k, v in ms.items())
        print(
            f"  {stages_str}"
            f"  total={median_total*1000:7.3f}ms"
            f"  -> {throughput_block['median_MPx_per_s']:.2f} MPx/s",
            flush=True,
        )

    return result


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _parse_op(s: str) -> dict:
    op = json.loads(s)
    if "type" not in op or "value" not in op:
        raise ValueError(f"op must have type+value, got {op}")
    op.setdefault("extras", {})
    return op


def _build_argparser():
    p = argparse.ArgumentParser(
        prog="python -m seaotter.eval.encode_complexity",
        description=(
            "Encoder-only complexity harness for the SEAOTTER CoRL paper. "
            "Measures encoder forward + native quant/entropy coding only; "
            "PIL decode / dtype casts to the encoder's input form are "
            "excluded from the timer."
        ),
    )
    p.add_argument(
        "--codec", required=True,
        choices=(
            "avif", "avifx", "jpeg", "jpeg_sub0", "webp", "jp2",
            "frp", "wal", "seaotter_jpeg",
        ),
        help="Codec short code.",
    )
    p.add_argument(
        "--op", required=True, type=_parse_op,
        help='Operating point as JSON, e.g. \'{"type":"n_ch","value":3}\'.',
    )
    p.add_argument(
        "--dataset", required=True,
        choices=("cls_384", "seg_512", "clip_naflex", "kodak_native"),
        help="Dataset label.",
    )
    p.add_argument("--n_warmup", type=int, default=DEFAULT_N_WARMUP)
    p.add_argument("--n_measurement", type=int, default=DEFAULT_N_MEASUREMENT)
    p.add_argument("--n_images", type=int, default=DEFAULT_N_IMAGES)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out_json", default=None)
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None):
    args = _build_argparser().parse_args(argv)
    run_codec_x_dataset(
        codec=args.codec,
        op=args.op,
        dataset=args.dataset,
        n_warmup=args.n_warmup,
        n_measurement=args.n_measurement,
        n_images=args.n_images,
        device=args.device,
        out_json=args.out_json,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
