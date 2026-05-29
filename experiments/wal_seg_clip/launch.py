"""Launch 10 production eval cells: wal x {seg, clip} x p in {4, 16, 36, 80, 100}.

Greedy bin-packs cells across 4 GPUs by estimated runtime (clip >> seg).
Per-GPU slates run serially via subprocess; all 4 GPU slates run concurrently
when each GPU index is invoked with its own subprocess.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

REPO = Path("/home/dgj335/danjacobellis/seaotter")
PROD = REPO / "experiments/wal_seg_clip/production"
LOG = REPO / "experiments/wal_seg_clip/logs"
EVAL = REPO / "pre_trained_convnext/experiments/iter6_extra_codec_baselines/eval_accuracy.py"
VENV_PY = "/home/dgj335/g/bin/python"


class Cell(NamedTuple):
    pipeline: str    # "wal"
    task: str        # "seg" | "clip"
    op_type: str     # "pixel_ratio"
    op_value: int
    extras: dict
    est_min: int

    @property
    def short_op(self) -> str:
        return f"p{self.op_value}"

    @property
    def out_path(self) -> Path:
        return PROD / f"eval_{self.pipeline}_{self.task}_{self.short_op}.json"

    @property
    def stdout_path(self) -> Path:
        return LOG / f"eval_{self.pipeline}_{self.task}_{self.short_op}.stdout"

    def op_json(self) -> str:
        return json.dumps({"type": self.op_type, "value": self.op_value, "extras": self.extras})


def build_cells() -> list[Cell]:
    cells: list[Cell] = []
    for p in (4, 16, 36, 80, 100):
        # seg is fast (~3-5 min on 2k images)
        cells.append(Cell("wal", "seg",  "pixel_ratio", p, {}, est_min=5))
        # clip is slow (~25-35 min on 50k images, bs=1)
        cells.append(Cell("wal", "clip", "pixel_ratio", p, {}, est_min=30))
    return cells


def partition_balanced(cells: list[Cell], n_gpus: int = 4) -> list[list[Cell]]:
    """Greedy bin-packing: heaviest first, into the lightest bin."""
    bins: list[list[Cell]] = [[] for _ in range(n_gpus)]
    weights = [0] * n_gpus
    for cell in sorted(cells, key=lambda c: -c.est_min):
        i = min(range(n_gpus), key=lambda k: weights[k])
        bins[i].append(cell)
        weights[i] += cell.est_min
    return bins


def run_slate(gpu_idx: int, slate: list[Cell]) -> int:
    device = f"cuda:{gpu_idx}"
    done = 0
    for cell in slate:
        if cell.out_path.exists():
            print(f"[gpu{gpu_idx} skip] {cell.out_path.name} already exists", flush=True)
            done += 1
            continue
        t0 = time.time()
        print(f"[gpu{gpu_idx} start {time.strftime('%H:%M:%S')}] {cell.pipeline} {cell.task} {cell.short_op}", flush=True)
        cmd = [
            VENV_PY, str(EVAL),
            "--pipeline", cell.pipeline,
            "--task", cell.task,
            "--op", cell.op_json(),
            "--device", device,
            "--out_json", str(cell.out_path),
        ]
        with cell.stdout_path.open("wb") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO))
        dt = time.time() - t0
        status = "ok" if proc.returncode == 0 else f"FAIL rc={proc.returncode}"
        print(f"[gpu{gpu_idx} done  {time.strftime('%H:%M:%S')} ({dt/60:.1f} min)] {cell.pipeline} {cell.task} {cell.short_op} -> {status}", flush=True)
        if proc.returncode == 0:
            done += 1
    return done


def main() -> None:
    PROD.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    cells = build_cells()
    print(f"Total cells: {len(cells)}")
    if len(sys.argv) > 1:
        gpu_idx = int(sys.argv[1])
        bins = partition_balanced(cells)
        slate = bins[gpu_idx]
        print(f"GPU {gpu_idx}: {len(slate)} cells, est {sum(c.est_min for c in slate)} min")
        for c in slate:
            print(f"  {c.pipeline} {c.task} {c.short_op}")
        n = run_slate(gpu_idx, slate)
        print(f"GPU {gpu_idx} complete: {n}/{len(slate)} cells")
        return
    # No arg: print the plan only.
    bins = partition_balanced(cells)
    for i, b in enumerate(bins):
        est = sum(c.est_min for c in b)
        print(f"\nGPU {i}: {len(b)} cells, est {est} min")
        for c in b:
            print(f"  {c.pipeline} {c.task} {c.short_op}  ({c.est_min} min)")


if __name__ == "__main__":
    main()
