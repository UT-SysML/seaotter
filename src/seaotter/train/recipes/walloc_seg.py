"""WaLLoC + UperNet seg — iter-7 LAM_SEG recipe.

Locked: LSDIR + UperNet pseudolabels + 512² random crop, λ=0.10,
lr_base=2e-4 @ bs=2, 4 epochs LSDIR. Inherits iter-7's recipe verbatim.

OPEN RECIPE QUESTIONS (see prompts/organize_src_code.md §"Open recipe
questions" #1 + #2):
  1. ADE20k + GT refresh has never been validated for seg on either
     codec. The natural mirror of iter-5 / iter-10 hasn't been run.
  2. The iter-10-style LR refresh for WaLLoC seg was a stage-2 candidate
     called out at iter-10 §5 but never run. iter-10's cls champion was
     seg-neutral in the p=36 spot-check, but a seg-specific sweep may
     find a real lift. Per `[[walloc-decoder-fine-tune-lr]]`, the
     iter-10-derived starting point would be lr_base=2e-5 (vs the
     iter-7-inherited 2e-4 locked here).
"""

from __future__ import annotations

import argparse

from seaotter.train.config import TrainerConfig
from seaotter.train.trainer import Trainer


LOCKED = dict(
    codec="walloc", task="seg",
    walloc_pixel_ratio=36,
    lam=0.10, lr_base=2e-4, lr_ratio_q=0.5, lr_ratio_x=1.0,
    batch_size=2, epochs=4,
    train_ds="danjacobellis/LSDIR",
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", required=True)
    p.add_argument("--exp_name", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--walloc_pixel_ratio", type=float,
                   default=LOCKED["walloc_pixel_ratio"])
    p.add_argument("--dataset_samples", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = TrainerConfig(
        codec=LOCKED["codec"], task=LOCKED["task"],
        walloc_pixel_ratio=args.walloc_pixel_ratio,
        lam=LOCKED["lam"], lr_base=LOCKED["lr_base"],
        lr_ratio_q=LOCKED["lr_ratio_q"], lr_ratio_x=LOCKED["lr_ratio_x"],
        batch_size=LOCKED["batch_size"], epochs=LOCKED["epochs"],
        train_ds=LOCKED["train_ds"],
        num_workers=args.num_workers,
        dataset_samples=args.dataset_samples,
        device=args.device, exp_name=args.exp_name, out_dir=args.out_dir,
        seed=args.seed,
    )
    Trainer(cfg).run()


if __name__ == "__main__":
    main()
