"""WaLLoC + SigLIP-2 clip — forward-projected from iter-10 + iter-9.

Locked: ImageNet train + hybrid per-patch + pooled cosine (α_clip=0.7),
λ=0.05 (mirrors iter-7 LAM_CLS for WaLLoC), lr_base=7e-6 @ bs=1
(= iter-10 champion 2e-5 @ bs=8 sqrt-rescaled to bs=1), 1 epoch.

NOT YET VALIDATED end-to-end. This recipe is the projection of two
known truths: (a) FRAPPE-clip locks at λ=0.10, lr_base=7e-5 (iter-9
Smoke 7), and (b) the FRAPPE→WaLLoC LR mapping is 10× lower
(iter-10 + `[[walloc-decoder-fine-tune-lr]]`). The first end-to-end
run is its validation. If clip top-1 regresses below `seab clip` at
p=36, apply iter-10's eval-driven sweep playbook (Stage 1: sweep λ at
fixed LR; Stage 2: sweep LR at champion λ).
"""

from __future__ import annotations

import argparse

from seaotter.train.config import TrainerConfig
from seaotter.train.trainer import Trainer


LOCKED = dict(
    codec="walloc", task="clip",
    walloc_pixel_ratio=36,
    lam=0.05, lr_base=7e-6, lr_ratio_q=0.5, lr_ratio_x=1.0,
    alpha_clip=0.7,
    batch_size=1, epochs=1,
    train_ds="timm/imagenet-1k-wds",
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
        alpha_clip=LOCKED["alpha_clip"],
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
