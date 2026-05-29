"""FRAPPE + ConvNeXt cls — iter-5 recipe.

Locked: ImageNet train + GT CE + squash 384², λ=0.10, lr_base=2e-4 @ bs=8,
1 epoch. Validated by iter-5 (mean Δ vs iter-4 S2 = +4.58 pp across 5
ops). n_ch is the operating-point knob; production sweep uses
n_ch ∈ {3, 6, 9, 12, 15}.
"""

from __future__ import annotations

import argparse

from seaotter.train.config import TrainerConfig
from seaotter.train.trainer import Trainer


LOCKED = dict(
    codec="frappe", task="cls",
    frappe_n_ch=12,
    lam=0.10, lr_base=2e-4, lr_ratio_q=0.5, lr_ratio_x=1.0,
    batch_size=8, epochs=1,
    train_ds="timm/imagenet-1k-wds",
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", required=True)
    p.add_argument("--exp_name", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--frappe_n_ch", type=int, default=LOCKED["frappe_n_ch"])
    p.add_argument("--dataset_samples", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = TrainerConfig(
        codec=LOCKED["codec"], task=LOCKED["task"],
        frappe_n_ch=args.frappe_n_ch,
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
