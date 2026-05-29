"""WaLLoC + ConvNeXt cls — iter-10 Stage-2 champion recipe.

Locked: ImageNet train + GT CE + squash 384², λ=0.05, lr_base=2e-5 @ bs=8
(10× below FRAPPE's anchor — the SD-VAE-class WaLLoC decoder is ~57M
params), 1 epoch. Beats walsand at every cls op (mean +2.78 pp) and
iter-7 walft at every (task, op) measured (mean +5.56 pp). Production
sweep is pixel_ratio ∈ {4, 16, 36, 80, 100}.

See `[[walloc-decoder-fine-tune-lr]]` — do NOT inherit FRAPPE's
lr_base=2e-4 for any hot-WaLLoC-decoder run.
"""

from __future__ import annotations

import argparse

from seaotter.train.config import TrainerConfig
from seaotter.train.trainer import Trainer


LOCKED = dict(
    codec="walloc", task="cls",
    walloc_pixel_ratio=16,
    lam=0.05, lr_base=2e-5, lr_ratio_q=0.5, lr_ratio_x=1.0,
    batch_size=8, epochs=1,
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
