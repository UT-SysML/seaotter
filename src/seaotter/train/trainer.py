"""Trainer — composes a TrainerConfig into a full training run.

Branches on (cfg.codec, cfg.task) to:
  - load codec (FRAPPE / WaLLoC), encoder frozen, decoder hot
  - load phase-2 K=3 sandwich (fwd, proxy, inv), all hot
  - load frozen teacher (ConvNeXt cls, UperNet seg, SigLIP-2 clip)
  - calibrate the RunLengthRate
  - build Adan(caution=True) over 3 param groups with raised-cosine schedule
  - run the per-task training loop, save a checkpoint + results JSON

The on-disk format uses `decoder_state_dict` for *both* codecs, keyed by
the saved `codec` field. This is backward-incompatible with the iter-7
walft format (which used `walloc_decoder_state_dict`); per the project's
"no backwards compatibility" rule, we accept the break.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import fastprogress
import numpy as np
import torch
import torch.nn as nn
from timm.optim import Adan
from torch.utils.data import DataLoader

from seaotter.rate import RunLengthRate
from seaotter.train.calibration import build_rate_calibration_set
from seaotter.train.codecs import (
    FrappeBundle, WallocBundle,
    frappe_pipeline_forward, walloc_pipeline_forward,
    load_frappe, load_walloc,
    set_frappe_decoder_freeze, set_walloc_decoder_freeze,
)
from seaotter.train.config import TrainerConfig
from seaotter.train.data import load_kodak_calib, load_train, make_collate
from seaotter.train.losses import cls_loss, seg_loss, clip_loss
from seaotter.train.sandwich import load_sandwich, set_sandwich_freeze
from seaotter.train.teachers import (
    in1k_timm_from_rgb_codec_out, in1k_timm_from_uint8,
    load_teacher,
)


def _rc_sched(
    i_step: int, total_steps: int, max_lr: float, min_lr: float, lr_pow: float,
) -> float:
    t = i_step / max(total_steps, 1)
    return (max_lr - min_lr) * (1 - np.cos(np.pi * t) ** (2 * lr_pow)) + min_lr


class Trainer:
    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.device = torch.device(cfg.device)

        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_path = self.out_dir / f"checkpoint_{cfg.exp_name}.pth"
        self.log_path = self.out_dir / f"log_{cfg.exp_name}.jsonl"
        self.results_path = self.out_dir / f"results_{cfg.exp_name}.json"

        # --- codec ----------------------------------------------------------
        if cfg.codec == "frappe":
            self.codec_bundle: FrappeBundle | WallocBundle = load_frappe(
                cfg.frappe_n_ch, self.device,
            )
            set_frappe_decoder_freeze(self.codec_bundle, hot=True)
            if cfg.task != "clip" and cfg.crop_size % self.codec_bundle.max_ps != 0:
                raise ValueError(
                    f"crop_size={cfg.crop_size} must be a multiple of FRAPPE max_ps "
                    f"({self.codec_bundle.max_ps})"
                )
        else:
            self.codec_bundle = load_walloc(
                cfg.walloc_pixel_ratio,
                # WaLLoC needs a fixed crop_size for the snap calculation;
                # clip-task variable shape is handled in the forward by
                # snapping per call.
                cfg.crop_size if cfg.crop_size is not None else 384,
                self.device,
            )
            set_walloc_decoder_freeze(self.codec_bundle, hot=True)

        # --- sandwich -------------------------------------------------------
        self.fwd, self.proxy, self.inv, self.p2meta = load_sandwich(
            cfg.phase2_init, cfg.phase2_k, self.device,
        )
        set_sandwich_freeze(self.fwd, self.proxy, self.inv, hot=True)

        # --- teacher --------------------------------------------------------
        self.teacher = load_teacher(cfg.task, self.device)

        # --- calibration ----------------------------------------------------
        calib_ds = load_kodak_calib()
        calib_imgs = build_rate_calibration_set(
            self.codec_bundle, self.fwd, calib_ds,
            task=cfg.task, crop_size=cfg.crop_size,
            n=cfg.calib_n, device=self.device,
        )
        calib_qtable = self.proxy.qtable().detach().cpu().clone()
        self.rate_proxy = RunLengthRate(calib_imgs, calib_qtable).to(self.device)

        # --- data -----------------------------------------------------------
        self.train_ds = load_train(cfg.train_ds, cfg.dataset_samples)
        self.collate_fn = make_collate(cfg.task, cfg.train_ds, cfg.crop_size)
        self.steps_per_epoch = max(1, self.train_ds.num_rows // cfg.batch_size)
        self.total_steps = cfg.epochs * self.steps_per_epoch

        # --- optimizer ------------------------------------------------------
        decoder_params = (
            list(self.codec_bundle.model.decoder.parameters())
            if cfg.codec == "frappe"
            else list(self.codec_bundle.codec.decoder.parameters())
        )
        peak_q = cfg.lr_base * cfg.lr_ratio_q
        peak_x = peak_q * cfg.lr_ratio_x
        param_groups = [
            {"params": decoder_params, "lr": 1.0},
            {"params": list(self.proxy.parameters()), "lr": 1.0},
            {"params": list(self.fwd.parameters()) + list(self.inv.parameters()), "lr": 1.0},
        ]
        self.optimizer = Adan(param_groups, lr=1.0, caution=True)
        peaks = [cfg.lr_base, peak_q, peak_x]
        lr_lambdas = [
            (lambda s, peak=p: _rc_sched(
                s, self.total_steps, peak, cfg.min_lr, cfg.lr_pow,
            ))
            for p in peaks
        ]
        self.schedule = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_lambdas,
        )

        # --- bookkeeping ----------------------------------------------------
        self._log_fp = None

    # -------------------------------------------------------------------------

    def _log(self, event: str, **fields) -> None:
        if self._log_fp is None:
            return
        rec = {"event": event, "wall_s": round(time.time(), 3)}
        rec.update(fields)
        self._log_fp.write(json.dumps(rec, default=float) + "\n")

    def _pipeline_forward(self, x_uint8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.codec == "frappe":
            return frappe_pipeline_forward(
                self.codec_bundle, self.fwd, self.proxy, self.inv,
                x_uint8, decoder_hot=True,
            )
        return walloc_pipeline_forward(
            self.codec_bundle, self.fwd, self.proxy, self.inv,
            x_uint8, decoder_hot=True,
        )

    def _train_step(self, batch) -> dict:
        cfg = self.cfg
        if cfg.task == "clip":
            x_uint8 = batch.to(self.device, non_blocking=True)
            with torch.no_grad():
                out_clean = self.teacher.vision_forward(x_uint8)
            rgb_codec_in, rgb_hat = self._pipeline_forward(x_uint8)
            out_codec = self.teacher.vision_forward(rgb_hat)
            task_loss, aux = clip_loss(out_clean, out_codec, cfg.alpha_clip)
        elif cfg.task == "cls":
            x_uint8, y = batch
            x_uint8 = x_uint8.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            rgb_codec_in, rgb_hat = self._pipeline_forward(x_uint8)
            logits_s = self.teacher(in1k_timm_from_rgb_codec_out(rgb_hat))
            task_loss = cls_loss(logits_s, y)
            aux = {}
        else:  # seg
            x_uint8 = batch.to(self.device, non_blocking=True)
            with torch.no_grad():
                logits_t = self.teacher(in1k_timm_from_uint8(x_uint8))
                pseudolabel = logits_t.argmax(dim=1).to(torch.long)
            rgb_codec_in, rgb_hat = self._pipeline_forward(x_uint8)
            logits_s = self.teacher(in1k_timm_from_rgb_codec_out(rgb_hat))
            task_loss = seg_loss(logits_s, pseudolabel)
            aux = {}

        rate_loss = self.rate_proxy(
            self.proxy._last_dct, self.proxy._last_q_map, rgb_codec_in,
        )
        loss = task_loss + cfg.lam * rate_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g in self.optimizer.param_groups for p in g["params"]],
            cfg.grad_clip, norm_type=2.0,
        )
        self.optimizer.step()
        self.schedule.step()

        return {
            "loss": float(loss.detach().item()),
            "task_loss": float(task_loss.detach().item()),
            "rate_loss": float(rate_loss.detach().item()),
            **aux,
        }

    def _set_modes(self) -> None:
        """Frozen → eval(); hot → train(). Encoder always eval."""
        if self.cfg.codec == "frappe":
            self.codec_bundle.model.eval()
            self.codec_bundle.model.decoder.train()
        else:
            self.codec_bundle.codec.eval()
            self.codec_bundle.codec.decoder.train()
        self.fwd.train(); self.proxy.train(); self.inv.train()
        if hasattr(self.teacher, "model"):  # SigLIP-2 wrapper
            self.teacher.model.eval()
        else:
            self.teacher.eval()
        self.rate_proxy.train()

    # -------------------------------------------------------------------------

    def run(self) -> dict:
        cfg = self.cfg
        self._log_fp = open(self.log_path, "w", buffering=1)
        t_start = time.time()
        try:
            self._log(
                "config",
                config=cfg.to_dict(),
                rate_alpha=self.rate_proxy.alpha.item(),
                total_steps=self.total_steps,
                phase2_arch=self.p2meta["arch"],
                phase2_lambdas=self.p2meta["lambdas"],
            )
            print(
                f"\n{'='*60}\n  {cfg.exp_name} [{cfg.codec}/{cfg.task}] on {self.device}\n"
                f"  λ={cfg.lam} lr_base={cfg.lr_base} bs={cfg.batch_size} "
                f"epochs={cfg.epochs} samples={self.train_ds.num_rows}\n"
                f"  phase2_arch={self.p2meta['arch']} k={cfg.phase2_k}\n"
                f"  rate_proxy alpha={self.rate_proxy.alpha.item():.4e} "
                f"calib_n={cfg.calib_n}\n{'='*60}",
                flush=True,
            )

            self._set_modes()

            epoch_records: list[dict] = []
            train_losses: list[float] = []
            train_rates: list[float] = []
            mb = fastprogress.master_bar(range(cfg.epochs))
            global_step = 0
            for i_epoch in mb:
                loader = DataLoader(
                    self.train_ds, batch_size=cfg.batch_size,
                    num_workers=cfg.num_workers, shuffle=True, drop_last=True,
                    collate_fn=self.collate_fn,
                    persistent_workers=(cfg.num_workers > 0),
                )
                ep_losses, ep_rates, ep_tasks = [], [], []
                ep_patch, ep_pool = [], []
                pb = fastprogress.progress_bar(loader, parent=mb)
                n_batches = len(loader)
                for i_batch, batch in enumerate(pb):
                    out = self._train_step(batch)
                    global_step += 1
                    train_losses.append(out["loss"])
                    train_rates.append(out["rate_loss"])
                    ep_losses.append(out["loss"])
                    ep_rates.append(out["rate_loss"])
                    ep_tasks.append(out["task_loss"])
                    if "loss_patch" in out:
                        ep_patch.append(out["loss_patch"])
                        ep_pool.append(out["loss_pool"])

                    if (i_batch + 1) % cfg.print_every == 0:
                        lrs_now = self.schedule.get_last_lr()
                        elapsed = time.time() - t_start
                        avg_task = float(np.mean(ep_tasks[-100:]))
                        avg_rate = float(np.mean(ep_rates[-100:]))
                        extra = ""
                        log_extra: dict = {}
                        if ep_patch:
                            avg_p = float(np.mean(ep_patch[-100:]))
                            avg_o = float(np.mean(ep_pool[-100:]))
                            extra = f" lpatch={avg_p:.4f} lpool={avg_o:.4f}"
                            log_extra = {"loss_patch": avg_p, "loss_pool": avg_o}
                        print(
                            f"  [{cfg.exp_name}] ep{i_epoch} "
                            f"batch {i_batch+1}/{n_batches} "
                            f"loss={float(np.mean(ep_losses[-100:])):+.3f} "
                            f"task={avg_task:.3f} bpp={avg_rate:.3f}{extra} "
                            f"lrs={[f'{x:.2e}' for x in lrs_now]} "
                            f"elapsed={elapsed/60:.1f}m",
                            flush=True,
                        )
                        self._log(
                            "step", step=global_step, epoch=i_epoch,
                            batch=i_batch + 1,
                            loss=float(np.mean(ep_losses[-100:])),
                            task_loss=avg_task, rate_loss=avg_rate,
                            lrs=[float(x) for x in lrs_now],
                            elapsed_s=round(elapsed, 2),
                            **log_extra,
                        )

                epoch_records.append({
                    "epoch": i_epoch,
                    "mean_loss": float(np.mean(ep_losses)),
                    "mean_task_loss": float(np.mean(ep_tasks)),
                    "mean_rate_loss": float(np.mean(ep_rates)),
                    "elapsed_min": (time.time() - t_start) / 60.0,
                })
                self._log("epoch_end", **epoch_records[-1])

            payload = self._build_checkpoint(epoch_records, train_losses, train_rates)
            torch.save(payload, self.ckpt_path)
            self._log("checkpoint_saved", path=str(self.ckpt_path))
            print(f"\n[{cfg.exp_name}] checkpoint -> {self.ckpt_path}", flush=True)

            out = {
                "exp_name": cfg.exp_name,
                "codec": cfg.codec,
                "task": cfg.task,
                "config": cfg.to_dict(),
                "phase2_arch": self.p2meta["arch"],
                "rate_alpha": self.rate_proxy.alpha.item(),
                "epoch_records": epoch_records,
                "total_time_hours": (time.time() - t_start) / 3600.0,
            }
            self.results_path.write_text(json.dumps(out, indent=2))
            print(f"[{cfg.exp_name}] results -> {self.results_path}", flush=True)
            self._log("done", total_time_hours=(time.time() - t_start) / 3600.0)
            return out
        finally:
            if self._log_fp is not None:
                self._log_fp.close()
                self._log_fp = None

    def _build_checkpoint(
        self, epoch_records: list[dict],
        train_losses: list[float], train_rates: list[float],
    ) -> dict:
        cfg = self.cfg
        decoder_sd = (
            self.codec_bundle.model.decoder.state_dict()
            if cfg.codec == "frappe"
            else self.codec_bundle.codec.decoder.state_dict()
        )
        payload: dict = {
            "config": cfg.to_dict(),
            "codec": cfg.codec,
            "task": cfg.task,
            "phase2_arch": self.p2meta["arch"],
            "phase2_init": self.p2meta["phase2_init"],
            "phase2_k": cfg.phase2_k,
            "rate_alpha": self.rate_proxy.alpha.item(),
            "epoch_records": epoch_records,
            "train_losses_tail": train_losses[-2000:],
            "train_rates_tail": train_rates[-2000:],
            "fwd_state_dict": self.fwd.state_dict(),
            "proxy_state_dict": self.proxy.state_dict(),
            "inv_state_dict": self.inv.state_dict(),
            "rate_proxy_state_dict": self.rate_proxy.state_dict(),
            "decoder_state_dict": decoder_sd,
        }
        if cfg.codec == "walloc":
            payload["walloc_pixel_ratio"] = float(cfg.walloc_pixel_ratio)
            payload["walloc_snap_h"] = int(self.codec_bundle.snap_h)
            payload["walloc_snap_w"] = int(self.codec_bundle.snap_w)
        else:
            payload["frappe_n_ch"] = int(cfg.frappe_n_ch)
        return payload
