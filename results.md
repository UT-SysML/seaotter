# Results

Raw per-operating-point JSONs from the SEA OTTER paper evaluation
harness. Each row of the paper's main results figure corresponds to one
`eval_*.json` + one `throughput_*.json` pair under `results/<task>/`.

This file documents what each JSON contains so downstream readers
(figure scripts, tables, ablation plots) don't have to re-discover the
schema. Updated as additional results land.

## Layout

```
results/
└── cls/
    ├── eval/           per-pipeline × per-op accuracy + distortion + bpp
    └── throughput/     per-pipeline × per-op encode / consumer wall-clock
```

Filenames follow `eval_<pipeline>_<task>_<op>.json` and
`throughput_<pipeline>_<task>_<op>.json`, where:

- `<pipeline>` ∈ `{avif, avifx, frp, wal, seab, walsand, seaft, walft}`:
  - `avif` — AVIF (libavif default speed).
  - `avifx` — AVIF (libavif max speed, `s10`).
  - `frp` — FRAPPE codec, no SEA OTTER sandwich.
  - `wal` — WaLLoC codec, no SEA OTTER sandwich.
  - `seab` — FRAPPE + SEA OTTER zero-shot sandwich (phase-2 K=3 warm-start,
    no fine-tune).
  - `walsand` — WaLLoC + SEA OTTER zero-shot sandwich.
  - `seaft` — FRAPPE + SEA OTTER, fine-tuned against the downstream task loss.
  - `walft` — WaLLoC + SEA OTTER, fine-tuned against the downstream task loss.
- `<task>` ∈ `{cls}` for now (seg / clip to follow).
- `<op>` is the operating-point id: `q{1,5,10,25,50}` (+ `_s10` for `avifx`)
  for AVIF, `n{3,6,9,12,15}` (FRAPPE latent channel count) for FRAPPE-side
  pipelines, `p{4,16,36,80,100}` (target pixel ratio %) for WaLLoC-side.

5 operating points × 8 pipelines × {eval, throughput} = 80 files per task.

## `eval_*.json` schema

Accuracy + distortion + bitrate, computed over the full ImageNet val 50k
under the model-card squash-384² preprocessing.

| field                  | meaning                                                                                  |
|------------------------|------------------------------------------------------------------------------------------|
| `transmit_bpp_mean`    | bits-per-pixel of what crosses the wireless link (FRAPPE-LS / WaLLoC latents for SEA OTTER; full JPEG / AVIF bytes for baselines). |
| `storage_bpp_mean`     | bits-per-pixel of the final on-disk JPEG (= `transmit_bpp_mean` for baselines; **different** for SEA OTTER, since the cloud transcoder one-time-rewrites the latents to a JPEG file). |
| `metrics.bpp_std`      | per-image bpp standard deviation.                                                        |
| `metrics.top1`         | ImageNet-1k top-1 accuracy.                                                              |
| `metrics.top5`         | ImageNet-1k top-5 accuracy.                                                              |
| `metrics.psnr_db`      | reconstruction PSNR (dB) vs the original sRGB.                                           |
| `metrics.ssim`         | reconstruction SSIM.                                                                     |
| `metrics.lpips_db`     | LPIPS, reported in dB (`-10 log10 LPIPS`).                                               |
| `metrics.dists_db`     | DISTS, reported in dB.                                                                   |
| `metrics.miou` / `metrics.pixel_accuracy` | segmentation metrics — `null` for cls; populated when `<task>=seg`.        |
| `n_eval`               | sample count (50000 for cls).                                                            |
| `operating_point`      | `{type, value, extras}` — e.g. `{n_ch, 12}` or `{q_pixel_ratio, 16.0}`.                  |
| `config.codec`         | codec identifier (`seaotter` / `walft` / `avif` / ...).                                  |
| `config.checkpoint`    | path to the fine-tune checkpoint when one exists (`seaft`, `walft`); `None` otherwise.   |
| `config.phase2_init` / `phase2_k` / `phase2_arch` | warm-start pin into the phase-2 K=3 sandwich (`S3_K3_lams_0p75_0p4_0p22_w_0p3_0p7_1p5.pth`). |
| `harness_version`, `pipeline`, `pipeline_label`, `task`, `val_ds`, `val_split`, `preprocessing` | provenance / run identity. |

Fine-tuned SEA OTTER pipelines (`seaft`, `walft`) trade reconstruction
PSNR (10-22 dB) for downstream accuracy — this is the "task-driven
recon drift" framing in the paper, not a measurement error.
Zero-shot SEA OTTER (`seab`, `walsand`) keeps PSNR closer to the
codec-only baselines (24-30 dB) because the sandwich was trained
against an R-D loss, not a task loss.

## `throughput_*.json` schema

Encode and consumer wall-clock distributions over a 256-image subset.
Threading is intentionally **not** pinned (no `OMP_NUM_THREADS=1` /
`MKL_NUM_THREADS=1` / `torch.set_num_threads`) to match the
FRAPPE / Pillow encode-complexity harness methodology.

| field                                    | meaning                                                          |
|------------------------------------------|------------------------------------------------------------------|
| `throughput.encode.{median,mean,p25,p75}_ms` | sensor-side encode wall-clock per image (ms).                |
| `throughput.encode.n`                    | sample count (256).                                              |
| `throughput.consumer.{median,mean,p25,p75}_ms` | consumer-side decode + ConvNeXt forward, bs=1 (ms).        |
| `n_throughput_images`                    | sample count (256).                                              |
| `config.cpu_model`                       | CPU model (`AMD EPYC 9354 32-Core Processor`).                   |
| `config.gpu_model`                       | GPU model (`NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation`).   |
| `config.device`                          | `cuda:0` for the consumer-side timings; encoder is CPU.          |
| `config.threading`                       | `"natural (library defaults; no OMP/MKL/torch caps)"`.           |
| `operating_point`, `config.codec`, `config.checkpoint`, `harness_version`, ... | same as the eval JSON.                          |

`accuracy` / `distortion` / `bpp` fields in throughput JSONs are
intentionally `null`; pair the throughput JSON with the matching
`eval_*.json` for those.

## Provenance — `cls` (current contents)

All 40 (pipeline × op) cells were produced by the iter-6 harness at
`pre_trained_convnext/experiments/iter6_extra_codec_baselines/`. The
`walft` row was refreshed 2026-05-22 with the iter-10 LR-sweep champion
(`λ=0.05, lr_base=2e-5`); pre-iter-10 `walft` JSONs are retired in
`stale_iter7/` at the source and are not mirrored here. All other
pipelines are as-evaluated on 2026-05-19. Nothing under `cls/` has been
superseded by later (seg / clip) work.
