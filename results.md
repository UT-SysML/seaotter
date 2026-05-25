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
├── cls/
│   ├── eval/           accuracy + distortion + bpp per pipeline × op
│   └── throughput/     encode / consumer wall-clock per pipeline × op
├── seg/
│   ├── eval/           FRAPPE-side only so far
│   └── throughput/     FRAPPE-side only so far
└── clip/
    └── eval/           FRAPPE-side only so far; no throughput measured yet
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
- `<task>` ∈ `{cls, seg, clip}` (see "Task-specific dataset / preprocessing" below).
- `<op>` is the operating-point id: `q{1,5,10,25,50}` (+ `_s10` for `avifx`)
  for AVIF, `n{3,6,9,12,15}` (FRAPPE latent channel count) for FRAPPE-side
  pipelines, `p{4,16,36,80,100}` (target pixel ratio %) for WaLLoC-side.

## Task-specific dataset / preprocessing

The schema below is shared across all three tasks, but **the underlying
data distributions and image sizes are different**, so distortion /
bpp values cannot be compared across tasks directly:

| task   | val dataset                  | n_eval | preprocessing                                  | populates                                   |
|--------|------------------------------|--------|------------------------------------------------|---------------------------------------------|
| `cls`  | `timm/imagenet-1k-wds`       | 50000  | squash 384×384                                 | `metrics.top1`, `metrics.top5`              |
| `seg`  | `danjacobellis/scene_parse_150` (ADE20K) | 2000   | squash 512×512                       | `metrics.miou`, `metrics.pixel_accuracy`    |
| `clip` | `timm/imagenet-1k-wds`       | 50000  | naflex (max_num_patches=256, patch_size=16, snap=32; aspect-preserving) | `metrics.top1`, `metrics.top5` (zero-shot via SigLIP-2 prototypes) |

All distortion fields (`psnr_db`, `ssim`, `lpips_db`, `dists_db`) and
both bpp fields (`transmit_bpp_mean`, `storage_bpp_mean`) are populated
on every task — they're just computed on whichever (dataset,
preprocessing) the task uses. Compare distortion / bpp **within a
task**, not across tasks.

`clip` JSONs additionally carry three top-level naflex-config fields
(`clip_naflex_max_patches`, `clip_naflex_patch_size`,
`clip_naflex_snap`) and stash the fine-tune checkpoint pointer under
`operating_point.extras.checkpoint` (in addition to `config.checkpoint`).

## `eval_*.json` schema

Accuracy + distortion + bitrate. The fields are identical across tasks;
only `val_ds` / `n_eval` / `preprocessing` change.

| field                  | meaning                                                                                  |
|------------------------|------------------------------------------------------------------------------------------|
| `transmit_bpp_mean`    | bits-per-pixel of what crosses the wireless link (FRAPPE-LS / WaLLoC latents for SEA OTTER; full JPEG / AVIF bytes for baselines). |
| `storage_bpp_mean`     | bits-per-pixel of the final on-disk JPEG (= `transmit_bpp_mean` for baselines; **different** for SEA OTTER, since the cloud transcoder one-time-rewrites the latents to a JPEG file). |
| `metrics.bpp_std`      | per-image bpp standard deviation.                                                        |
| `metrics.top1`         | top-1 accuracy. `cls`: ImageNet-1k supervised. `clip`: SigLIP-2 zero-shot prototype matching. `null` for `seg`. |
| `metrics.top5`         | top-5 accuracy (same convention as `top1`).                                              |
| `metrics.miou`         | mean IoU on ADE20K val. Populated for `seg`; `null` otherwise.                           |
| `metrics.pixel_accuracy` | per-pixel accuracy on ADE20K val. Populated for `seg`; `null` otherwise.               |
| `metrics.psnr_db`      | reconstruction PSNR (dB) vs the original sRGB at the task's preprocessing resolution.    |
| `metrics.ssim`         | reconstruction SSIM.                                                                     |
| `metrics.lpips_db`     | LPIPS, reported in dB (`-10 log10 LPIPS`).                                               |
| `metrics.dists_db`     | DISTS, reported in dB.                                                                   |
| `metrics.elapsed_s`    | total eval wall-clock seconds.                                                           |
| `n_eval`               | sample count for this row (see Task-specific section).                                   |
| `operating_point`      | `{type, value, extras}` — e.g. `{n_ch, 12}` or `{q_pixel_ratio, 16.0}`. `clip` also stashes the fine-tune checkpoint in `extras.checkpoint`. |
| `config.codec`         | codec identifier (`seaotter` / `walft` / `avif` / ...).                                  |
| `config.checkpoint`    | path to the fine-tune checkpoint when one exists (`seaft`, `walft`); `None` otherwise.   |
| `config.phase2_init` / `phase2_k` / `phase2_arch` | warm-start pin into the phase-2 K=3 sandwich (`S3_K3_lams_0p75_0p4_0p22_w_0p3_0p7_1p5.pth`). |
| `clip_naflex_*` (clip only) | `max_patches=256`, `patch_size=16`, `snap=32`. Pin the variable-resolution preprocessing geometry. |
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

## Current contents and provenance

### `cls/` — 40 eval + 40 throughput (full pipeline coverage)

All 8 pipelines × 5 operating points, both eval and throughput.
Produced by the iter-6 harness at
`pre_trained_convnext/experiments/iter6_extra_codec_baselines/`. The
`walft` row was refreshed 2026-05-22 with the iter-10 LR-sweep champion
(`λ=0.05, lr_base=2e-5`); pre-iter-10 `walft` JSONs are retired in
`stale_iter7/` at the source and are not mirrored here. All other
pipelines are as-evaluated on 2026-05-19.

### `seg/` — 15 eval + 15 throughput (FRAPPE-side only)

FRAPPE pipelines `{frp, seab, seaft}` × `n ∈ {3, 6, 9, 12, 15}` on
ADE20K val 2k under squash-512² preprocessing. Same iter-6 harness.
WaLLoC-side seg (`wal`, `walsand`, `walft`) and AVIF / non-FRAPPE
baselines are intentionally **not** mirrored here yet.

### `clip/` — 15 eval, no throughput (FRAPPE-side only)

FRAPPE pipelines `{frp, seab, seaft}` × `n ∈ {3, 6, 9, 12, 15}` on
ImageNet val 50k under naflex preprocessing (zero-shot via SigLIP-2
prototypes). `n=12` for `frp` and `seab` was sourced from the iter-6
production directory (smoke-4 anchor row, same harness, same
preprocessing); the other 13 come from
`experiments/clip_production/production/`. **All five `seaft` cells
come from the same `checkpoint_clip_prod_frappe_n{N}.pth` sweep** —
including `n=3`. (`n=3` was the one cell where fine-tuning did not
beat the zero-shot baseline; a separate λ=0.025 re-run did
marginally better but still failed, and is intentionally **not**
mirrored here so the row reflects the original sweep.)

No `throughput_*.json` files exist for clip in the source repo —
clip-task throughput was not measured.
