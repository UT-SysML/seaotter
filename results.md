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
├── clip/
│   └── eval/           FRAPPE-side only so far; no throughput measured yet
├── codec_kodak/        standalone codec eval on Kodak (no downstream task)
└── codec_kodak_cls/    standalone codec eval on ImageNet 50k under the cls protocol
```

Filenames follow `eval_<pipeline>_<task>_<op>.json` and
`throughput_<pipeline>_<task>_<op>.json`, where:

- `<pipeline>` ∈ `{avif, avifx, jpeg, jp2, webp, frp, wal, seab, walsand, seaft, walft, raw}`:
  - `avif` — AVIF (libavif default speed).
  - `avifx` — AVIF (libavif max speed, `s10`).
  - `jpeg` — Vanilla JPEG baseline (Pillow defaults; 4:2:0 chroma subsampling).
  - `jp2` — JPEG 2000 (Pillow / OpenJPEG defaults).
  - `webp` — WebP (libwebp defaults).
  - `frp` — FRAPPE codec, no SEA OTTER sandwich.
  - `wal` — WaLLoC codec, no SEA OTTER sandwich.
  - `seab` — FRAPPE + SEA OTTER zero-shot sandwich (phase-2 K=3 warm-start,
    no fine-tune).
  - `walsand` — WaLLoC + SEA OTTER zero-shot sandwich.
  - `seaft` — FRAPPE + SEA OTTER, fine-tuned against the downstream task loss.
  - `walft` — WaLLoC + SEA OTTER, fine-tuned against the downstream task loss.
  - `raw` — no-codec ceiling (lossless PNG reference). Single `_ref` op per task;
    no `throughput_*` companion (`raw` is a reconstruction-fidelity / accuracy upper bound, not a deployment configuration).
- `<task>` ∈ `{cls, seg, clip}` (see "Task-specific dataset / preprocessing" below).
- `<op>` is the operating-point id, **per pipeline family**:
  - `q{1,5,10,25,50}` (+ `_s10` suffix for `avifx`) — AVIF / JPEG / WebP quality.
  - `r{12,25,50,100,200}` — JPEG 2000 compression ratio (Pillow `quality_layers` semantics).
  - `n{3,6,9,12,15}` — FRAPPE latent channel count (FRAPPE-side pipelines).
  - `p{4,16,36,80,100}` — target pixel-ratio % (WaLLoC-side pipelines).
  - `ref` — single no-codec reference cell for the `raw` pipeline.

## Task-specific dataset / preprocessing

The schema below is shared across all three tasks, but **the underlying
data distributions and image sizes are different**, so distortion /
bpp values cannot be compared across tasks directly:

| task           | val dataset                  | n_eval | preprocessing                                  | populates                                   |
|----------------|------------------------------|--------|------------------------------------------------|---------------------------------------------|
| `cls`          | `timm/imagenet-1k-wds`       | 50000  | squash 384×384                                 | `metrics.top1`, `metrics.top5`              |
| `seg`          | `danjacobellis/scene_parse_150` (ADE20K) | 2000   | squash 512×512                       | `metrics.miou`, `metrics.pixel_accuracy`    |
| `clip`         | `timm/imagenet-1k-wds`       | 50000  | naflex (max_num_patches=256, patch_size=16, snap=32; aspect-preserving) | `metrics.top1`, `metrics.top5` (zero-shot via SigLIP-2 prototypes) |
| `kodak_recon`  | `danjacobellis/kodak`        | 24     | native (no resize, no crop; 768×512 or 512×768; bs=1) | per-image + summary `bpp`, `psnr_db`, `ssim`, `lpips_db`, `dists_db` (no downstream task accuracy) |
| `kodak_cls`    | `timm/imagenet-1k-wds`       | 50000  | squash 384×384; bpp denominator pinned at 384·384 = 147456 | `metrics.{top1, top5, psnr_db, ssim, lpips_db, dists_db, bpp_mean, bpp_std}` — standalone codec only (no FRAPPE upstream); `metrics.{miou, pixel_accuracy} = null` |

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

### `cls/` — 56 eval + 55 throughput

8 SEA OTTER / FRAPPE / WaLLoC / AVIF pipelines × 5 ops + 3 alt-codec
baselines × 5 ops + 1 raw (no-codec) ceiling row. Produced by the
iter-6 harness at
`pre_trained_convnext/experiments/iter6_extra_codec_baselines/`.

- The 8-pipeline core (`avif, avifx, frp, wal, seab, walsand, seaft, walft`)
  × 5 ops × {eval, throughput} = 80 files. The `walft` row was refreshed
  2026-05-22 with the iter-10 LR-sweep champion (`λ=0.05, lr_base=2e-5`);
  pre-iter-10 `walft` JSONs are retired in `stale_iter7/` at the source and
  are not mirrored here. All other 8-pipeline cells are as-evaluated on
  2026-05-19.
- Alt-codec baselines: `jpeg` (q ∈ {1, 5, 10, 25, 50}), `jp2` (r ∈
  {12, 25, 50, 100, 200}), `webp` (q ∈ {1, 5, 10, 25, 50}) × {eval,
  throughput} = 30 files.
- Raw ceiling: `eval_raw_cls_ref.json` (no-codec lossless reference;
  85.13% top-1 — matches the iter-4 §0 anchor and the
  `convnext_tiny.in12k_ft_in1k_384` model card). No throughput
  companion (raw is a ceiling, not a deployment configuration).

### `seg/` — 41 eval (throughput pruned 2026-05-29 — see [`DELETIONS.md`](DELETIONS.md))

FRAPPE pipelines + WaLLoC-zero-shot/fine-tuned coverage + AVIF / alt-codec
baselines + raw ceiling, all on ADE20K val 2k under squash-512²
preprocessing. Same iter-6 harness.

- FRAPPE pipelines `{frp, seab, seaft}` × `n ∈ {3, 6, 9, 12, 15}` ×
  {eval, throughput} = 30 files.
- Alt-codec baselines: `avif` / `avifx` (q ∈ {1, 5, 10, 25, 50}, `_s10`
  suffix on `avifx`), `jpeg` (q ∈ {1, 5, 10, 25, 50}), `jp2` (r ∈ {12,
  25, 50, 100, 200}), `webp` (q ∈ {1, 5, 10, 25, 50}) × {eval,
  throughput} = 50 files.
- Raw ceiling: `eval_raw_seg_ref.json` (no-codec lossless reference).
  Iter-6 harness `raw seg` reproduces ~44.51 % mIoU under the squash-512²
  protocol — about 1.5 pp below the sliding-window paper-protocol number
  (45.96 % mIoU) for the same teacher. No throughput companion.
- WaLLoC-side seg (`wal`, `walsand`, `walft`) is **not** mirrored here.

### `clip/` — 16 eval, no throughput (FRAPPE-side + raw ceiling)

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

- Raw ceiling: `eval_raw_clip_ref.json` (no-codec lossless reference;
  0.6959 zero-shot top-1 / 0.8568 top-5 via SigLIP-2 — matches iter-9
  Smoke 3).
- WaLLoC-side clip (`wal`, `walsand`, `walft`) is **not** mirrored here.

No `throughput_*.json` files exist for clip in the source repo —
clip-task throughput was not measured.

### `codec_kodak/` — 17 eval cells + `summary.json` (standalone codec, no downstream task)

**Harness version: `codec-kodak-v1`** (separate harness from the
iter-6 pipeline harness; different schema — see below). Output of
`experiments/codec_kodak_eval/run_eval.py` in the research tree, run
CPU-only at `bs=1` on Kodak validation at **native resolution** (no
resize, no crop; 16 images at 768×512 + 8 at 512×768).

Compares three codecs:

- `seaotter` — SEA OTTER S3 K=3 production bundle
  (`danjacobellis/seaotter @ seaotter_jpeg_s3`,
  λ = [0.75, 0.40, 0.22]; this is the warm-start used by every
  phase-4 production pipeline at `phase2_k=2`). Loaded via
  `from seaotter import load_from_hub; bundle = load_from_hub()`
  (S3 is the package default).
- `jpeg` — Pillow JPEG, default chroma subsampling (4:2:0).
- `jpeg_sub0` — Pillow JPEG, no chroma subsampling (4:4:4).

7-q ladder (`q0p5`, `q1`, `q1p5`, `q2`, `q2p5`, `q3`, `q3p5`) selected
by anchoring `q1` / `q2` / `q3` to the **smallest integer JPEG-sub=0
quality** where SEA OTTER@k strictly dominates JPEG-sub=0@q on both
mean Kodak PSNR (SEA OTTER > JPEG) and mean Kodak bpp (SEA OTTER
< JPEG), then interpolating:

- `q1p5 = round((q1 + q2) / 2)`
- `q2p5 = round((q2 + q3) / 2)`
- `q0p5 = round(q1 - (q2 - q1) / 2)`
- `q3p5 = round(q3 + (q3 - q2) / 2)`

(all clamped to `[1, 99]`). The same q values are reused for the
default-subsampling JPEG variant.

**Per-cell file schema** (different from the iter-6 envelope —
no `metrics.*`, no `transmit_bpp_*` / `storage_bpp_*` split since
storage ≡ transmit ≡ JPEG bytes for a standalone codec):

| field                       | meaning                                                                  |
|-----------------------------|--------------------------------------------------------------------------|
| `per_image.bpp`             | per-image bit-rate `8 * len(jpeg_bytes) / (H * W)` (length-24 array).     |
| `per_image.psnr_db`         | per-image PSNR (dB), uint8 vs uint8 reconstruction.                       |
| `per_image.ssim`            | per-image SSIM via `piq.ssim(data_range=1.0)`.                            |
| `per_image.lpips_db`        | per-image `-10 * log10(piq.LPIPS()(...))`. piq net on CPU, `[0, 1]` input. |
| `per_image.dists_db`        | per-image `-10 * log10(piq.DISTS()(...))`. piq net on CPU, `[0, 1]` input. |
| `per_image.image_h`, `image_w` | native pixel dimensions per image (no resize).                        |
| `per_image.kodak_index`     | source-row index in `danjacobellis/kodak[validation]` (0..23).            |
| `summary.{bpp,psnr_db,ssim,lpips_db,dists_db}` | `{mean, median}` over the 24 images.                  |
| `operating_point.type`      | `seaotter_k` for SEA OTTER cells, `jpeg_q` for JPEG cells.                |
| `operating_point.value`     | `k ∈ {0, 1, 2}` (seaotter) or integer Pillow quality (jpeg).              |
| `operating_point.ladder_id` | (jpeg only) `q0p5` / `q1` / … so a JSON can be located by ladder rung.    |
| `config.codec`              | `seaotter` / `jpeg` / `jpeg_sub0`.                                        |
| `config.subsampling`        | `0` for jpeg_sub0 + seaotter, `2` (= 4:2:0) for the default jpeg variant. |
| `config.quality`            | Pillow `quality` (null for seaotter).                                     |
| `config.seaotter_k`         | `0` / `1` / `2` for seaotter, null otherwise.                             |
| `config.seaotter_source`    | `"danjacobellis/seaotter @ seaotter_jpeg_s3 (load_from_hub default)"`.    |
| `config.seaotter_lambdas`   | `[0.75, 0.4, 0.22]` (S3 production sister).                               |
| `config.device`             | `"cpu"` — no GPU used anywhere.                                           |

Distortion metric conventions match the iter-6 harness
(`piq.psnr` / `piq.ssim` / `piq.LPIPS` / `piq.DISTS` on `[0, 1]`
float tensors; LPIPS/DISTS reported in dB via `-10·log10`), so
codec_kodak distortion values can be compared directly with iter-6
`metrics.{psnr_db, ssim, lpips_db, dists_db}` on a like-for-like
preprocessing basis — but **not** numerically against the other
tasks here, since the underlying images and resolutions differ
(see "Task-specific dataset / preprocessing").

**`summary.json`** (side artifact) records the q-selection
process: the three SEA OTTER anchors with mean bpp/PSNR and the
matched JPEG-sub=0 q; the full JPEG-sub=0 mean-bpp / mean-PSNR
sweep over q ∈ [1, 99] used for anchor selection; the resolved
q-ladder; any anchor/interpolation collisions; and the verbatim
q-selection + interpolation rules.

**File list** (18 total):

```
codec_kodak/
├── eval_seaotter_kodak_k{0,1,2}.json
├── eval_jpeg_kodak_{q0p5,q1,q1p5,q2,q2p5,q3,q3p5}.json
├── eval_jpeg_sub0_kodak_{q0p5,q1,q1p5,q2,q2p5,q3,q3p5}.json
└── summary.json
```

No `throughput_*.json` for this harness — codec_kodak is a
distortion-only standalone-codec eval; sensor / consumer
throughput numbers for the codecs in scope live in `cls/throughput/`
(SEA OTTER appears there as `seab`, JPEG as the JPEG-only / no-codec
baselines).

### `codec_kodak_cls/` — 17 eval cells + raw anchor (standalone codec, ImageNet 50k cls)

**Harness version: `codec-kodak-cls-v1`** — companion to `codec_kodak/`.
Same three codecs and same 17 operating points (3 SEA OTTER `k ∈ {0, 1, 2}`,
7 `jpeg` 4:2:0 + 7 `jpeg_sub0` 4:4:4 over the
`{q0p5, q1, q1p5, q2, q2p5, q3, q3p5}` ladder loaded verbatim from the
Kodak-native `codec_kodak/summary.json`), but evaluated this time on
ImageNet val 50k under the cls protocol (`squash 384×384` →
`convnext_tiny.in12k_ft_in1k_384` top-1/top-5 + the same
`piq.{psnr, ssim, LPIPS, DISTS}` distortion metrics + bpp). No FRAPPE
upstream — these codecs operate directly on the 384²-squashed RGB.

Output of `experiments/codec_kodak_eval/run_cls_eval.py` in the research
tree, run single-process on `cuda:0` (the codec round-trip stays on CPU
via Pillow / `seaotter.load_from_hub`; only the ConvNeXt teacher + the
piq.LPIPS / piq.DISTS nets go to GPU). bpp denominator is pinned at
`384·384 = 147456` (`config.bpp_denominator`), so per-cell
`metrics.bpp_mean = 8 * mean(len(jpeg_bytes)) / 147456`.

Schema mirrors the iter-6 `eval_*.json` envelope so figures / tables
can stack `codec-kodak-cls-v1` rows next to existing
`seab` / `walsand` / `seaft` / `walft` rows without munging:
`metrics.{top1, top5, bpp_mean, bpp_std, psnr_db, ssim, lpips_db, dists_db}`
populated; `metrics.{miou, pixel_accuracy} = null` (cls task);
`storage_bpp_mean == transmit_bpp_mean` (no FRAPPE/transcode split for
a standalone codec). `config.codec` is one of
`{seaotter, jpeg, jpeg_sub0}`; the SEA OTTER cells additionally carry
`config.seaotter_k`, `config.seaotter_source`, and
`config.seaotter_lambdas = [0.75, 0.4, 0.22]`. The raw (no-codec)
ImageNet 50k cls anchor is also persisted here as
`eval_raw_cls_kodak_anchor.json` (top-1 = 85.13%, top-5 = 97.616%,
exact match to the iter-4 §0 anchor and to `cls/eval/eval_raw_cls_ref.json`);
the anchor file carries only `metrics.{top1, top5, elapsed_s}` — no
distortion / bpp fields.

**File list** (18 total):

```
codec_kodak_cls/
├── eval_raw_cls_kodak_anchor.json
├── eval_seaotter_cls_kodak_k{0,1,2}.json
├── eval_jpeg_cls_kodak_{q0p5,q1,q1p5,q2,q2p5,q3,q3p5}.json
└── eval_jpeg_sub0_cls_kodak_{q0p5,q1,q1p5,q2,q2p5,q3,q3p5}.json
```

No `throughput_*.json` companion — `codec_kodak_cls` is an
inference-only accuracy + distortion eval. The relevant sensor /
consumer throughput numbers for these codecs live in `cls/throughput/`
(SEA OTTER as `seab`, vanilla JPEG via the iter-6 `jpeg` baseline).

## Post-submission maintenance (2026-05-29, public-release prep)

- **`seg/throughput/` pruned.** The 40 iter-6 (`iter6-1`, `device=cuda:0`)
  seg throughput JSONs were removed. Their **encode** field is superseded
  by `encode_complexity/seg_512/`, and their **consumer** field by the
  iter-11 CPU `cls/throughput/` measurement (decode MPx/s is treated as
  task-invariant in the tables, so all tasks read `cls/throughput`). No
  figure/table generator reads `seg/throughput/`; every paper figure and
  table regenerates byte-identically after removal. Recoverable from git
  history. Full rationale + the kept-set in [`DELETIONS.md`](DELETIONS.md).
- **The per-section counts above are historical.** They predate the
  matched-rate / storage-CR backfill; the eval directories now hold more
  cells (`cls` 69, `seg` 73, `clip` 54, `codec_kodak` 42). The additions
  are extra operating points and the `frp_jpeg` / `wal_jpeg` storage-CR
  cells — all in scope. Nothing was removed except `seg/throughput/`.
- **Traceability.** Every reported number is mapped to its backing file
  and generator in [`TRACEABILITY.md`](TRACEABILITY.md).
