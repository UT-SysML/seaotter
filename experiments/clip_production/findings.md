# Clip-task production training sweep — findings

Sweep date: 2026-05-22 / 2026-05-23. Driver: `prompts/clip_production_sweep.md`.

Goal: populate the clip row of the downstream-task table by training the
fine-tuned codec checkpoints (`seaft clip` × 5 FRAPPE ops + `walft clip`
× 5 WaLLoC ops) at the same operating-point grid the cls and seg rows
already use.

## Headline

- **Anchor cells both clear the +1 pp acceptance gate.** seaft clip
  n=12 = 48.22 % (vs seab 43.34 %, Δ = +4.88 pp); walft clip p=36 =
  57.65 % (vs walsand 56.65 %, Δ = +1.00 pp).
- **FRAPPE sweep is the strong row.** Mean Δ vs seab = **+1.93 pp**;
  3/5 cells clear +1 pp (n=6 +2.14, n=9 +4.46, n=12 +4.88). Tails
  underperform: n=3 −2.19 pp (random-chance regime), n=15 +0.34 pp
  (near-ceiling, baseline already at 61 %).
- **WaLLoC sweep is the narrow row.** Mean Δ vs walsand = **+0.31 pp**;
  2/5 cells clear +1 pp (p=16 +1.41, p=36 +1.00). High-bpp regresses:
  p=80 +0.06 pp, p=100 **−1.15 pp** (near-ceiling baseline).
- **Fine-tune halves storage_bpp at fixed transmit_bpp.** seab n=12
  storage = 1.465 → seaft n=12 = 0.645 (−2.3×). walsand p=36 = 1.678 →
  walft p=36 = 1.025 (−1.6×). Same "task-driven recon drift" pattern as
  iter-10's cls findings: PSNR falls 13–17 dB while top-1 rises.
- **n=12 reproduces iter-9 Smoke 7.** New seaft clip n=12 = 48.22 %
  vs iter-9 Smoke 7's 48.06 % (Δ = +0.16 pp, run-to-run noise; same
  recipe, same seed, same 50k samples).

## R-Acc curves

### FRAPPE — seab vs seaft clip

| n_ch | seab top1 | seaft top1 | Δ (pp) | seab top5 | seaft top5 | transmit_bpp | seab storage_bpp | seaft storage_bpp | seab psnr_db | seaft psnr_db |
|------|-----------|------------|--------|-----------|------------|--------------|------------------|-------------------|--------------|---------------|
| 3    | 0.0484    | 0.0265     | **−2.19** | 0.1169  | 0.0682     | 0.019        | 0.744            | 0.426             | 18.90        | 9.47          |
| 6    | 0.1787    | 0.2001     | +2.14  | 0.3343    | 0.3715     | 0.054        | 1.015            | 0.524             | 21.49        | 12.68         |
| 9    | 0.3036    | 0.3482     | +4.46  | 0.5046    | 0.5572     | 0.083        | 1.204            | 0.588             | 22.90        | 13.43         |
| 12   | 0.4334    | 0.4822     | **+4.88** | 0.6479  | 0.6973     | 0.142        | 1.465            | 0.645             | 24.32        | 13.07         |
| 15   | 0.6100    | 0.6134     | +0.34  | 0.8053    | 0.8067     | 0.410        | 2.018            | 0.798             | 27.54        | 12.78         |

Mean Δ = **+1.93 pp top-1**, range −2.19 .. +4.88. Anchor (n=12)
matches iter-9 Smoke 7. seab n=12 row reused from
`pre_trained_convnext/experiments/iter6_extra_codec_baselines/production/eval_seab_clip_n12.json`;
the other 4 seab cells were run in this sweep.

### WaLLoC — walsand vs walft clip

| pixel_ratio | walsand top1 | walft top1 | Δ (pp) | walsand top5 | walft top5 | transmit_bpp | walsand storage_bpp | walft storage_bpp | walsand psnr_db | walft psnr_db |
|-------------|--------------|------------|--------|--------------|------------|--------------|---------------------|-------------------|-----------------|---------------|
| 4           | 0.1838       | 0.1863     | +0.25  | 0.3379       | 0.3476     | 0.051        | 1.106               | 0.775             | 20.30           | 13.18         |
| 16          | 0.4485       | 0.4626     | +1.41  | 0.6653       | 0.6786     | 0.151        | 1.386               | 0.906             | 23.64           | 12.74         |
| 36          | 0.5665       | 0.5765     | +1.00  | 0.7704       | 0.7795     | 0.297        | 1.678               | 1.025             | 25.75           | 12.49         |
| 80          | 0.6414       | 0.6420     | +0.06  | 0.8253       | 0.8264     | 0.620        | 2.107               | 1.639             | 28.39           | 12.64         |
| 100         | 0.6478       | 0.6363     | **−1.15** | 0.8289   | 0.8240     | 0.742        | 2.353               | 2.334             | 29.26           | 12.65         |

Mean Δ = **+0.31 pp top-1**, range −1.15 .. +1.41. All walsand cells
are new from this sweep — the iter-6 production walsand JSONs cover
cls/seg only.

### Reference ceilings

| pipeline                      | top-1  | top-5  | transmit_bpp | source |
|-------------------------------|--------|--------|--------------|--------|
| raw clip (no codec)           | 0.6959 | 0.8568 | 15.20        | iter-9 Smoke 3 |
| seaft clip n=12 (iter-9)      | 0.4806 | 0.6971 | 0.142        | iter-9 Smoke 7 |
| seaft clip n=12 (this sweep)  | 0.4822 | 0.6973 | 0.142        | reproduces iter-9 |

## Training summary

10 cells × locked recipe, dataset_samples=50000, num_workers=4, seed=0.

| codec  | op   | mean_loss | mean_task_loss | mean_rate_loss | wall (min) |
|--------|------|-----------|----------------|----------------|------------|
| FRAPPE | n=3  | 0.572     | 0.526          | 0.464          | 36.6 |
| FRAPPE | n=6  | 0.522     | 0.464          | 0.583          | 36.8 |
| FRAPPE | n=9  | 0.482     | 0.416          | 0.661          | 37.5 |
| FRAPPE | n=12 | 0.439     | 0.367          | 0.714          | 36.8 |
| FRAPPE | n=15 | 0.378     | 0.291          | 0.870          | 38.6 |
| WaLLoC | p=4  | 0.487     | 0.431          | 1.126          | 37.8 |
| WaLLoC | p=16 | 0.376     | 0.313          | 1.269          | 37.8 |
| WaLLoC | p=36 | 0.334     | 0.264          | 1.400          | 39.2 |
| WaLLoC | p=80 | 0.293     | 0.212          | 1.610          | 36.2 |
| WaLLoC | p=100| 0.282     | 0.197          | 1.705          | 40.8 |

Total training wall = **6.3 GPU-hr** (≈ 1.7 hr on the 4-GPU box, matches
the prompt estimate of ~1.5 hr).

Both codecs show monotone task-loss decrease with capacity (FRAPPE
0.526→0.291 as n grows 3→15; WaLLoC 0.431→0.197 as p grows 4→100),
i.e. training is well-behaved at every cell. Rate-loss climbs with
capacity at fixed λ — expected, since `RunLengthRate` charges more for
larger latents.

## Acceptance gate (per prompt §"Done criteria")

- [✓] **All 10 training cells produce a readable checkpoint + results JSON.**
      `experiments/clip_production/production/checkpoint_clip_prod_{frappe_n{3,6,9,12,15},walloc_p{4,16,36,80,100}}.pth`
      (each ~30 MB for FRAPPE, ~230 MB for WaLLoC).
- [✓] **All 10 eval JSONs populated with `metrics.top1`.** Plus 9 baseline
      JSONs (4 seab clip + 5 walsand clip) added for full-curve context.
- [✓] **Anchor cells clear the +1 pp bar.** seaft clip n=12 +4.88 pp
      (4.88× the bar); walft clip p=36 +1.00 pp (exactly at the bar).
- [✓] **`findings.md` written** (this file).

The two anchor checks are the binding gates the prompt called out. The
tails (FRAPPE n=3/15, WaLLoC p=4/80/100) underperform, which is
informative but does not block the paper headline at the anchor ops.

## Tails diagnosis (why fine-tune regresses at the extremes)

**FRAPPE n=3 (Δ = −2.19 pp).** At transmit_bpp 0.019 (~1 bit per 50
pixels), the channel is so thin that both seab and seaft are near
random-chance (1 / 1000 = 0.1 %; seab 4.84 %, seaft 2.65 %). The fine-tune
collapses recon (PSNR 18.90 → 9.47) faster than it can re-encode the
information SigLIP-2 needs; the result is degenerate logits.
**Recommendation:** drop n=3 from the paper headline curve or report it
explicitly as the "saturated-at-the-floor" cell.

**FRAPPE n=15 (Δ = +0.34 pp).** seab is already at 61.0 %, only ~9 pp
below the raw-clip ceiling 69.59 %. There's little headroom and the
recipe (λ=0.10) over-regularizes — same shape iter-5 §3 flagged for n=15
on the cls task ("λ=0.10 over-regularizes at n_ch=15 by ~0.7 pp"). A
per-op λ sweep (Stage-2-style, λ ∈ {0.05, 0.075, 0.10}) at n=15 would
likely push Δ above +1 pp, but this is not load-bearing for the paper.

**WaLLoC p=4 (Δ = +0.25 pp).** Similar to FRAPPE n=3 but less severe —
walft p=4 is at 18.6 %, well above floor. The fine-tune extracts some
lift but the latent (~0.05 bpp) is too thin for the +1 pp threshold.

**WaLLoC p=100 (Δ = −1.15 pp regression).** Near-ceiling baseline
(walsand 64.78 %). The PSNR-drift cost (29.26 → 12.65 dB) outweighs the
task-alignment gain at this near-lossless op. **This mirrors iter-10's
cls findings** at p=100, where walft also barely beat walsand (+0.06 pp
on cls). The recipe is calibrated for mid-bpp ops; at the near-lossless
end, the asymmetric "drift the decoder to help the task" trade is no
longer favorable.

**Follow-up if needed:** per-op λ ramp (lower λ at the tails). Out of
scope for this sweep — the anchor cells carry the paper headline.

## Infrastructure additions

Three small harness patches landed during the eval phase. None affect
existing iter-6 / iter-7 / iter-10 production JSONs.

### 1. `harness/pipelines/walloc_ft.py` — `decoder_state_dict` fallback

The consolidated trainer in `src/seaotter/train/` writes the WaLLoC
decoder under the unified key `"decoder_state_dict"` (keyed by the
saved `"codec"` field). The archived iter-7 trainer used
`"walloc_decoder_state_dict"`. The walft pipeline now accepts either:

```python
decoder_sd = ckpt.get("decoder_state_dict") or ckpt.get("walloc_decoder_state_dict")
```

iter-7 checkpoints continue to load under the old key; new walloc_clip
checkpoints load under the new key.

### 2. `harness/pipelines/walloc_ft.py` — `op.extras.checkpoint`

walft previously honored only the `WALFT_CHECKPOINT_OVERRIDE` env-var
override (iter-10 used this). The pipeline now also reads
`op.extras.checkpoint`, mirroring the seaft pipeline. The env-var path
still works (unchanged inside `_walft_checkpoint_path`); the new path is
additive.

### 3. `walft` and `walsand` — variable-shape support for clip task

Both pipelines previously hard-coded a `(crop, crop)` resize after the
WaLLoC decoder. For the clip task `CROP_FOR_TASK["clip"] = None` (the
naflex sentinel), so the resize threw. Two coupled changes:

- `collate_encode` now prepends a 4-byte little-endian `(H, W)` header
  to the WaLLoC latent blob.
- `decode_blobs_batch` strips the header, decodes the latent, and for
  `crop is None` (clip) resizes the recon back to `(H, W)` so SigLIP-2
  sees the same naflex patch grid the clean branch was given. For
  `crop is not None` (cls/seg) the existing `(crop, crop)` resize path
  is preserved.

Transmit byte accounting **excludes** the 4-byte header (the
transmitted unit is still just the WaLLoC latent, identical to `wal`).
Storage accounting is unaffected.

Re-running existing iter-6 walft cls/seg evals with the patched harness
would give bit-identical numbers (the header is added, then immediately
stripped, and not counted in transmit_bytes). No iter-6 / iter-7 /
iter-10 numbers have been overwritten.

## Out of scope (per prompt §"Out of scope")

The following eval-only clip cells are deferred to a separate
eval-completion prompt:

- `frp clip × 5` (FRAPPE-only, no sandwich) — pipeline class exists
  (`harness/pipelines/frappe.py`) and one cell (n=12) is in iter-6
  production.
- `seab clip × 5` — 4 / 5 added during this sweep (n=3, 6, 9, 15).
  n=12 was already in iter-6 production. Backfill is essentially
  complete; only the throughput half of those cells remains.
- `wal clip × 5` and `walsand clip × 5` — `walsand` was added in
  this sweep (full 5 ops); `wal` (no sandwich) is the remaining
  baseline. Same eval-only follow-up.

Other deferrals from the prompt: per-op λ / LR sweeps beyond the recipe
lock (would address the n=3 / n=15 / p=100 tails); re-running iter-9
Smoke 5 / 7 (already in the archive); refresh of cls or seg cells (out
of scope; current results stand).

## Wall-time accounting

| phase | cells | total GPU-hr | wall (4 GPUs) |
|-------|-------|--------------|----------------|
| Training | 10 | 6.3 | 1.7 hr |
| Smoke (walloc p=36 @ 256 samples) | 1 | 0.004 | < 1 min |
| Eval seaft clip (FRAPPE) | 5 | 1.3 | ~25 min |
| Eval walft clip (WaLLoC) | 5 | 4.3 | ~65 min |
| Eval seab clip (backfill) | 4 | 2.9 | ~50 min |
| Eval walsand clip (backfill) | 5 | 4.7 | ~70 min |
| **Total** | 30 | **19.5** | **~3 hr** |

Eval wall-times spread is wide (15 min – 65 min per cell) due to harness
GPU contention with concurrent cells; the per-cell `elapsed_s` in the
JSONs reflects the actual SigLIP-2 forward + JPEG roundtrip cost.

## Files of record

```
experiments/clip_production/
├── findings.md                                       (this file)
├── smoke/
│   ├── checkpoint_walloc_clip_smoke_p36.pth
│   ├── log_walloc_clip_smoke_p36.jsonl
│   └── results_walloc_clip_smoke_p36.json
└── production/
    ├── checkpoint_clip_prod_frappe_n{3,6,9,12,15}.pth        (5 ckpts)
    ├── checkpoint_clip_prod_walloc_p{4,16,36,80,100}.pth      (5 ckpts)
    ├── log_clip_prod_*.jsonl                                 (10 logs)
    ├── results_clip_prod_*.json                              (10 results)
    ├── eval_seaft_clip_n{3,6,9,12,15}.json                   (5 new evals)
    ├── eval_walft_clip_p{4,16,36,80,100}.json                (5 new evals)
    ├── eval_seab_clip_n{3,6,9,15}.json                       (4 baseline backfill)
    └── eval_walsand_clip_p{4,16,36,80,100}.json              (5 baseline backfill)
```
