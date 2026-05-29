# λ sweep — outlier closure attempt

Goal: close three single-cell regressions identified in
[`recipes.md`](../../recipes.md) where the fine-tuned codec (`seaft` /
`walft`) underperformed `max(V, ZS)` — FRAPPE seg n=15, FRAPPE clip n=3,
WaLLoC clip p=100 — by sweeping λ around the recipe-locked value while
holding every other knob fixed. And test whether the FRAPPE-aligned
WaLLoC seg recipe (LSDIR + pseudolabel @ lr=2e-5) lifts WaLLoC seg p=36
above ZS.

Recipe followed: [`prompts/lambda_sweep_outliers.md`](../../prompts/lambda_sweep_outliers.md).
Wrapper: [`tools/lambda_sweep_cell.py`](../../tools/lambda_sweep_cell.py)
overrides `lam` and `lr_base` against `TrainerConfig`/`Trainer` — the
LOCKED recipes in `src/seaotter/train/recipes/` were not modified.

## Verdict

| Stage | Outlier cell        | Best λ | FT vs `max(V, ZS)` | Verdict |
|-------|---------------------|--------|--------------------|---------|
| 1     | FRAPPE seg n=15     | **0.025**  | **+0.66 pp mIoU** | **PASS** |
| 2     | FRAPPE clip n=3     | 0.025  | −2.14 pp top-1     | **FAIL** (every λ) |
| 3     | WaLLoC clip p=100   | **0.025**  | **+0.12 pp top-1** | **PASS** (tied with 0.0125 at lower storage_bpp) |
| 4     | WaLLoC seg p=36 (FRAPPE-aligned) | 0.025 | −0.47 pp mIoU vs ZS | **FAIL** (every λ); also below iter-10 production walft (0.3871) |
| 5     | (extend champion to other walloc seg ops) | — | — | **Deferred** — Stage 4 found no passing λ AND user requested GPUs for other work |

Headline: two of three single-cell regressions close cleanly by halving
the locked λ. The third (FRAPPE clip n=3, the lowest-rate point on the
clip Pareto frontier) does not respond to λ; it needs an LR sweep or a
recipe rethink. Stage 4 closes the open question about whether the
FRAPPE-aligned recipe (LSDIR + pseudolabel) lifts WaLLoC seg — it does
not, at lr=2e-5; the current production walft-seg p=36 (iter-10
cls-champion recipe) already beats every Stage-4 cell.

## Stage 1 — FRAPPE seg n=15 λ sweep

Locked recipe: `lr_base=2e-4 @ bs=2`, `lr_ratio_q=0.5`, `lr_ratio_x=1.0`,
4 epochs LSDIR + UperNet pseudolabel + 512² aug. Only λ varies.

| λ      | FT mIoU | Δ vs max(V, ZS)=0.3942 | Pass? | transmit_bpp |
|--------|---------|------------------------|-------|--------------|
| 0.025  | **0.4008** | **+0.66 pp** | **yes** | 0.3042 |
| 0.05   | 0.3990  | +0.48 pp  | yes   | 0.3042 |
| 0.10*  | 0.3859  | −0.83 pp  | no    | 0.3042 |
| 0.20   | 0.3704  | −2.38 pp  | no    | 0.3042 |
| 0.40   | 0.3531  | −4.11 pp  | no    | 0.3042 |

*recipe-locked baseline (no new training; row reproduces recipes.md).*

**Champion: λ=0.025 (FT mIoU 0.4008, +0.66 pp).** The recipe-locked
λ=0.10 over-regularizes at the high-channel end of the FRAPPE seg
operating range — exactly the iter-5 §3 flag that this sweep was sent
to test. Two adjacent λ pass; sweeping is robust.

Reconstruction-metric trade-off (informational; for the paper-level
view see `recipes.md`):

| λ      | PSNR (dB) | SSIM | LPIPS-dB | DISTS-dB |
|--------|-----------|------|----------|----------|
| 0.025  | 15.50     | 0.81 | 4.72     | 8.91     |
| 0.05   | 13.53     | 0.75 | 4.63     | 8.26     |
| 0.20   | 11.02     | 0.61 | 3.99     | 6.79     |
| 0.40   | 10.14     | 0.53 | 3.50     | 6.03     |

Lower λ keeps more spatial detail (PSNR/SSIM rise) and the seg head wins
because mIoU rewards crisp boundaries the segmentor was trained to
expect.

## Stage 2 — FRAPPE clip n=3 λ sweep

Locked recipe: `lr_base=7e-5 @ bs=1`, `α_clip=0.7`, 1 epoch ImageNet
(`dataset_samples=50000`), hybrid SigLIP-2 cosine loss. Only λ varies.

| λ      | FT top-1 | Δ vs max(V, ZS)=0.0484 | Pass? | transmit_bpp |
|--------|----------|------------------------|-------|--------------|
| 0.025  | 0.0270   | −2.14 pp  | no    | 0.0187 |
| 0.05   | 0.0261   | −2.23 pp  | no    | 0.0187 |
| 0.10*  | 0.0265   | −2.19 pp  | no    | 0.0187 |
| 0.20   | 0.0205   | −2.79 pp  | no    | 0.0187 |
| 0.40   | 0.0010   | −4.74 pp  | no    | 0.0187 |

*recipe-locked baseline.*

**Champion: λ=0.025 (FT top-1 0.0270, −2.14 pp) — fails.** Sweeping λ
moves the metric by at most 0.005 around the locked baseline; the
0.0484 → 0.0265 gap is structural, not a λ tuning miss. n=3 sits at
transmit_bpp ≈ 0.019, where the FRAPPE encoder has only 3 channels
left after the codec; the seab/seaft+clip pipeline is below the
sandwich's "linear" operating region.

**Open question deferred:** per the prompt fallback, the natural next
move at this cell is an LR sweep (lower lr_base at the champion λ).
Do **not** retry λ here — Stage 2 establishes that λ is not the
limiting axis.

## Stage 3 — WaLLoC clip p=100 λ sweep

Locked recipe: `lr_base=7e-6 @ bs=1` (= iter-10 cls champion sqrt-rescaled
for bs=1), `α_clip=0.7`, 1 epoch ImageNet, hybrid SigLIP-2 cosine.

| λ       | FT top-1  | Δ vs max(V, ZS)=0.6478 | Pass? | transmit_bpp | storage_bpp |
|---------|-----------|------------------------|-------|--------------|-------------|
| 0.0125  | **0.6490** | **+0.12 pp** | **yes** | 0.7422 | 3.66 |
| 0.025   | **0.6490** | **+0.12 pp** | **yes** | 0.7422 | 3.40 |
| 0.05*   | 0.6363    | −1.15 pp  | no    | 0.7422 | (locked) |
| 0.10    | 0.6154    | −3.24 pp  | no    | 0.7422 | 2.95 |
| 0.20    | 0.5802    | −6.76 pp  | no    | 0.7422 | 1.65 |

*recipe-locked baseline.*

**Champion: λ=0.025** (tied with λ=0.0125 at top-1=0.6490, but lower
storage_bpp at the same task-metric value: 3.40 vs 3.66). Same pattern
as Stage 1 — halving the locked λ closes the regression. The locked
λ=0.05 (a forward-projection from iter-7's λ=0.05 LAM_CLS) is too
heavy for clip on the highest-rate operating point.

## Stage 4 — WaLLoC seg p=36, FRAPPE-aligned recipe + λ sweep

Recipe: `seaotter.train.recipes.walloc_seg` LOCKED (LSDIR + UperNet
pseudolabel + 512² aug + bs=2 + 4 epochs) **but with lr_base lowered
to 2e-5** (the iter-10 WaLLoC LR lesson — see
`[[walloc-decoder-fine-tune-lr]]`; FRAPPE seg's locked 2e-4 is 10× too
high for the WaLLoC decoder per iter-10 cls).

This was the open-recipe question: does the FRAPPE-aligned data axis
(LSDIR + pseudolabel) beat the iter-7 walft-seg data axis (ADE20K + GT)
once you correct the LR — and if so, what's the right λ?

Reference values:

| Pipeline / source                 | p=36 mIoU |
|-----------------------------------|-----------|
| wal (vanilla)                     | 0.3833    |
| walsand (zero-shot sandwich)      | 0.3866    |
| **walft prod** (iter-10 cls champion config) | **0.3871** |

| λ      | FT mIoU | Δ vs ZS=0.3866 | Δ vs prod walft=0.3871 | Pass? |
|--------|---------|----------------|--------------------------|-------|
| 0.025  | 0.3819  | −0.47 pp       | −0.52 pp                 | no    |
| 0.05   | 0.3788  | −0.78 pp       | −0.83 pp                 | no    |
| 0.10*  | 0.3760  | −1.06 pp       | −1.11 pp                 | no    |
| 0.20   | 0.3681  | −1.85 pp       | −1.90 pp                 | no    |

*matches the FRAPPE-seg locked λ but at WaLLoC lr.*

**No passing λ.** All 4 cells fall below ZS, and all 4 fall below the
current production walft-seg p=36 (which uses LSDIR + pseudolabel +
**iter-10 cls champion's** λ=0.05 + lr_base=2e-5 — i.e., the cls
recipe applied to seg, not the FRAPPE seg recipe). The
"FRAPPE-aligned-recipe will lift WaLLoC seg" hypothesis is falsified at
p=36 across this λ grid.

**Why this matters for `recipes.md`:** the open-recipe note about a
data-axis drift (iter-7 ADE20K+GT → consolidated LSDIR+pseudolabel) is
not a missed opportunity — at this LR, the FRAPPE-aligned recipe is
**strictly worse** than the iter-10 cls-champion config that's already
in production at p=36. The consolidated `walloc_seg.py` recipe LOCKED
values (lr_base=2e-4, λ=0.10) inherited from iter-7 are even further
off; the current production walft-seg p=36 ckpt was not made by the
consolidated recipe and could not be reproduced from it without the
iter-10 patch (lr_base=2e-5 + λ=0.05 — i.e., the iter-10 cls recipe
applied to seg). This is an open data-hygiene issue, not a model
issue.

## Stage 5 (deferred)

Skipped per user — GPUs needed for other work. Original conditional
was: if any Stage-4 λ passed at p=36, extend that champion to
p ∈ {4, 16, 80, 100} (4 cells, ~4h on 4 GPUs). Two reasons not to
extend even when GPUs come back:

1. Stage 4 found no passing λ. The conditional itself is unmet — the
   FRAPPE-aligned recipe at lr=2e-5 doesn't beat ZS at p=36, so it
   would almost certainly fail at the other 4 ops too.
2. The 4 currently-failing walloc-seg ops (p ∈ {4, 16, 80, 100}) still
   use the iter-7 walft-seg recipe (ADE20K + GT + λ=0.10 + lr_base=2e-4).
   The correct first move is the iter-10-style refresh (champion's
   λ=0.05 + lr_base=2e-5 + ADE20K + GT, mirroring what iter-10 did
   for p=36) — not the FRAPPE-aligned recipe.

Logged as the natural follow-up; no `experiments/lambda_sweep/stage5_*/`
directory is created in this run.

## Implications for `recipes.md`

Apply when convenient (these change locked λ values, no other knobs):

| Recipe / op            | Current LOCKED λ | Proposed LOCKED λ | Source        |
|------------------------|------------------|--------------------|---------------|
| FRAPPE seg n=15        | 0.10 (recipe-wide) | 0.025 at n=15    | Stage 1       |
| WaLLoC clip p=100      | 0.05 (recipe-wide) | 0.025 at p=100   | Stage 3       |

Two caveats before updating:

- The `frappe_seg` and `walloc_clip` recipes set a single λ for the
  whole operating-point range. Lowering λ at the high-n_ch or high-p
  end may regress the low-channel/low-pixel-ratio cells. Confirm at
  the other ops before changing the recipe-wide locked value;
  alternatively, change the recipe to accept λ as a function of op
  (one λ per op).
- The Stage 2 outlier (FRAPPE clip n=3) is **not** closed; do not
  touch the `frappe_clip` locked λ. Open follow-up is an LR sweep at
  that single cell.

Stage 4 outcome leaves `walloc_seg` unchanged but documents that the
FRAPPE-aligned recipe at lr=2e-5 is not the lift.

## Open follow-ups (none kicked off here)

1. **FRAPPE clip n=3 LR sweep** at champion λ=0.025: try
   lr_base ∈ {3.5e-5, 1.75e-5, 7e-6} (×0.5, ×0.25, ×0.1 of locked).
   Only Stage 2 needs this; everything else is closed or deferred.
2. **WaLLoC seg p ∈ {4, 16, 80, 100} iter-10 refresh**: apply the
   iter-10 cls champion config (λ=0.05, lr_base=2e-5) to the iter-7
   ADE20K-GT data axis at the 4 remaining ops. This is **not** the
   Stage-5-deferred work — it's a different recipe-cell.
3. **Confirm the proposed `recipes.md` λ updates at adjacent ops**
   before changing recipe-wide LOCKED values. For FRAPPE seg n=15 win:
   re-eval seaft seg with λ=0.025 at n ∈ {12, 9, 6, 3} (no retraining;
   ckpts exist). Wait — these don't exist; do a single-cell smoke at
   n=12 before committing.

## Artifacts

```
experiments/lambda_sweep/
├── stage1_frappe_seg_n15/   4 ckpts (228 MB each) + 4 eval JSONs + 4 results JSONs
├── stage2_frappe_clip_n3/   4 ckpts + 4 eval JSONs + 4 results JSONs
├── stage3_walloc_clip_p100/ 4 ckpts + 4 eval JSONs + 4 results JSONs
├── stage4_walloc_seg_p36/   4 ckpts + 4 eval JSONs + 4 results JSONs
└── findings.md              (this file)
```

Total wall time: ~12.6h on 4 GPUs (Stages 1 + 4 are 4-epoch LSDIR runs
at ~5.5h each; Stages 2/3 are 1-epoch ImageNet runs at ~35min/~50min;
evals add ~1h per clip stage + ~10min per seg stage). Original prompt
estimated ~3-4h wall — the gap was the LSDIR 4-epoch step count
(170k steps/cell × 4 cells) being heavier than the prompt's nominal
"4 epochs × 4-cell" estimate, plus the ImageNet val being 50k samples
(not 5k as I initially assumed for naflex clip eval).

## Reproduction

Stage 1 (champion cell only):

```bash
python tools/lambda_sweep_cell.py \
    --codec frappe --task seg --op_value 15 \
    --lam 0.025 --lr_base 2e-4 \
    --batch_size 2 --epochs 4 \
    --train_ds danjacobellis/LSDIR \
    --device cuda:0 \
    --exp_name lam_frappe_seg_n15_lam0p025 \
    --out_dir experiments/lambda_sweep/stage1_frappe_seg_n15 \
    --num_workers 4 --seed 0
```

Stage 3 (champion cell only):

```bash
python tools/lambda_sweep_cell.py \
    --codec walloc --task clip --op_value 100 \
    --lam 0.025 --lr_base 7e-6 \
    --batch_size 1 --epochs 1 \
    --train_ds timm/imagenet-1k-wds --dataset_samples 50000 \
    --alpha_clip 0.7 \
    --device cuda:0 \
    --exp_name lam_walloc_clip_p100_lam0p025 \
    --out_dir experiments/lambda_sweep/stage3_walloc_clip_p100 \
    --num_workers 4 --seed 0
```

Eval (Stage-1 champion):

```bash
python pre_trained_convnext/experiments/iter6_extra_codec_baselines/eval_accuracy.py \
    --pipeline seaft --task seg \
    --op '{"type":"n_ch","value":15,"extras":{"checkpoint":"<abs ckpt path>"}}' \
    --device cuda:0 \
    --out_json experiments/lambda_sweep/stage1_frappe_seg_n15/eval_lam0p025.json
```
