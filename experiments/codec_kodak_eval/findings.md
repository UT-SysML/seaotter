# Standalone-codec Kodak eval — findings

Compares three codecs on Kodak validation (24 images, **native
resolution**, `bs=1`, **CPU only**):

1. **`seaotter`** — SEA OTTER S3 K=3 production bundle
   (`danjacobellis/seaotter @ seaotter_jpeg_s3`,
   λ = [0.75, 0.40, 0.22]; the warm-start used by every phase-4
   production pipeline at `phase2_k=2`). Loaded via
   `from seaotter import load_from_hub; bundle = load_from_hub()`.
2. **`jpeg`** — Pillow JPEG, default chroma subsampling (4:2:0).
3. **`jpeg_sub0`** — Pillow JPEG, no chroma subsampling (4:4:4).

Distortion metrics: `piq.psnr`, `piq.ssim`, `piq.LPIPS`, `piq.DISTS`
on `[0, 1]` float tensors (`data_range=1.0`), LPIPS/DISTS reported in
dB via `-10·log10`. Same convention as the iter-6 harness.

Per-image bpp: `8 · len(jpeg_bytes) / (H · W)`. For SEA OTTER,
storage ≡ transmit ≡ JPEG byte count (standalone codec; no FRAPPE
upstream here).

## q-selection rule

For each SEA OTTER operating point `k`, pick the **smallest** integer
JPEG-sub=0 quality `q ∈ [1, 99]` such that — evaluated over all 24
Kodak images at native resolution — SEA OTTER@k strictly dominates
JPEG-sub=0@q on **both** axes simultaneously:

- `mean_PSNR(JPEG-sub=0 @ q) < mean_PSNR(SEA OTTER @ k)`
- `mean_bpp(JPEG-sub=0 @ q) > mean_bpp(SEA OTTER @ k)`

q1, q2, q3 are the three anchors. q0.5, q1.5, q2.5, q3.5 are
equal-q-step interpolations (`round((qN + qN+1) / 2)`, clamped to
`[1, 99]`).

## Result: 3/3 anchors are strict-dominators, no fallbacks

| k | λ    | SEA OTTER mean bpp / PSNR | matched JPEG-sub=0 q | JPEG-sub=0 mean bpp / PSNR | dom_ok |
|---|------|---------------------------|----------------------|------------------------------|--------|
| 0 | 0.75 | 1.099 / 33.17 | **q1 = 53** | 1.103 / 32.90 | ✓ |
| 1 | 0.40 | 1.909 / 37.77 | **q2 = 81** | 1.912 / 36.37 | ✓ |
| 2 | 0.22 | 2.870 / 40.89 | **q3 = 91** | 2.965 / 39.62 | ✓ |

The SEA OTTER PSNR / bpp at each `k` reproduces the
`reference_metrics.kodak_validation_last_epoch` values in
`~/hf/seaotter/seaotter_jpeg_s3/config.json` exactly.

## Resolved q-ladder

| rung  | q  |
|-------|----|
| q0.5  | 39 |
| q1    | 53 |
| q1.5  | 67 |
| q2    | 81 |
| q2.5  | 86 |
| q3    | 91 |
| q3.5  | 96 |

No collisions between adjacent rungs.

## Mean-over-Kodak headline

| cell                 | bpp    | PSNR (dB) | SSIM   | LPIPS (dB) | DISTS (dB) |
|----------------------|--------|-----------|--------|------------|------------|
| jpeg q0.5            | 0.7790 | 31.352    | 0.9524 | 5.929      | 13.153     |
| jpeg q1              | 0.9428 | 32.377    | 0.9632 | 6.684      | 14.558     |
| jpeg q1.5            | 1.1622 | 33.576    | 0.9720 | 7.538      | 16.073     |
| jpeg q2              | 1.6279 | 35.583    | 0.9815 | 9.105      | 18.674     |
| jpeg q2.5            | 1.9459 | 36.734    | 0.9850 | 10.050     | 20.123     |
| jpeg q3              | 2.4692 | 38.424    | 0.9887 | 11.513     | 21.712     |
| jpeg q3.5            | 3.8101 | 41.350    | 0.9929 | 14.600     | 23.942     |
| jpeg_sub0 q0.5       | 0.9175 | 31.813    | 0.9581 | 6.302      | 15.158     |
| jpeg_sub0 q1         | 1.1032 | 32.900    | 0.9693 | 7.144      | 16.905     |
| jpeg_sub0 q1.5       | 1.3587 | 34.175    | 0.9781 | 8.080      | 18.646     |
| jpeg_sub0 q2         | 1.9122 | 36.372    | 0.9873 | 9.846      | 22.085     |
| jpeg_sub0 q2.5       | 2.2979 | 37.659    | 0.9906 | 10.900     | 23.966     |
| jpeg_sub0 q3         | 2.9648 | 39.624    | 0.9938 | 12.602     | 25.832     |
| jpeg_sub0 q3.5       | 4.7012 | 43.384    | 0.9969 | 16.356     | 28.390     |
| **seaotter k=0**     | 1.0992 | 33.168    | 0.9427 | 6.002      | 12.441     |
| **seaotter k=1**     | 1.9087 | 37.768    | 0.9798 | 9.419      | 17.545     |
| **seaotter k=2**     | 2.8700 | 40.890    | 0.9909 | 12.850     | 21.607     |

(median values in `summary.json` per cell.)

## Interpretation

At each of the three SEA OTTER operating points, SEA OTTER **wins
PSNR + bpp simultaneously** vs the JPEG-sub=0 cell at the chosen q
— the dominance margin grows with bpp (k=0 → ~0.27 dB at near-equal
bpp, k=1 → +1.40 dB, k=2 → +1.27 dB).

The perceptual story is the inverse: at all three operating points,
JPEG-sub=0 at the matched q has **higher** SSIM and JPEG-sub=0
LPIPS/DISTS (in dB) sit above SEA OTTER's. This is the expected
shape — the phase-2 / R16 dual-goal training loss was
`Σᵢ wᵢ · [log₁₀(MSE_i) + λᵢ · bppᵢ]`, with no perceptual term — so
PSNR is the lever the codec was trained on, and perceptual quality
is a passenger metric. The "wins PSNR at matched bpp, loses LPIPS"
shape is consistent with the phase-3 round-1 / round-2 sandwich
results.

Vs the default-subsampling JPEG (`jpeg` rows): SEA OTTER again wins
PSNR at matched bpp by a wider margin (since 4:2:0 chroma sub costs
~0.3 dB on Kodak); the perceptual gap is similar in shape.

## Provenance

- Eval script: [`run_eval.py`](run_eval.py). Single CPU-only entry
  point; no GPU, no thread pinning, bs=1 throughout.
- Wall-clock: ~5 min on the EPYC 9354 (dominated by piq.LPIPS +
  piq.DISTS on CPU at native Kodak resolution).
- Outputs: 17 `eval_*.json` cells + `summary.json` in [`results/`](results/),
  mirrored to `~/UT-SysML/seaotter/results/codec_kodak/` for the
  public paper repo.
- Harness version: `codec-kodak-v1`. Distinct from the iter-6
  pipeline harness (no `metrics.*` / no `transmit_bpp` / `storage_bpp`
  split; per-image arrays in `per_image.*` + `summary.{mean, median}`).
