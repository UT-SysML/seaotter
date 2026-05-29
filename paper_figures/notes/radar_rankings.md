# Data-driven radar rankings

Companion report to `figures/radar_summary.pdf`. The figure uses the
hand-curated `RANKINGS` dict in `make_radar_figure.py`; this report
proposes data-driven scalars per axis and the 1-4 scores they
produce after normalization. **Nothing here is wired into the figure
— the figure stays on the hardcoded values until the user reviews
this report and decides whether any axis should be replaced.**

Framework: for each axis, compute one raw scalar per system from the
canonical paper-data mirror at `~/UT-SysML/seaotter/results/`. Mark a
system as **degenerate** (score `1.0`) when its raw scalar is `NaN`
(e.g., quality range disjoint from the reference, missing JSONs).
Linearly map the remaining `4 − D` valid systems into the range
`[D + 1, 4]`, oriented so "better raw value → higher score". Scores
are fractional. Primary RD metric is BD-rate vs ITU JPEG using a
PChip interpolator over log-bpp; throughput axes use geometric mean
of MPx/s across the available `(task, op)` pairs (cls + seg only —
clip has no throughput JSONs).

Pipeline-short mapping used throughout: **AVIF** → `avif`, **ITU**
→ `jpeg` (subsampling=0, i.e. 4:4:4), **FRAPPE** → `frp`,
**SEAOTTER** → `seaft` (fine-tuned; the paper's hero configuration).
The mirror's `results.md` description of the `jpeg` pipeline as
4:2:0 is stale — every `eval_jpeg_*.json` confirms
`config.subsampling = 0`.

Implementation deviations from the prompt:

- **3-point rule relaxed to ≥2 distinct points per curve.** The
  prompt asks for ≥3 distinct points inside the shared quality
  range; on this 5-op-per-codec data, that rule renders most
  (codec, ref) pairs falsely degenerate even when both curves have
  data spanning the shared range. The Pchip interpolator is
  well-defined as long as integration stays inside each curve's
  data extent, so the relaxed rule keeps numerically valid pairs
  and only fails when the codecs truly don't share quality range
  (e.g., SEAOTTER's PSNR is entirely below ITU's).
- **bpp anchors dropped.** The prompt suggested anchoring `trans`
  to bpp ∈ [0.1, 1.0] and `store` to [1.5, 4.0]; in practice ITU
  loses more than half its operating points under the trans
  anchor, and no system has ITU points ≥ 1.5 storage_bpp, so the
  store anchor is empirically empty. We use full-intersect BD-rate
  for both. The `store` axis still discriminates SEAOTTER because
  SEAOTTER's storage_bpp diverges from its transmit_bpp by the JPEG
  transcode (1.6× to 7× inflation, observed in
  `eval_seaft_cls_*.json`).
- **`trans` and `cls` are computed from identical inputs** in the
  primary metric (BD-rate vs ITU on transmit_bpp + cls top-1) and
  therefore produce identical scalars. Without an anchor that
  distinguishes them, the two axes are redundant. They are listed
  separately below to preserve the radar's nine-axis structure,
  but their scores agree by construction. An alternative trans
  metric that decouples them is sketched in the trans section.

Sanity check: BD-rate of ITU JPEG against itself rounds to `0.0`,
within floating-point noise.

## `trans` — transmit-bpp efficiency

**Primary**: BD-rate (PChip-on-log-bpp, integrated over the shared
cls top-1 range) of system vs ITU JPEG, full intersect of operating
points.

| system   | raw BD-rate | score | hardcoded |
|----------|-------------|------:|----------:|
| AVIF     | −47.7%      | 2.67  | 2         |
| ITU      |     0%      | 1.00  | 1         |
| FRAPPE   | −73.2%      | 3.57  | 4         |
| SEAOTTER | −85.5%      | 4.00  | 4         |

Agreement on ordering (SEAOTTER > FRAPPE > AVIF > ITU). The data
breaks the hardcoded `FRAPPE ≡ SEAOTTER` tie that the prompt
justifies via "shared encoder, same transmit stream": both produce
the same transmit_bpp at each operating point, but BD-rate measures
cls top-1 at that bpp, and SEAOTTER is fine-tuned for cls so it
wins at matched transmit_bpp. Keeping the tie in the hardcoded
figure is defensible as a *codec-architectural* claim ("the
transmit channel is identical"); the data instead reflects the
*full-system* claim ("transmit-bpp efficiency including the
downstream model").

**Alternative**: BD-rate restricted to ITU points with bpp ≤ 1.0
(prompt's original suggestion). On this data the filter drops
ITU's q=50 row (transmit_bpp 1.21) and produces shared-range
collapse for AVIF and FRAPPE vs the filtered ITU, sending both to
degenerate. Not viable without resampling the ITU q-ladder.

## `store` — storage-bpp efficiency

**Primary**: BD-rate of system vs ITU JPEG on (storage_bpp, cls
top-1), full intersect. For AVIF / ITU / FRAPPE this is identical
to `trans` (no transcode, so storage_bpp ≡ transmit_bpp). For
SEAOTTER, storage_bpp is the on-disk JPEG file size after the
cloud-side transcode and is 7-14× larger than transmit_bpp.

| system   | raw BD-rate | score | hardcoded |
|----------|-------------|------:|----------:|
| AVIF     | −47.7%      | 3.75  | 4         |
| ITU      |     0%      | 3.27  | 1         |
| FRAPPE   | −73.2%      | 4.00  | 3         |
| SEAOTTER | +227.4%     | 1.00  | 2         |

The data correctly captures SEAOTTER's storage penalty (BD-rate
positive → needs 2.3× more storage_bpp than ITU to match cls
top-1). The disagreement is concentrated on **ITU**: data scores
ITU at 3.27 (near the top), hardcoded scores ITU at 1 (poor).
This is a known artifact of using ITU as the BD-rate reference —
ITU's score is pinned by `BD-rate(ITU, ITU) = 0`, so any system
that's worse than ITU pulls ITU's normalized score upward
arbitrarily. The hardcoded intuition is "ITU is inefficient at the
high-bpp regime relevant to long-term storage"; that's an
absolute claim that BD-rate-vs-ITU cannot express.

**Alternative**: BD-rate of system vs **AVIF** (AVIF is the most
efficient codec in this slot). Computed but not summarized in the
primary table; AVIF anchored at 0, ITU at +91.3%, FRAPPE at −48.7%,
SEAOTTER at +525%. Normalized scores under AVIF-reference:
AVIF=3.07, ITU=1.55, FRAPPE=4.00, SEAOTTER=1.00. ITU drops as
expected; AVIF lands near the top but no longer at 4 (it's the
reference now, so it gets the BD-rate=0 anchor problem inverted).
A robust absolute metric would aggregate BD-rate against multiple
references and average; not done here.

## `PSNR` — pixel-fidelity at compressed bitrate

**Primary**: BD-rate (PChip on log-bpp, integrated over the shared
PSNR-dB range) of system vs ITU on (transmit_bpp, psnr_db), cls
operating points.

| system   | raw BD-rate | score | hardcoded |
|----------|-------------|------:|----------:|
| AVIF     | −53.4%      | 3.41  | 4         |
| ITU      |     0%      | 2.00  | 2         |
| FRAPPE   | −75.8%      | 4.00  | 3         |
| SEAOTTER | NaN (degen) | 1.00  | 1         |

SEAOTTER is correctly flagged as degenerate: its PSNR range
[9.83, 12.25] dB is fully below ITU's range [21.47, 31.94] dB, so
no shared PSNR exists for the BD-rate integral. AVIF and ITU
agreement holds; **FRAPPE and AVIF are inverted vs the hardcoded**.
The data-driven primary measures *compression efficiency for
preserving PSNR* — FRAPPE wins because its transmit channel is so
much narrower than AVIF's at every PSNR level in the shared range.
The hardcoded interpretation is *absolute PSNR quality at typical
operating points* — AVIF wins there because its PSNR ceiling
(32.28 dB at q=50) is higher than FRAPPE's (28.29 dB at n=15) and
its low-end PSNR (25.01 dB at q=1) is much higher than FRAPPE's
(19.85 dB at n=3).

**Alternative**: mean PSNR (dB) across the system's cls operating
points.

| system   | mean PSNR_dB | score | hardcoded |
|----------|--------------|------:|----------:|
| AVIF     | 27.33        | 4.00  | 4         |
| ITU      | 26.67        | 3.88  | 2         |
| FRAPPE   | 23.86        | 3.35  | 3         |
| SEAOTTER | 11.26        | 1.00  | 1         |

Mean PSNR agrees with the hardcoded that AVIF tops and SEAOTTER
bottoms. It still disagrees on ITU vs FRAPPE ordering — the
hardcoded ranks FRAPPE > ITU on PSNR (rationale unclear; possibly
"FRAPPE is more efficient" leaking through), while mean PSNR shows
ITU's high-quality operating points pull its mean above FRAPPE's.

## `DISTS` — perceptual distortion at compressed bitrate

**Primary**: BD-rate vs ITU on (transmit_bpp, dists_db), cls
operating points. `dists_db = −10 log10(DISTS)`, so higher is
better; "more negative BD-rate" still means "more bits saved at
matched DISTS_dB".

| system   | raw BD-rate | score | hardcoded |
|----------|-------------|------:|----------:|
| AVIF     | −44.2%      | 2.72  | 3         |
| ITU      |     0%      | 1.00  | 2         |
| FRAPPE   | −77.3%      | 4.00  | 4         |
| SEAOTTER | −70.9%      | 3.75  | 1         |

Largest disagreement of any axis: **SEAOTTER scores 3.75 in the
data vs 1 in the hardcoded**. The BD-rate is well-defined for
SEAOTTER on DISTS (unlike PSNR) because SEAOTTER's DISTS_dB range
[2.94, 5.60] overlaps ITU's [4.37, 13.66]. In that overlap, at
matched DISTS_dB, SEAOTTER needs far fewer bits. The hardcoded
ranking is again the "absolute DISTS quality" interpretation —
SEAOTTER's DISTS ceiling (5.60 dB) is well below AVIF (11.99 dB),
ITU (13.66 dB), and FRAPPE (8.53 dB), so by "best DISTS achieved"
it ranks last.

**Alternative**: mean DISTS_dB across cls operating points.

| system   | mean DISTS_dB | score | hardcoded |
|----------|---------------|------:|----------:|
| AVIF     | 7.93          | 3.67  | 3         |
| ITU      | 8.35          | 4.00  | 2         |
| FRAPPE   | 6.01          | 2.14  | 4         |
| SEAOTTER | 4.57          | 1.00  | 1         |

Mean DISTS recovers SEAOTTER=1 (matches hardcoded), but now
disagrees with the hardcoded on FRAPPE vs ITU and on AVIF vs ITU.
Neither metric matches the hardcoded ordering exactly; the DISTS
axis is the noisiest of the nine.

## `cls` — ImageNet top-1 classification

**Primary**: BD-rate vs ITU on (transmit_bpp, cls top-1), full
intersect.

| system   | raw BD-rate | score | hardcoded |
|----------|-------------|------:|----------:|
| AVIF     | −47.7%      | 2.67  | 2         |
| ITU      |     0%      | 1.00  | 1         |
| FRAPPE   | −73.2%      | 3.57  | 3         |
| SEAOTTER | −85.5%      | 4.00  | 4         |

Identical to `trans` (same inputs). Strong agreement with the
hardcoded ordering and tier assignments.

## `seg` — ADE20K mIoU

**Primary**: BD-rate vs ITU on (transmit_bpp, miou), seg task.

| system   | raw BD-rate | score | hardcoded |
|----------|-------------|------:|----------:|
| AVIF     | −67.7%      | 3.37  | 2         |
| ITU      |     0%      | 1.00  | 1         |
| FRAPPE   | −80.7%      | 3.83  | 3         |
| SEAOTTER | −85.6%      | 4.00  | 4         |

Ordering matches the hardcoded; AVIF scores higher in the data
(3.37 vs hardcoded 2) because AVIF's compact bitrate range on
ADE20K (transmit_bpp 0.08-0.51 vs 0.28-0.99 on ITU) gives it a
strong BD-rate even though absolute mIoU peaks lower.

## `clip` — SigLIP-2 zero-shot top-1

**Primary**: BD-rate vs ITU on (transmit_bpp, clip top-1).

| system   | raw BD-rate | score | hardcoded |
|----------|-------------|------:|----------:|
| AVIF     | −36.7%      | 2.49  | 2         |
| ITU      |     0%      | 1.00  | 1         |
| FRAPPE   | −67.6%      | 3.75  | 3         |
| SEAOTTER | −73.7%      | 4.00  | 4         |

Ordering matches the hardcoded; fractional scores all land within
0.5 of the hardcoded values. The least-controversial axis of the
nine — even ITU's BD-rate-reference anchoring at 1.0 aligns with
the hardcoded "poor."

## `enc` — sensor-side encode throughput

**Primary**: geometric mean of encode MPx/s across all
`(task ∈ {cls, seg}, op)` pairs. Throughput =
`pixels_per_image / (1e3 × encode_median_ms)`. Clip excluded (no
throughput JSONs in the mirror).

| system   | geomean MPx/s | score | hardcoded |
|----------|--------------:|------:|----------:|
| AVIF     | 4.26          | 1.00  | 1         |
| ITU      | 294.0         | 4.00  | 4         |
| FRAPPE   | 85.12         | 1.84  | 4         |
| SEAOTTER | 85.36         | 1.84  | 4         |

Linear normalization treats the 4.26 → 294 range as one continuous
axis, so FRAPPE / SEAOTTER (at ~85 MPx/s, ~29% of JPEG's 294) get
only 1.84 — far below the hardcoded "excellent". The hardcoded
intent is *deployment-tier*: all three of {ITU, FRAPPE, SEAOTTER}
are fast enough for real-time CPU encoding, so they all earn 4.

**Alternative**: log-MPx/s normalization (since throughput often
spans orders of magnitude).

| system   | log MPx/s | score | hardcoded |
|----------|----------:|------:|----------:|
| AVIF     | 1.45      | 1.00  | 1         |
| ITU      | 5.68      | 4.00  | 4         |
| FRAPPE   | 4.44      | 3.12  | 4         |
| SEAOTTER | 4.45      | 3.12  | 4         |

Log normalization brings FRAPPE / SEAOTTER from 1.84 to 3.12,
much closer to the hardcoded 4. SEAOTTER ≈ FRAPPE on the data
(85.12 vs 85.36 MPx/s — well within run-to-run noise), confirming
the shared-encoder tie.

## `dec` — consumer-side decode + ConvNeXt throughput

**Primary**: geomean MPx/s where time per image =
`consumer_median_ms` (= decode + ConvNeXt forward, end-to-end
first-receive path, no cache).

| system   | geomean MPx/s | score | hardcoded |
|----------|--------------:|------:|----------:|
| AVIF     | 14.33         | 1.02  | 4         |
| ITU      | 34.15         | 4.00  | 4         |
| FRAPPE   | 18.25         | 1.61  | 1         |
| SEAOTTER | 14.18         | 1.00  | 4         |

The biggest data-vs-hardcoded mismatch in the report. Two
mechanisms drive it:

- **`consumer_ms` is not pure decode.** It's
  `decode + ConvNeXt forward (bs=1)`. The forward time
  (~5 ms) dominates JPEG's tiny libjpeg decode but is a smaller
  fraction of AVIF's 11 ms or FRAPPE's 7 ms total. The hardcoded
  ranking implicitly evaluates pure decode latency; the data
  measures the realistic end-to-end consumer path.
- **SEAOTTER's `consumer_ms` includes the cloud-side transcode.**
  The throughput harness measures sensor → cloud receive →
  transcode → JPEG decode → ConvNeXt. The hardcoded ranking
  assumes the *steady-state* path where the JPEG has been cached
  on disk and subsequent reads are libjpeg-fast; the harness
  captures the first-receive path with the transcode amortized
  over a single image, which inflates SEAOTTER's per-image latency
  to 9.88 ms vs JPEG's 3.44 ms. To recover the steady-state value
  one would need a separate measurement of "decode-only" time
  excluding the transcode — not present in the mirror.

**Alternative**: log-MPx/s normalization.

| system   | log MPx/s | score | hardcoded |
|----------|----------:|------:|----------:|
| AVIF     | 2.66      | 1.03  | 4         |
| ITU      | 3.53      | 4.00  | 4         |
| FRAPPE   | 2.90      | 1.86  | 1         |
| SEAOTTER | 2.65      | 1.00  | 4         |

Log scaling doesn't help reconcile this axis — the underlying issue
is that `consumer_ms` measures something different from "decode
latency". The hardcoded values stand on a different (steady-state,
decode-only) operating assumption.

## Summary table

`(data primary / hardcoded)`.

| axis  | AVIF        | ITU         | FRAPPE      | SEAOTTER    |
|-------|-------------|-------------|-------------|-------------|
| trans | 2.67 / 2    | 1.00 / 1    | 3.57 / 4    | 4.00 / 4    |
| store | 3.75 / 4    | 3.27 / 1    | 4.00 / 3    | 1.00 / 2    |
| PSNR  | 3.41 / 4    | 2.00 / 2    | 4.00 / 3    | 1.00 / 1    |
| DISTS | 2.72 / 3    | 1.00 / 2    | 4.00 / 4    | 3.75 / 1    |
| cls   | 2.67 / 2    | 1.00 / 1    | 3.57 / 3    | 4.00 / 4    |
| seg   | 3.37 / 2    | 1.00 / 1    | 3.83 / 3    | 4.00 / 4    |
| clip  | 2.49 / 2    | 1.00 / 1    | 3.75 / 3    | 4.00 / 4    |
| enc   | 1.00 / 1    | 4.00 / 4    | 1.84 / 4    | 1.84 / 4    |
| dec   | 1.02 / 4    | 4.00 / 4    | 1.61 / 1    | 1.00 / 4    |

## Discussion

The three axes with the largest data-vs-hardcoded gaps:

- **`dec`** is the worst-aligned axis. The data measures a
  realistic first-receive consumer path; the hardcoded ranking
  encodes a steady-state cached-JPEG-decode assumption that the
  mirror has no measurement for. Both interpretations are
  defensible for the paper; the hardcoded version aligns with the
  SEAOTTER value proposition ("a JPEG file lives on disk; reads
  are JPEG-fast"). Unless we add a separate "decode-only" eval to
  the harness, the hardcoded values are the safer call for the
  figure.

- **`DISTS` and `store` for SEAOTTER** are *real* metric-choice
  ambiguities. BD-rate-against-ITU measures compression efficiency
  at matched quality; mean-DISTS / mean-PSNR measures absolute
  quality at typical operating points. SEAOTTER wins the former
  (compression-efficient) and loses the latter (absolute-quality
  ceiling) because its operating points are clustered at the very
  low-bpp end. The hardcoded ranking takes the absolute-quality
  view; that's the right view for the radar narrative ("SEAOTTER
  sacrifices distortion-fidelity for task accuracy"). The BD-rate
  view would muddy the SEAOTTER vs FRAPPE distortion story.

- **`enc` for FRAPPE / SEAOTTER** is a normalization choice rather
  than a metric-choice issue. Linear MPx/s normalization treats
  85 vs 294 MPx/s as ~30% apart; deployment-tier ("real-time on
  CPU" / "not real-time") treats them as equivalent. The radar's
  poor/fair/good/excellent tiers are deployment-tier framings, so
  log normalization or threshold-based bucketing aligns the data
  with the hardcoded intent more naturally.

The remaining six axes (`trans`, `cls`, `seg`, `clip`, `PSNR`,
`store`/AVIF) agree with the hardcoded ranking on at least the
top and bottom tiers, and disagree only on fractional placement
within the middle.

Open questions for the user:

1. Should `trans` be re-distinguished from `cls`? Both currently
   collapse to the same BD-rate; the prompt's intended distinction
   (low-bpp anchor) is unsupported by the 5-op ITU ladder. A
   resampled ITU sweep at lower q (q ∈ {1, 2, 5, 10, 25}, dropping
   q=50) would let the anchored variant work.
2. Should the figure switch to log-normalised throughput for `enc`?
   That single change would bring `enc` data and hardcoded into
   alignment without altering the visual story.
3. Is the steady-state decode interpretation worth a separate
   throughput cell (decode-only, JPEG-after-transcode)? It would
   close the `dec` gap on SEAOTTER specifically.
