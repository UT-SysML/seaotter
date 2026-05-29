# Data-driven radar rankings — Table 1 matched-rate (v3)

Companion to `figures/radar_summary.pdf`. Re-derives each radar axis
from a **single principled scalar per system**, computed from the
matched-rate operating points shown in Table 1 (each codec at the op
closest to FRAPPE $n_\text{ch}{=}12$'s transmit-bpp on that
subsection's dataset) plus the matching eval-JSON distortion +
throughput numbers in `~/UT-SysML/seaotter/results/`.

Differs from `radar_rankings.md` (v1, BD-rate-based, full-curve) and
from `radar_rankings_v2.md` (audit of hardcoded values against
Table 1, hand-curated scores). This file proposes integer scores
derived mechanically from a fixed normalization rule, with no
hand-curation. The current hardcoded radar (`RANKINGS` in
`make_radar_figure.py`, as updated from v2) is shown alongside each
score for comparison.

## Scoring rule (applied to every axis the same way)

For each axis:

1. Pick **one raw scalar per system** from the per-task matched-rate
   data. For axes that aggregate across cls / seg / clip (trans,
   store, PSNR, DISTS) the scalar is the **arithmetic mean** across
   the three tasks. For task-specific axes (cls, seg, clip) the
   scalar is the task's accuracy at the matched-rate pick. For
   throughput axes (enc, dec) the scalar is the cls-task MPx/s,
   which is task-invariant for the same op.
2. Choose orientation so "higher raw scalar → better axis score."
   For CR / accuracy / PSNR / DISTS_dB / MPx/s, higher is naturally
   better.
3. **Linear-normalize** the five-system raw range onto `[1, 5]`:
   `score(x) = 1 + (x − x_min) / (x_max − x_min) · 4`. Linear is
   the default; throughput axes (enc, dec) span 3+ orders of
   magnitude and are instead **log-normalized** in the same range
   to avoid AVIF/FRAPPE getting compressed against ITU's libjpeg.
4. Round to the nearest integer for the radar grid.

This rule is deliberately blunt — it gives the radar a reproducible
numerical foundation. Some sharp judgement calls in the previous v2
audit (e.g., "AVIF on seg is competitive enough to deserve a 4")
fall out of the rule automatically; others don't, and the
divergences are discussed at the end.

## Raw inputs (Table 1 + JSONs)

For convenience, the five systems' matched-rate values
(see Table 1 / `eval_jpeg_*_q1.json` for ITU):

| system       | cls bpp | cls CR | cls top-1 | cls PSNR | cls DISTS_dB |
|--------------|--------:|-------:|----------:|---------:|-------------:|
| AVIF (q=1)   | 0.146   | 165    | 61.15     | 25.01    | 6.23         |
| ITU (q=1)    | 0.298   | 80     | 25.75     | 21.47    | 4.37         |
| FRAPPE (n=12)| 0.109   | 221    | 56.22     | 25.08    | 6.56         |
| SEAOTTER-ZS  | 0.109   | 221    | 60.25     | 25.04    | 6.70         |
| SEAOTTER-FT  | 0.109   | 221    | 69.02     | 10.39    | 5.08         |

| system       | seg bpp | seg CR | seg mIoU  | seg PSNR | seg DISTS_dB |
|--------------|--------:|-------:|----------:|---------:|-------------:|
| AVIF (q=5)   | 0.086   | 279    | 32.75     | 26.97    | 7.38         |
| ITU (q=1)    | 0.276   | 87     | 11.59     | 22.09    | 5.02         |
| FRAPPE       | 0.094   | 256    | 29.09     | 26.81    | 7.49         |
| SEAOTTER-ZS  | 0.094   | 256    | 30.09     | 26.78    | 7.71         |
| SEAOTTER-FT  | 0.094   | 256    | 32.77     | 10.76    | 6.19         |

| system       | clip bpp | clip CR | clip top-1 | clip PSNR | clip DISTS_dB |
|--------------|---------:|--------:|-----------:|----------:|--------------:|
| AVIF (q=1)   | 0.250    | 96      | 42.59      | 24.35     | 5.46          |
| ITU (q=1)    | 0.348    | 69      | 18.66      | 21.08     | 3.84          |
| FRAPPE       | 0.142    | 169     | 41.51      | 24.37     | 5.66          |
| SEAOTTER-ZS  | 0.142    | 169     | 43.34      | 24.32     | 5.80          |
| SEAOTTER-FT  | 0.142    | 169     | 48.22      | 13.07     | 4.18          |

| system       | storage CR (cls / seg / clip) | enc MPx/s | dec MPx/s |
|--------------|------------------------------:|----------:|----------:|
| AVIF         | 165 / 279 / 96                | 5.16      | 20.75     |
| ITU          | 80 / 87 / 69                  | 303       | 192       |
| FRAPPE       | 221 / 256 / 169               | 38.20     | 0.61      |
| SEAOTTER-ZS  | 19 / 20 / 16                  | 38.20*    | 56.83     |
| SEAOTTER-FT  | 27 / 46 / 37                  | 38.20*    | 57.90     |

(*: SEAOTTER-{ZS,FT} encode TP canonicalized to FRAPPE n=12's value;
same physical encoder.)

---

## `trans` — transmit-bpp efficiency

Scalar: arithmetic mean of transmit CR across cls / seg / clip.

| system       | mean tx CR | score | hardcoded |
|--------------|-----------:|------:|----------:|
| AVIF         | 180.0      | 3.97  | 2         |
| ITU          | 78.7       | 1.00  | 1         |
| FRAPPE       | 215.3      | 5.00  | 3         |
| SEAOTTER-ZS  | 215.3      | 5.00  | 4         |
| SEAOTTER-FT  | 215.3      | 5.00  | 5         |

**Integer scores**: AVIF = 4, ITU = 1, FRAPPE = 5, SEAOTTER-ZS = 5,
SEAOTTER-FT = 5.

**Disagreement**: The data-driven scalar collapses the FRAPPE family
to a flat 5 because they emit identical sensor-uplink bytes at
$n_\text{ch}{=}12$. The hardcoded radar separates them (3/4/5) to
encode the downstream-accuracy-at-matched-bpp story, which is a
*different* axis from raw rate efficiency. AVIF moves up from 2 to 4
because its mean CR (180) is much closer to the FRAPPE-family ceiling
(215) than to ITU (79).

---

## `store` — storage-bpp efficiency

Scalar: arithmetic mean of storage CR across cls / seg / clip.

| system       | mean st CR | score | hardcoded |
|--------------|-----------:|------:|----------:|
| AVIF         | 180.0      | 4.29  | 4         |
| ITU          | 78.7       | 2.23  | 1         |
| FRAPPE       | 215.3      | 5.00  | 5         |
| SEAOTTER-ZS  | 18.3       | 1.00  | 2         |
| SEAOTTER-FT  | 36.7       | 1.37  | 3         |

**Integer scores**: AVIF = 4, ITU = 2, FRAPPE = 5, SEAOTTER-ZS = 1,
SEAOTTER-FT = 1.

**Disagreement**: data-driven is harsher on SEAOTTER than the
hardcoded v2 — both variants get pinned at 1 because their storage CR
(16–46) is so far below the others (69–280) that linear normalization
collapses them to the floor. The v2 hand-curation softened this to
2/3 to leave room above the floor for SEAOTTER-FT vs ZS distinction
("FT has slightly less-bad storage because the downstream accuracy
amortizes the transcode"). The data alone doesn't support that
distinction on this axis.

---

## `PSNR` — pixel fidelity

Scalar: arithmetic mean of PSNR_dB across cls / seg / clip.

| system       | mean PSNR_dB | score | hardcoded |
|--------------|-------------:|------:|----------:|
| AVIF         | 25.44        | 5.00  | 4         |
| ITU          | 21.55        | 3.89  | 2         |
| FRAPPE       | 25.42        | 4.99  | 4         |
| SEAOTTER-ZS  | 25.38        | 4.98  | 4         |
| SEAOTTER-FT  | 11.41        | 1.00  | 1         |

**Integer scores**: AVIF = 5, ITU = 4, FRAPPE = 5, SEAOTTER-ZS = 5,
SEAOTTER-FT = 1.

**Disagreement**: data-driven puts AVIF, FRAPPE, SEAOTTER-ZS in a
near-perfect three-way tie at the top (their means are within 0.06 dB
of each other), which is what the data actually says. The hardcoded
v2 keeps them all at 4 (a tier below SEAOTTER-FT's hardcoded 1) but
the linear-normalized data wants 5. ITU lifts from 2 to 4 — ITU at
its lowest-q is ~3.5 dB below the codec cluster, which the linear
normalization treats as "good," but absolute PSNR is mediocre.

---

## `DISTS` — perceptual distortion

Scalar: arithmetic mean of DISTS_dB across cls / seg / clip.

| system       | mean DISTS_dB | score | hardcoded |
|--------------|--------------:|------:|----------:|
| AVIF         | 6.36          | 4.35  | 4         |
| ITU          | 4.41          | 1.00  | 1         |
| FRAPPE       | 6.57          | 4.71  | 5         |
| SEAOTTER-ZS  | 6.74          | 5.00  | 5         |
| SEAOTTER-FT  | 5.15          | 2.27  | 3         |

**Integer scores**: AVIF = 4, ITU = 1, FRAPPE = 5, SEAOTTER-ZS = 5,
SEAOTTER-FT = 2.

Very close agreement with hardcoded v2 (1-unit difference only on
SEAOTTER-FT: 2 vs 3). The data-driven rule keeps SEAOTTER-FT above ITU
but more clearly mid-pack than the v2's "3 — fair."

---

## `cls` — ImageNet Top-1

Scalar: cls top-1 (%) at the matched-rate pick.

| system       | top-1 (%) | score | hardcoded |
|--------------|----------:|------:|----------:|
| AVIF         | 61.15     | 4.27  | 3         |
| ITU          | 25.75     | 1.00  | 1         |
| FRAPPE       | 56.22     | 3.82  | 2         |
| SEAOTTER-ZS  | 60.25     | 4.19  | 4         |
| SEAOTTER-FT  | 69.02     | 5.00  | 5         |

**Integer scores**: AVIF = 4, ITU = 1, FRAPPE = 4, SEAOTTER-ZS = 4,
SEAOTTER-FT = 5.

**Disagreement**: linear normalization compresses AVIF / FRAPPE /
SEAOTTER-ZS into the same integer bucket (4). The hardcoded v2
splits them (3/2/4) by giving more weight to the rate context (AVIF
uses 35 % more bits than FRAPPE-family at this op). The data alone
treats absolute top-1 as the axis, and the three are within 5 pp.

---

## `seg` — ADE20K mIoU

Scalar: seg mIoU (%) at the matched-rate pick.

| system       | mIoU (%) | score | hardcoded |
|--------------|---------:|------:|----------:|
| AVIF (q=5)   | 32.75    | 5.00  | 4         |
| ITU          | 11.59    | 1.00  | 1         |
| FRAPPE       | 29.09    | 4.30  | 2         |
| SEAOTTER-ZS  | 30.09    | 4.49  | 3         |
| SEAOTTER-FT  | 32.77    | 5.00  | 4         |

**Integer scores**: AVIF = 5, ITU = 1, FRAPPE = 4, SEAOTTER-ZS = 4,
SEAOTTER-FT = 5.

**Disagreement**: data-driven puts AVIF tied with SEAOTTER-FT at the
top (5 vs 5; the raw mIoUs differ by 0.02 pp). The hardcoded v2 had
AVIF/SEAOTTER-FT tied at 4. FRAPPE alone climbs to 4 in the
data-driven version — it's only 3.5 pp behind SEAOTTER-FT, which
linear normalization treats as "good."

---

## `clip` — SigLIP-2 zero-shot Top-1

Scalar: clip top-1 (%) at the matched-rate pick.

| system       | top-1 (%) | score | hardcoded |
|--------------|----------:|------:|----------:|
| AVIF         | 42.59     | 4.24  | 2         |
| ITU          | 18.66     | 1.00  | 1         |
| FRAPPE       | 41.51     | 4.09  | 2         |
| SEAOTTER-ZS  | 43.34     | 4.34  | 4         |
| SEAOTTER-FT  | 48.22     | 5.00  | 5         |

**Integer scores**: AVIF = 4, ITU = 1, FRAPPE = 4, SEAOTTER-ZS = 4,
SEAOTTER-FT = 5.

**Disagreement**: similar shape to cls — AVIF/FRAPPE/SEAOTTER-ZS
cluster within 2 pp and land in the same integer bucket. The
hardcoded v2 separated them by rate context (AVIF uses 76 % more
bits, FRAPPE/SEAOTTER share encoder), which the data-driven scalar
doesn't capture.

---

## `enc` — sensor-side encode throughput (log scale)

Scalar: cls-task encode MPx/s. Range spans 5.2 to 303 (two orders
of magnitude), so log normalization replaces linear.

| system       | enc MPx/s | log₁₀ | score | hardcoded |
|--------------|----------:|------:|------:|----------:|
| AVIF         | 5.16      | 0.71  | 1.00  | 1         |
| ITU          | 303       | 2.48  | 5.00  | 5         |
| FRAPPE       | 38.20     | 1.58  | 2.97  | 4         |
| SEAOTTER-ZS  | 38.20     | 1.58  | 2.97  | 4         |
| SEAOTTER-FT  | 38.20     | 1.58  | 2.97  | 4         |

**Integer scores**: AVIF = 1, ITU = 5, FRAPPE = 3, SEAOTTER-ZS = 3,
SEAOTTER-FT = 3.

**Disagreement**: even with log normalization, FRAPPE/SEAOTTER end up
at 3 — between AVIF (worst) and ITU (libjpeg-fast). The hardcoded v2
puts them at 4 ("clears the deployment-tier threshold"), which is
not a normalization choice but a deployment-tier framing. The
data-driven rule has no concept of "fast enough for 30 fps 1080p."

---

## `dec` — consumer-side decode throughput (log scale)

Scalar: cls-task decode MPx/s. Log normalization (range spans 0.61
to 192, ~2.5 orders of magnitude).

| system       | dec MPx/s | log₁₀ | score | hardcoded |
|--------------|----------:|------:|------:|----------:|
| AVIF         | 20.75     | 1.32  | 3.46  | 4         |
| ITU          | 192       | 2.28  | 5.00  | 5         |
| FRAPPE       | 0.61      | −0.21 | 1.00  | 1         |
| SEAOTTER-ZS  | 56.83     | 1.75  | 4.15  | 5         |
| SEAOTTER-FT  | 57.90     | 1.76  | 4.16  | 5         |

**Integer scores**: AVIF = 3, ITU = 5, FRAPPE = 1, SEAOTTER-ZS = 4,
SEAOTTER-FT = 4.

**Disagreement**: data-driven keeps SEAOTTER variants near (but not
at) the top — they're 3.4× slower than libjpeg, which log
normalization treats as "good" not "best." The hardcoded v2 puts
them at 5 (tied with ITU). AVIF drops from 4 to 3 (it's only
slightly faster than the FRAPPE-decoder-less SEAOTTER path's
overhead floor).

---

## Summary table

`(data-driven / hardcoded)` per axis × system; rounded integer
scores. Bolded entries differ by ≥ 1 unit.

| axis  | AVIF        | ITU         | FRAPPE      | SEAOTTER-ZS  | SEAOTTER-FT |
|-------|-------------|-------------|-------------|--------------|-------------|
| trans | **4** / 2   | 1 / 1       | **5** / 3   | **5** / 4    | 5 / 5       |
| store | 4 / 4       | **2** / 1   | 5 / 5       | **1** / 2    | **1** / 3   |
| PSNR  | **5** / 4   | **4** / 2   | **5** / 4   | **5** / 4    | 1 / 1       |
| DISTS | 4 / 4       | 1 / 1       | 5 / 5       | 5 / 5        | **2** / 3   |
| cls   | 4 / 3       | 1 / 1       | **4** / 2   | 4 / 4        | 5 / 5       |
| seg   | **5** / 4   | 1 / 1       | **4** / 2   | **4** / 3    | **5** / 4   |
| clip  | **4** / 2   | 1 / 1       | **4** / 2   | 4 / 4        | 5 / 5       |
| enc   | 1 / 1       | 5 / 5       | **3** / 4   | **3** / 4    | **3** / 4   |
| dec   | **3** / 4   | 5 / 5       | 1 / 1       | **4** / 5    | **4** / 5   |

## Where the data and the hardcoded radar disagree, and why

Three structural mismatches:

1. **The data-driven rule has no way to encode "rate context."**
   On `cls` and `clip`, AVIF / FRAPPE / SEAOTTER-ZS land in the same
   integer bucket because their absolute top-1 values differ by only
   2–5 pp. The hardcoded radar spreads them out to capture the fact
   that AVIF spends ~35 % more bits on cls and ~76 % more on clip
   to reach those accuracies. A radar axis labeled "downstream
   accuracy at matched rate" cannot, by construction, see the rate
   side of the trade. To keep the rate-context separation, the
   radar needs the hand-curated story; to keep it data-driven, the
   radar should be re-labeled as "absolute downstream accuracy."

2. **Linear normalization compresses against extreme outliers.**
   On `store` the SEAOTTER-ZS storage CR (18.3) drags the floor so
   low that anything between 80 and 215 looks the same (all ≥ 2.2).
   ITU lifts from 1 to 2 mostly because the floor moved, not
   because ITU's storage improved. The v2 hardcoded "1 / 2 / 5 / 2 / 3"
   distribution is more readable but is anchoring to a different
   intuition than the data.

3. **Throughput axes need a deployment-tier framing the data lacks.**
   `enc` and `dec` both span 2+ orders of magnitude, so even log
   normalization can't reproduce the hardcoded "4 — good enough for
   real-time" assignment to the FRAPPE family and SEAOTTER variants.
   The hardcoded radar treats throughput as a step function (real-
   time-on-CPU yes/no); the data-driven rule treats it as a
   continuous magnitude.

## Recommendation

Three options for the figure:

- **A. Adopt the data-driven scores wholesale.** Replace the
  `RANKINGS` dict with the rounded values from the summary table
  above. The radar becomes a faithful linear/log projection of
  Table 1; cross-row scores stay defensible. Costs: collapses the
  FRAPPE / SEAOTTER-ZS / SEAOTTER-FT separation on `trans`, makes
  SEAOTTER's `store` rating look unforgivingly harsh, and demotes
  throughput. The radar's visual signature becomes "SEAOTTER-FT
  dominates the downstream-accuracy half cleanly; the bottom
  spokes (store, dec, enc) are honestly mediocre."

- **B. Keep the current hardcoded radar; cite this audit in
  `notes/` for transparency.** No figure changes. The radar
  represents a deliberate editorial framing (rate-context-aware
  accuracy, deployment-tier-aware throughput, etc.) that the data
  alone doesn't surface. The cost is the data-driven critique
  exists separately and reviewers may notice the gap.

- **C. Hybrid — adopt the data-driven axes where they agree
  closely, hand-curate the few that don't.** The cls / DISTS / PSNR
  axes are within 1 of hardcoded almost everywhere; adopt those.
  Hand-curate `trans`, `clip`, `seg` (rate-context disagreements)
  and `enc`, `dec` (deployment-tier disagreements). Most defensible
  for a paper, most work.

The most natural axis to keep data-driven is **`DISTS`** — only one
disagreement (SEAOTTER-FT 2 vs 3) and the values cluster naturally.
The most natural axis to keep hand-curated is **`enc` / `dec`** —
deployment-tier framing matters more than the raw MPx/s magnitude
for the SEAOTTER deployment story.
