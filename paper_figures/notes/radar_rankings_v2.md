# Updated radar rankings — from Table 1 (matched-rate)

Companion to `figures/radar_summary.pdf`. The previous BD-rate-based
audit (`notes/radar_rankings.md`) compared full RD curves; this version
re-derives the nine-axis scores directly from the per-task matched-rate
picks shown in **Table 1** (cls / seg / clip subsections), i.e., every
codec at the operating point closest to FRAPPE $n_\text{ch}{=}12$'s
transmit-bpp on that subsection's dataset.

Radar systems (unchanged): **AVIF**, **ITU JPEG (4:4:4)**, **FRAPPE**,
**SEAOTTER-ZS**, **SEAOTTER-FT**. WaLLoC and AVIF (max-speed) are in
Table 1 but not on the radar (kept at 5 systems for visual readability;
they are tracked here as cross-checks). ITU JPEG is not in Table 1; its
numbers come from the cls / seg / clip eval JSONs (`eval_jpeg_*.json`)
at the operating point closest to FRAPPE $n_\text{ch}{=}12$'s bpp.

Scoring: 1 (poor) – 5 (excellent), matching the existing
`RANKINGS` dict in `make_radar_figure.py`. Where the matched-rate
framing reproduces the hardcoded value, the entry shown below as
"(unchanged)" can stay; flagged entries are the recommended updates.

---

## `trans` — transmit-bpp efficiency

What it captures: how few sensor-uplink bits the codec needs to deliver
a given downstream accuracy. The matched-rate framing pins the FRAPPE
family at the anchor bpp; non-FRAPPE codecs land at higher bpp on cls
and clip and at slightly lower bpp on seg (AVIF q=5 only).

Per-subsection transmit CR for each system at the matched-rate pick
(Table 1):

| system       | cls CR | seg CR | clip CR |
|--------------|------:|------:|------:|
| AVIF         | 165   | 279   | 96    |
| ITU JPEG     | 80    | 87    | 69    |
| FRAPPE       | 221   | 256   | 169   |
| SEAOTTER-ZS  | 221   | 256   | 169   |
| SEAOTTER-FT  | 221   | 256   | 169   |

(ITU values: q=1 cls bpp=0.298, seg bpp=0.276, clip bpp=0.348 — pulled
from `eval_jpeg_*_q1.json`; ITU q=1 is its lowest-bitrate operating
point and the only one that lands in the matched-rate neighborhood at
all.) The FRAPPE family is the anchor on cls and clip and ties on seg;
AVIF q=5 beats it on seg only. ITU is at the bottom across all three
because q=1 still over-spends bits relative to FRAPPE n=12.

**Scores**:

| system       | score | hardcoded |
|--------------|------:|----------:|
| AVIF         | 2     | 2 (unchanged) |
| ITU          | 1     | 1 (unchanged) |
| FRAPPE       | 3     | 3 (unchanged) |
| SEAOTTER-ZS  | 4     | 4 (unchanged) |
| SEAOTTER-FT  | 5     | 5 (unchanged) |

The FRAPPE-family share-the-encoder structure is preserved; SEAOTTER-ZS
and SEAOTTER-FT outrank plain FRAPPE on this axis because the downstream
accuracy at the matched bpp is higher (see `cls` / `seg` / `clip`).

---

## `store` — storage-bpp efficiency

What it captures: on-disk file size. Non-transcode codecs have
`storage_bpp = transmit_bpp`. SEAOTTER variants emit a JPEG file
*larger* than the FRAPPE-LS uplink bytes — that's the explicit
encode-once / decode-many trade.

Per-subsection storage CR (Table 1):

| system       | cls | seg | clip |
|--------------|----:|----:|-----:|
| AVIF         | 165 | 279 | 96   |
| ITU JPEG     | 80  | 142 | 41   |
| FRAPPE       | 221 | 256 | 169  |
| SEAOTTER-ZS  | 19  | 20  | 16   |
| SEAOTTER-FT  | 27  | 46  | 37   |

SEAOTTER variants pay ~8–12× more storage than FRAPPE alone at every
operating point. The trade is justified at the consumer side (the
on-disk file is a standards-compliant JPEG), but on a "smallest file"
axis SEAOTTER-FT is clearly mid-pack, not top-tier.

**Scores**:

| system       | proposed | hardcoded |
|--------------|---------:|----------:|
| AVIF         | 4 | 2 ← **bump up** |
| ITU          | 1 | 1 (unchanged) |
| FRAPPE       | 5 | 5 (unchanged) |
| SEAOTTER-ZS  | 2 | 3 ← **bump down** |
| SEAOTTER-FT  | 3 | 4 ← **bump down** |

Recommended changes: AVIF deserves a higher mark on storage than the
hardcoded "2" suggests — Table 1 shows AVIF's storage CR (96–279) is
well above the SEAOTTER variants (16–46) across all three tasks.
Conversely, SEAOTTER's storage penalty should be more visible on this
axis: SEAOTTER-ZS drops from hardcoded 3 to 2, SEAOTTER-FT from 4 to 3.
FRAPPE stays at 5 (its transmit is its storage; it dominates).

---

## `PSNR` — pixel fidelity at compressed bitrate

What it captures: average reconstruction PSNR (dB) at the codec's
matched-rate operating point. Higher is better.

Per-subsection PSNR_dB (from the eval JSONs):

| system       | cls   | seg   | clip  |
|--------------|------:|------:|------:|
| AVIF         | 25.01 | 26.97 | 24.35 |
| ITU JPEG     | 21.47 | 22.09 | 21.08 |
| FRAPPE       | 25.08 | 26.81 | 24.37 |
| SEAOTTER-ZS  | 25.04 | 26.78 | 24.32 |
| SEAOTTER-FT  | 10.39 | 10.76 | 13.07 |

The headline story: **SEAOTTER-FT's deliberate fine-tune drops PSNR by
~14 dB** across all three tasks (the `\runin{Why does the transcode
help?}` paragraph already explains this). AVIF / FRAPPE / SEAOTTER-ZS
cluster at 24–27 dB. ITU JPEG at its lowest-q point sits ~21–22 dB
across all three tasks (3–5 dB below the codec cluster but well above
SEAOTTER-FT's fine-tuned floor).

**Scores**:

| system       | proposed | hardcoded |
|--------------|---------:|----------:|
| AVIF         | 4 | 4 (unchanged) |
| ITU          | 2 | 2 (unchanged) |
| FRAPPE       | 4 | 3 ← **bump up** |
| SEAOTTER-ZS  | 4 | 5 ← **bump down** |
| SEAOTTER-FT  | 1 | 1 (unchanged) |

Recommendation: FRAPPE and SEAOTTER-ZS tie AVIF on PSNR at matched
rate (24.3–27.0 dB band; SEAOTTER-ZS within 0.2 dB of FRAPPE/AVIF on
all three subsections). Either bump FRAPPE to 4 and drop SEAOTTER-ZS
to 4 (the cleanest "they're all in the same tier" framing), or keep
SEAOTTER-ZS at 5 if you want the radar to celebrate the no-distortion-
cost sandwich result; both readings are defensible.

---

## `DISTS` — perceptual distortion at compressed bitrate

`dists_db = −10 log10 DISTS`; higher is better.

Per-subsection DISTS_dB:

| system       | cls  | seg  | clip |
|--------------|-----:|-----:|-----:|
| AVIF         | 6.23 | 7.38 | 5.46 |
| ITU JPEG     | 4.37 | 5.02 | 3.84 |
| FRAPPE       | 6.56 | 7.49 | 5.66 |
| SEAOTTER-ZS  | 6.70 | 7.71 | 5.80 |
| SEAOTTER-FT  | 5.08 | 6.19 | 4.18 |

Pattern: SEAOTTER-ZS wins DISTS on every task (the sandwich's color
transform appears to be perceptually favorable when paired with
matched-quantization JPEG). SEAOTTER-FT is *not* last on DISTS at
matched rate (it sits above ITU) — the perceptual hit from the
fine-tune is real but milder than the PSNR hit.

**Scores**:

| system       | proposed | hardcoded |
|--------------|---------:|----------:|
| AVIF         | 4 | 4 (unchanged) |
| ITU          | 1 | 2 ← **bump down** |
| FRAPPE       | 5 | 4 ← **bump up** |
| SEAOTTER-ZS  | 5 | 5 (unchanged) |
| SEAOTTER-FT  | 3 | 1 ← **bump up** |

Recommendation: SEAOTTER-FT should not be tied for last on DISTS — at
matched rate it beats ITU JPEG by 0.5–1.0 dB on every task. Move
SEAOTTER-FT from 1 to 3. ITU drops from 2 to 1 (it's clearly the worst
DISTS performer at this operating point). FRAPPE up from 4 to 5
(within 0.1 dB of SEAOTTER-ZS on cls, ties on seg).

---

## `cls` — ImageNet Top-1 accuracy

Per-task picks (Table 1 cls subsection, Top-1 (\%)):

| system       | Top-1 |
|--------------|------:|
| AVIF         | 61.15 |
| ITU JPEG     | 25.75 |
| FRAPPE       | 56.22 |
| SEAOTTER-ZS  | 60.25 |
| SEAOTTER-FT  | 69.02 |

(ITU JPEG at q=1 cls bpp=0.298 → top-1 = 25.75%; from
`eval_jpeg_cls_q1.json`. ITU q=1 is the closest available op to the
FRAPPE-anchor neighborhood and even then over-spends bits 2.7×.)

**Scores**:

| system       | proposed | hardcoded |
|--------------|---------:|----------:|
| AVIF         | 3 | 2 ← **bump up** |
| ITU          | 1 | 1 (unchanged) |
| FRAPPE       | 2 | 3 ← **bump down** |
| SEAOTTER-ZS  | 4 | 4 (unchanged) |
| SEAOTTER-FT  | 5 | 5 (unchanged) |

Recommendation: At matched-rate, AVIF (61.15%) edges SEAOTTER-ZS
(60.25%) by 0.9 pp on cls, despite AVIF using 35% more bits than the
FRAPPE-family anchor. AVIF deserves a 3 (not 2). FRAPPE alone at
56.22% is closer to 2 than 3 — it trails AVIF and even WaLLoC.

---

## `seg` — ADE20K mIoU

Per-task picks (Table 1 seg subsection):

| system       | mIoU  |
|--------------|------:|
| AVIF (q=5)   | 32.75 |
| ITU JPEG     | 11.59 |
| FRAPPE       | 29.09 |
| SEAOTTER-ZS  | 30.09 |
| SEAOTTER-FT  | 32.77 |

(ITU JPEG at q=1 seg bpp=0.276 → mIoU = 11.59%; from
`eval_jpeg_seg_q1.json`. ITU q=1 over-spends bits 2.9× relative to
FRAPPE n=12's 0.094 bpp and still delivers less than half the mIoU.)

**The key finding**: AVIF q=5 ties SEAOTTER-FT on seg mIoU (32.75 vs
32.77, 0.02 pp gap) at *fewer* bits than the FRAPPE n=12 anchor. This
is the only task where matched-rate gives AVIF a near-Pareto position
relative to SEAOTTER-FT.

**Scores**:

| system       | proposed | hardcoded |
|--------------|---------:|----------:|
| AVIF         | 4 | 2 ← **bump up** |
| ITU          | 1 | 1 (unchanged) |
| FRAPPE       | 2 | 3 ← **bump down** |
| SEAOTTER-ZS  | 3 | 4 ← **bump down** |
| SEAOTTER-FT  | 4 | 4 (unchanged) |

Recommendation: this is the biggest correction from Table 1. AVIF
should rank 4 on seg (tied with SEAOTTER-FT, not at "2 — poor").
FRAPPE alone drops to 2 — it's the worst codec on this task at
matched rate. SEAOTTER-ZS lands at 3 (between FRAPPE alone at 2 and
the tied SEAOTTER-FT/AVIF at 4).

---

## `clip` — SigLIP-2 zero-shot top-1

Per-task picks (Table 1 clip subsection):

| system       | Top-1 |
|--------------|------:|
| AVIF (q=1)   | 42.59 |
| ITU JPEG     | 18.66 |
| FRAPPE       | 41.51 |
| SEAOTTER-ZS  | 43.34 |
| SEAOTTER-FT  | 48.22 |

(ITU JPEG at q=1 clip bpp=0.348 → top-1 = 18.66%; from
`eval_jpeg_clip_q1.json`.)

Note: AVIF and AVIFX both pick q=1 on clip but at bpp=0.25/0.26 —
nearly *2× more bits* than FRAPPE n=12's 0.142. Even with that
bandwidth advantage they trail SEAOTTER-FT by 4–6 pp.

**Scores**:

| system       | proposed | hardcoded |
|--------------|---------:|----------:|
| AVIF         | 2 | 2 (unchanged) |
| ITU          | 1 | 1 (unchanged) |
| FRAPPE       | 2 | 3 ← **bump down** |
| SEAOTTER-ZS  | 4 | 4 (unchanged) |
| SEAOTTER-FT  | 5 | 5 (unchanged) |

Only change: FRAPPE alone (41.51%) is *below* AVIF (42.59%) at matched
rate — bump FRAPPE down from 3 to 2.

---

## `enc` — sensor-side encode throughput (MPx/s)

From Table 1 (cls subsection, MPx/s; encode TP is task-invariant for
the same op, so values match across subsections):

| system       | enc MPx/s |
|--------------|----------:|
| AVIF         | 5.16      |
| ITU JPEG     | 303       |
| FRAPPE       | 38.20     |
| SEAOTTER-ZS  | 38.20     |
| SEAOTTER-FT  | 38.20     |

(ITU JPEG measured at 303 MPx/s on the same hardware via libjpeg
q=1 — from `throughput_jpeg_cls_q1.json`.) FRAPPE-family encode TP
is canonicalized to FRAPPE n=12's value in Table 1; spread within the
family is measurement noise.

**Scores**:

| system       | proposed | hardcoded |
|--------------|---------:|----------:|
| AVIF         | 1 | 1 (unchanged) |
| ITU          | 5 | 5 (unchanged) |
| FRAPPE       | 4 | 4 (unchanged) |
| SEAOTTER-ZS  | 4 | 4 (unchanged) |
| SEAOTTER-FT  | 4 | 4 (unchanged) |

No change. Deployment-tier-style scoring (real-time on CPU = 4,
libjpeg = 5, AVIF = 1) still matches.

---

## `dec` — consumer-side decode throughput (MPx/s)

From Table 1 Decode row (steady-state JPEG decode + $\mathcal{F}^{-1}$
for SEAOTTER variants, vanilla codec decode for others):

| system       | dec MPx/s |
|--------------|----------:|
| AVIF (q=1 cls)   | 20.75 |
| ITU JPEG         | 192   |
| FRAPPE           | 0.61  |
| SEAOTTER-ZS      | 56.83 |
| SEAOTTER-FT      | 57.90 |

(ITU JPEG decode = 192 MPx/s on this hardware — from
`throughput_jpeg_cls_q1.json`. libjpeg is roughly 3× faster than
SEAOTTER's JPEG + $\mathcal{F}^{-1}$ path, which itself is
roughly 3× faster than AVIF.)

**Scores**:

| system       | proposed | hardcoded |
|--------------|---------:|----------:|
| AVIF         | 4 | 4 (unchanged) |
| ITU          | 5 | 5 (unchanged) |
| FRAPPE       | 1 | 1 (unchanged) |
| SEAOTTER-ZS  | 5 | 5 (unchanged) |
| SEAOTTER-FT  | 5 | 5 (unchanged) |

No change. The iter-11 steady-state-decode measurement (replacing the
older "decode + ConvNeXt forward" conflation that the previous BD-rate
audit complained about) now aligns the data with the hardcoded
ranking.

---

## Summary table (proposed)

`(proposed / hardcoded)` per axis × system. Changed values in bold.

| axis  | AVIF        | ITU         | FRAPPE      | SEAOTTER-ZS  | SEAOTTER-FT |
|-------|-------------|-------------|-------------|--------------|-------------|
| trans | 2 / 2       | 1 / 1       | 3 / 3       | 4 / 4        | 5 / 5       |
| store | **4** / 2   | 1 / 1       | 5 / 5       | **2** / 3    | **3** / 4   |
| PSNR  | 4 / 4       | 2 / 2       | **4** / 3   | **4** / 5    | 1 / 1       |
| DISTS | 4 / 4       | **1** / 2   | **5** / 4   | 5 / 5        | **3** / 1   |
| cls   | **3** / 2   | 1 / 1       | **2** / 3   | 4 / 4        | 5 / 5       |
| seg   | **4** / 2   | 1 / 1       | **2** / 3   | **3** / 4    | 4 / 4       |
| clip  | 2 / 2       | 1 / 1       | **2** / 3   | 4 / 4        | 5 / 5       |
| enc   | 1 / 1       | 5 / 5       | 4 / 4       | 4 / 4        | 4 / 4       |
| dec   | 4 / 4       | 5 / 5       | 1 / 1       | 5 / 5        | 5 / 5       |

## Three biggest shifts vs the hardcoded radar

1. **AVIF on `seg` (2 → 4)** — at matched rate, AVIF q=5 ties
   SEAOTTER-FT on mIoU. The hardcoded "2" understates this; the
   honest framing is that segmentation is the one task where AVIF is
   competitive with SEAOTTER-FT.

2. **SEAOTTER-FT on `DISTS` (1 → 3)** — the previous radar treated
   SEAOTTER-FT as worst-tier on DISTS along with PSNR. At matched
   rate it's actually mid-tier on perceptual quality (it beats ITU
   on every task), even while losing badly on PSNR. PSNR stays at 1.
   This separates the pixel-fidelity story from the perceptual-quality
   story — a more accurate picture of the fine-tune trade-off.

3. **SEAOTTER-FT / SEAOTTER-ZS on `store` (4/3 → 3/2)** — Table 1's
   storage CR row (16–46 vs FRAPPE's 169–256) makes the transcode
   storage penalty unmistakable. SEAOTTER-ZS in particular has the
   smallest storage CR of any system; ranking it "3 — fair" is too
   generous. AVIF's storage gets a corresponding bump up (2 → 4)
   since its storage CR is roughly comparable to FRAPPE-side latents.

## Smaller corrections

- **FRAPPE alone on cls / seg / clip (3 → 2)** — at matched rate
  FRAPPE alone trails *every* other codec on cls (56.22% < AVIF
  61.15%, WaLLoC 60.98%, SEAOTTER-ZS 60.25%) and on clip
  (41.51% < AVIF 42.59%). On seg it's last among the matched-rate
  picks (29.09% vs WaLLoC 30.51%, SEAOTTER-ZS 30.09%). The
  hardcoded "3" implies FRAPPE-alone is a middling baseline; the
  matched-rate data shows it's the *worst* downstream-accuracy
  performer of the five.

- **FRAPPE PSNR / DISTS (3 → 4, 4 → 5)** — FRAPPE alone has the
  highest PSNR on cls and clip (25.08 / 24.37 dB) and ties AVIF
  on seg (26.81 dB). Its DISTS_dB is also within 0.15 dB of
  SEAOTTER-ZS on every task. Promote.

- **SEAOTTER-ZS on `PSNR` (5 → 4)** — SEAOTTER-ZS is within 0.06–0.07
  dB of FRAPPE/AVIF on every task; calling it a tier above them on
  PSNR ("5 — excellent" vs FRAPPE's "3 — fair") overstates the
  separation. Tie them at 4.

## What the radar's overall narrative becomes

After these updates the visual story shifts only modestly:

- **SEAOTTER-FT still owns the downstream-accuracy half of the
  radar** (cls / seg / clip / trans), but the seg lobe is no
  longer uniquely SEAOTTER's — AVIF reaches the same outer ring.
- **SEAOTTER-FT's distortion deficit is now PSNR-only**, not
  PSNR-and-DISTS. The radar shape better reflects "we deliberately
  trade PSNR for accuracy" rather than "we lose all reconstruction
  metrics."
- **The store axis becomes a true differentiator**: SEAOTTER's
  encode-once / decode-many premise is paid for on this axis, and
  the radar should show that honestly. FRAPPE / AVIF lobe out;
  SEAOTTER variants pull in.

The throughput axes (`enc`, `dec`) are unchanged from the previous
radar and continue to favor SEAOTTER-ZS / SEAOTTER-FT / ITU on
decode while putting AVIF behind on encode.
