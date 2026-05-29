# Superseded-result prune (public-release prep, 2026-05-29)

Records the one set of superseded result files removed from this mirror
while preparing the public release. **Everything here is recoverable
from git history** — the files were committed and pushed (last touched
by `bd7dfb7 results`; `HEAD` is an ancestor of `origin/main`), so they
remain retrievable via `git show`/`git checkout <commit> -- <path>` if
ever needed. The numbers were superseded, so it is unlikely.

## Hypothesis going in

> The only superseded data is the **old encode-throughput metric**,
> replaced on 2026-05-28 by `encode_complexity/` after re-running under
> the FRAPPE evaluation setting (`compressors` reference harness).

## What was verified (not assumed)

Static + empirical dependency analysis of all six generators
(`make_{figures,extra_figures,codec_kodak_fig,tables,radar_figure_data}.py`):

1. **No generator reads `seg/throughput/`.** `make_tables.py` calls
   `_load_thr("cls")` three times (headline / deployment-tier /
   throughput tables) but **never** `_load_thr("seg")`; the
   consumer-decode column for *every* task (incl. seg and clip) is
   sourced from `cls/throughput/` because decode MPx/s is treated as
   task-invariant at the same op (see `emit_headline_table._thr_for`,
   which always indexes `thr_cls`). No other generator opens a
   `*/throughput/` path except `make_radar_figure_data.py`, which also
   reads only `cls/throughput`.
2. **Empirical confirmation.** Moving `seg/throughput/` aside and
   regenerating all 8 `.tex` tables produced **byte-identical** output
   to the committed `corl_2026/figures/*.tex`. Removing it changes no
   paper artifact.
3. **The old encode reader is dead code.** `make_tables.py._enc_mpx`
   (the deprecated reader of `throughput.encode.median_ms`) has **zero
   call sites**; every encode MPx/s comes from `encode_complexity/` via
   `_enc_only_mpx`.
4. **`seg/throughput/` is the old convention end-to-end.** Sampled
   `throughput_seaft_seg_n12.json` is `harness_version: iter6-1`,
   `config.device: cuda:0`, throughput blocks `['encode','consumer']`
   with **no `consumer_decode_only`** split. Both of its fields are
   superseded: the `encode` field by `encode_complexity/seg_512/`, and
   the `consumer` field by the iter-11 CPU measurement in
   `cls/throughput/` (`iter11-cpu-2`, which *does* carry the
   `consumer_decode_only` block the paper's consumer-cost column
   requires). `CLAUDE.md` already flags iter-6 `seg/throughput` as "not
   to be used as-is for consumer-cost claims."

## Deleted (40 files)

```
results/seg/throughput/throughput_{avif,avifx,jpeg,jp2,webp,frp,wal,seab,seaft,walsand}_seg_*.json
```

(all 40 `throughput_*_seg_*.json` under `results/seg/throughput/`; the
now-empty directory is removed.)

## Kept despite containing superseded data

- **`cls/throughput/` (60 files) — KEPT.** Mixed: its `encode` field is
  superseded by `encode_complexity/cls_384/`, **but** its
  `consumer_decode_only` field is the live source for the decode column
  of every table (task-invariant). `harness_version: iter11-cpu-2`.
  Removing or hand-editing it would change paper output and strip a
  load-bearing field, so the whole file stays (we do not hand-edit
  result JSONs).

## Kept because valid-but-unused ≠ superseded

These are *current* data that simply isn't plotted in the paper — not
flawed/superseded, so not pruned (same policy as `walft`):

- **`encode_complexity/kodak_native/` (47 files)** — current
  FRAPPE-reference encode methodology on Kodak; no paper figure plots
  it, but it is not superseded.
- **`walft` rows, per-cell lambda-retry sweeps, pre-iter-10 `walft`** —
  intentionally excluded from tables per `results.md` / `CLAUDE.md`, but
  valid.

## Conclusion (stated plainly per the prune brief)

The deletion set is **exactly `seg/throughput/` (40 files)** — slightly
narrower than "the old encode-throughput metric everywhere," because at
*file* granularity the cls old-encode numbers are co-located with the
load-bearing consumer-decode field and so their files are kept;
`seg/throughput/` is the only directory where the superseded throughput
convention sits in files that no generator reads. After deletion, all
generators reproduce identical `figures/` and `.tex` output.
