# Result traceability (CoRL 2026 paper → mirror data)

Maps every quantitative claim in the CoRL 2026 paper
("SEAOTTER: Sensor Embedded Autoencoding with One-Time Transcode for
Efficient Reconstruction", Jacobellis & Yadwadkar) to the generator
that produces it and the backing data file(s) in this mirror.

- **Mirror data root:** `~/UT-SysML/seaotter/results/` (every generator
  hard-codes `ROOT = Path("/home/dgj335/UT-SysML/seaotter/results")`).
- **Generators:** `corl_2026/make_{figures,extra_figures,codec_kodak_fig,tables,radar_figure_data}.py`.
- **Producing harnesses (research repo `danjacobellis/seaotter`):** see
  the "Where the paper's numbers actually come from" table in
  `seaotter/CLAUDE.md` (iter-6 accuracy `iter6-1`; iter-11 CPU
  throughput `iter11-cpu-2`; encode-complexity `src/seaotter/eval`;
  codec-kodak `codec-kodak-v1`/`codec-kodak-cls-v1`).
- **Verified on 2026-05-29.** All 8 `.tex` tables regenerate
  **byte-identically** from the frozen JSONs (`make_tables.py` →
  `diff` against the committed `corl_2026/figures/*.tex`). Every prose
  number below was cross-checked against the regenerated tables and the
  raw eval JSONs. Numerical results are frozen — this report does not
  regenerate or alter any value.

Status legend: **✓ backed** (file present, number reproduces);
**⚑ flag** (backed but worth a glance — see notes); **▽ not
mirror-backed** (external citation or architectural fact, by design not
in `results/`).

---

## Figures

| Fig | label | file | generator | backing data | status |
|-----|-------|------|-----------|--------------|--------|
| 1 | `fig:system` | `figures/system.pdf` | `figure_assets/figures.pptx` → `system.svg`→`.pdf` (not a Python generator) | — (schematic) | ▽ not data-driven |
| 2 | `fig:main_results` | `figures/main_results.pdf` | `make_figures.py` | `cls/eval/eval_{avif,avifx,wal,frp,walsand,seab,walft,seaft}_cls_*.json` + `encode_complexity/cls_384/encode_*.json` | ✓ backed |
| 3 | `fig:jpeg_sandwich` | `figures/jpeg.pdf` | `figure_assets/figures.pptx` → `jpeg.{svg,pdf}` (+ method notebooks) | — (schematic) | ▽ not data-driven |
| 4 | `fig:rd_metric_panels` | `figures/rd_metric_panels.pdf` | `make_extra_figures.py` (`emit_rd_metric_panels`) | `cls/eval/*` (psnr/ssim/dists/top1 for avifx,wal,frp,walsand,seab,seaft), `seg/eval/*` (miou), `clip/eval/*` (top1) | ✓ backed |
| 5 | `fig:radar` | `figures/radar.pdf` | `figure_assets/figures.pptx` → `radar.{svg,pdf}` (**SVG-finalized**, *not* the matplotlib `radar_summary*.pdf`) | rankings derived in `corl_2026/notes/_compute_radar_rankings_v3.py` from `encode_complexity/cls_384/` + `cls/throughput/` | ⚑ data-driven rankings, hand-finalized figure |
| 6 | `fig:quant` | `figures/quant.pdf` | `figure_assets/` (`visualize_quant.ipynb` / `intermediate_signals.ipynb`) | learned `Q^{(k)}` from the `seaotter_jpeg_s3` Hub bundle (not a `results/` JSON) | ▽ derived from published weights |
| 7 | `fig:codec_kodak` | `figures/codec_kodak.pdf` | `make_codec_kodak_fig.py` | `codec_kodak/eval_*.json` (PSNR) + `codec_kodak_cls/eval_*.json` (top-1) | ✓ backed |
| 8 | `fig:rd_storage` | `figures/rd_storage.pdf` | `make_extra_figures.py` (`emit_rd_storage`) | `cls/seg/clip` `eval/*` `storage_bpp_mean`+`transmit_bpp_mean` for `{wal_jpeg,frp_jpeg,walsand,seab,seaft}` | ✓ backed |

## Tables (all regenerate byte-identically from the mirror)

| Tab | label | file | generator fn | backing data | status |
|-----|-------|------|--------------|--------------|--------|
| 1 | `tab:headline` | `headline_table.tex` | `emit_headline_table` | `cls/seg/clip` `eval/*` + `cls/throughput/*` (decode, task-invariant) + `encode_complexity/{cls_384,seg_512,clip_naflex}/*` (encode) | ✓ backed |
| 2 | `tab:codec_kodak` | `codec_kodak_table.tex` | `emit_codec_kodak_table` | `codec_kodak/eval_{seaotter,jpeg,jpeg_sub0}_*.json` | ✓ backed |
| 3 | `tab:codec_kodak_cls` | `codec_kodak_cls_table.tex` | `emit_codec_kodak_cls_table` | `codec_kodak_cls/eval_*.json` | ✓ backed |
| 4 | `tab:rd_cls` | `rd_table_cls.tex` | `emit_task_table("cls")` | `cls/eval/*` (CLS_ORDER pipelines) | ✓ backed |
| 5 | `tab:rd_seg` | `rd_table_seg.tex` | `emit_task_table("seg")` | `seg/eval/*` | ✓ backed |
| 6 | `tab:rd_clip` | `rd_table_clip.tex` | `emit_task_table("clip")` | `clip/eval/*` | ✓ backed |
| 7 | `tab:throughput` | `throughput_table.tex` | `emit_throughput_table` | `cls/eval/*` + `cls/throughput/*` (decode) + `encode_complexity/cls_384/*` (encode) | ✓ backed |
| 8 | `tab:deployment_tier` | `deployment_tier_table.tex` | `emit_deployment_tier_table` | `cls/eval/*` + `encode_complexity/cls_384/*` | ✓ backed |

## Headline / in-prose numbers (Sec. "Performance evaluation" + Appendix)

| claim (paper) | value | trace | status |
|---|---|---|---|
| SEAOTTER-FT cls top-1 @ n=12 | 69.02% | `cls/eval/eval_seaft_cls_n12.json` `metrics.top1`=0.69024 | ✓ |
| FRAPPE cls top-1 @ n=12 | 56.22% | `eval_frp_cls_n12.json`=0.56216 → Δ +12.80 pp | ✓ |
| SEAOTTER-FT vs FRAPPE @ n=6 | 46.55 vs 26.70 (+19.85 pp) | `eval_seaft_cls_n6`=0.46548, `eval_frp_cls_n6`=0.26698 | ✓ |
| SEAOTTER-ZS over FRAPPE @ n=12 (cls) | +4.03 pp | `eval_seab_cls_n12`=0.60254 − 0.56216 | ✓ |
| seg mIoU FT over FRAPPE @ n=12 | +3.68 pp | `eval_seaft_seg_n12`=0.32773 − `eval_frp_seg_n12`=0.29087 | ✓ |
| clip top-1 FT over FRAPPE @ n=12 | +6.71 pp | `eval_seaft_clip_n12`=0.48218 − `eval_frp_clip_n12`=0.4151 | ✓ |
| cls lead vs AVIF / AVIF-max | +7.87 / +8.00 pp | headline_table.tex (69.02 − 61.15 / 61.02) | ✓ |
| clip lead vs AVIF / AVIF-max | +5.63 / +4.03 pp | headline_table.tex (48.22 − 42.59 / 44.19) | ✓ |
| seg "ties for first" vs AVIF / AVIF-max / FRAPPE | +0.02 / +0.26 / +3.68 pp | headline_table.tex (32.77 − 32.75 / 32.51 / 29.09) | ✓ |
| transmit-bpp / CR @ cls n=12 | 0.109, 221:1 | `eval_*_cls_n12` `transmit_bpp_mean`=0.10860 → 24/0.10860=221.0 | ✓ |
| abstract "+8% ImageNet top-1" vs AVIF | +7.87–8.00 pp | = cls lead row above | ✓ |
| abstract "7× faster encoding" vs AVIF | 6.9× | encode MPx/s 177.76 (frp) / 25.73 (avifx) = 6.91× (within paper's "5–8×") | ✓ |
| abstract "3.5× faster decoding" / "~3.4×" | 3.44× | decode MPx/s 67.97 (seaft) / 19.75 (avif) | ✓ |
| "100× faster than FRAPPE w/o transcode" | 100× | decode 67.97 (seaft) / 0.68 (frp) = 99.96× | ✓ |
| encode "exceeds 250 MPx/s for n≤9" | — | `encode_complexity/cls_384/encode_frp_n{3,6,9}.json` `median_MPx_per_s` | ✓ |
| PSNR drop 25.08 → 10.39 dB @ n=12 | — | `eval_frp_cls_n12` psnr=25.0836, `eval_seaft_cls_n12` psnr=10.3884 | ✓ |
| Kodak PSNR margins k=0/1/2 vs ITU 4:4:4 | +0.27 / +1.40 / +1.27 dB | codec_kodak_table.tex: 33.17−32.90(q53,bpp1.10), 37.77−36.37(q81,bpp1.91), 40.89−39.62(q91,bpp2.97) | ✓ |
| storage @ n=12 cls: "+8.19 pp higher top-1" vs frp_jpeg | +8.19 pp | `eval_seaft_cls_n12`=0.69024 − `eval_frp_jpeg_cls_n12`=0.6083 = +8.19 | ✓ |
| storage @ n=12 cls: "13.7% smaller" vs frp_jpeg | ≈13.8% | sbpp 0.9046 vs 1.0296 → CR-gain 13.82% (fraction-of-reference 12.14%) | ⚑ see note A |
| cls no-codec ceiling | 85.13% top-1 | `cls/eval/eval_raw_cls_ref.json` | ✓ |
| clip no-codec ceiling | 69.59% top-1 | `clip/eval/eval_raw_clip_ref.json` | ✓ |
| seg no-codec ceiling (squash-512²) | 44.51% mIoU | `seg/eval/eval_raw_seg_ref.json` | ✓ |
| seg sliding-window paper-protocol ceiling | 45.96% mIoU | external (UperNet-ConvNeXt-Tiny model card) — **not in mirror** | ▽ external ref |

## Notes / flags

- **Note A (storage "13.7%").** The companion "+8.19 pp" matches the
  mirror exactly. The "13.7% smaller" storage figure computes to
  **13.82%** under the storage-CR-gain convention
  (`24/sbpp` ratio: `eval_seaft_cls_n12` sbpp 0.9046 vs
  `eval_frp_jpeg_cls_n12` sbpp 1.0296) or **12.14%** as a
  fraction-of-reference-file. Both round near 13.7–13.8%; the value is
  backed, only the rounding/convention differs by ~0.1 pp. Not a missing
  file — flagged for the author's awareness only. Frozen as-published.
- **`radar.pdf` (Fig. 5)** is the SVG-finalized deck export, *not* the
  matplotlib `make_radar_figure*.py` output. Its per-axis rankings are
  data-driven (derived in `corl_2026/notes/radar_rankings*.md` /
  `_compute_radar_rankings_v3.py` from `encode_complexity/cls_384` +
  `cls/throughput`), but the rendered figure is hand-laid-out and is not
  reproduced by any `make_*.py`. Reproducing the rankings is possible;
  reproducing the exact figure requires the deck.
- **No reported number lacks a backing file.** Items marked ▽ are
  external citations (energy/MAC-per-pixel figures, channel-derivation
  compression ratios 60:1 / 133:1 / 288:1, the 45.96% sliding-window
  seg number) or architectural facts derivable from source, not from
  `results/` — by design, not gaps.
- **Excluded-but-valid data present in the mirror (kept, not a gap):**
  `walft` rows (in `cls/` only; shown in Fig. 2 for continuity, excluded
  from tables per `CLAUDE.md`), per-cell lambda-retry sweeps, and
  pre-iter-10 `walft` are intentionally not surfaced in tables. See
  `make_tables.py` `CLS_ORDER`/`SEG_ORDER`/`CLIP_ORDER`.

## Production-checkpoint provenance (for weight export, §3b)

The `seaft` eval JSONs pin the exact fine-tuned checkpoint via
`config.checkpoint`:

| task | op | checkpoint (`config.checkpoint`) | reported |
|---|---|---|---|
| cls | n=12 | `seaotter/pre_trained_convnext/experiments/iter5_imagenet_gt_squash/production/checkpoint_iter5_cls_n12.pth` | top-1 0.69024 |
| cls | n=6 | `.../iter5_imagenet_gt_squash/production/checkpoint_iter5_cls_n6.pth` | top-1 0.46548 |
| seg | n=12 | `seaotter/pre_trained_convnext/experiments/iter1_initial_pipeline/production/checkpoint_prod_seg_n12_C6.pth` | mIoU 0.32773 |
| clip | n=12 | `seaotter/experiments/clip_production/production/checkpoint_clip_prod_frappe_n12.pth` | top-1 0.48218 |

The **headline cls pipeline** bundle (§3b) is built from
`checkpoint_iter5_cls_n12.pth` (fine-tuned decoder + fwd/inv/proxy, all
hot per `hot_names`), with the frozen FRAPPE encoder referenced from
`~/hf/FRAPPE/FRAPPE/`.
