# Reproducing the SEAOTTER (CoRL 2026) results

Public code + data release for *"SEAOTTER: Sensor Embedded Autoencoding
with One-Time Transcode for Efficient Reconstruction"* (Jacobellis &
Yadwadkar). Numerical results are **frozen as-published**; the scripts
here regenerate the paper's figures and tables from the committed data
without re-running any evaluation.

## What's in this repo

| path | what |
|------|------|
| `src/seaotter/` | the `seaotter` **v1.0.0** package — JPEG codec, learned color/quant sandwich, fine-tunable pipeline, training recipes (`train/recipes/`), encoder-throughput harness (`eval/`), and the **downstream accuracy + consumer-throughput eval harness** (`eval/accuracy/` — the code that produced every `eval_*.json` / `throughput_*.json`). Mirrors PyPI `seaotter==1.0.0`. |
| `results/` | every per-operating-point JSON behind the paper (`results.md` = schema; `TRACEABILITY.md` = number→file audit; `DELETIONS.md` = prune log). |
| `paper_figures/` | the `make_*.py` generators that turn `results/` into the paper's figures/tables (+ `notes/` radar derivations). |
| `experiments/` | the late-paper sweep harnesses (`lambda_sweep`, `storage_5line`, `wal_seg_clip`, `clip_production`, `codec_kodak_eval`) with their `launch.py` / `run_*.py` / `check_acceptance.py` / `findings.md`, plus the `iter6_extra_codec_baselines/` writeup (`findings.md` + `README.md`). The accuracy/consumer-throughput **harness itself** now lives in the package at `src/seaotter/eval/accuracy/`. Heavy outputs/checkpoints/logs excluded. |
| `tools/export_pipeline_bundle.py` | rebuilds the pretrained pipeline bundle from a production checkpoint. |

## Install

```bash
pip install seaotter==1.0.0      # canonical (PyPI)
# or, from this mirror:
pip install .                    # builds src/seaotter
```

Runtime dependencies: `torch`, `safetensors`, `huggingface_hub`,
`Pillow`, `numpy`; plus the authors' **`gigatorch`** (color-transform
ops in `color_transform.py`) and **`compressors`** (FRAPPE / WaLLoC
codec loaders used by the full pipeline and the eval harnesses).

## Load the pretrained models (one call)

```python
# Headline ImageNet-classification pipeline: frozen FRAPPE encoder +
# fine-tuned decoder (G_S) + learned JPEG sandwich. Reference top-1
# 0.69024 at transmit-bpp 0.10860 (CR 221:1).
from seaotter import load_pipeline_from_hub
pipe = load_pipeline_from_hub(subdir="seaotter_cls")   # danjacobellis/seaotter
jpeg = pipe.transcode(image_uint8)   # cloud one-time transcode -> JPEG bytes
rgb  = pipe.decode(jpeg)             # consumer steady-state decode -> uint8 RGB
rgb  = pipe.reconstruct(image_uint8) # end-to-end (== decode(transcode(.)))

# Warm-start transcoder bundle only (shared color pair + K qtables):
from seaotter import load_from_hub
bundle = load_from_hub()             # seaotter_jpeg_s3 (default), λ=[0.75,0.40,0.22]
```

The frozen FRAPPE encoder is pulled separately from
`danjacobellis/FRAPPE` (never duplicated in the SEAOTTER bundle).

## Datasets

| task | dataset |
|------|---------|
| cls / clip (ImageNet val 50k) | `timm/imagenet-1k-wds` |
| seg (ADE20K val 2k) | `danjacobellis/scene_parse_150` |
| standalone codec (Kodak 24) | `danjacobellis/kodak` |
| de novo sandwich training | `danjacobellis/LSDIR` (train) |

## Regenerate the paper figures & tables

The generators hard-code `ROOT =
Path("/home/dgj335/UT-SysML/seaotter/results")`. If this repo lives
elsewhere, edit `ROOT` (or symlink) to point at this repo's `results/`.

```bash
python paper_figures/make_tables.py         # Tables 1-8 (.tex)
python paper_figures/make_figures.py         # Fig 2  main_results.pdf
python paper_figures/make_extra_figures.py   # Fig 4  rd_metric_panels, Fig 8 rd_storage
python paper_figures/make_codec_kodak_fig.py # Fig 7  codec_kodak.pdf
```

Figs 1 (system), 3 (jpeg sandwich), 5 (radar), 6 (quant) are
hand-laid-out diagrams (slide-deck / notebook sources), not regenerated
from `results/`; the radar's per-axis rankings are derived in
`paper_figures/notes/`.

## Result → source map

Full audit in [`results/TRACEABILITY.md`](results/TRACEABILITY.md).
Condensed:

| paper artifact | regenerate / data |
|----------------|-------------------|
| Table 1 (headline) | `make_tables.py` ← `{cls,seg,clip}/eval` + `cls/throughput` + `encode_complexity/{cls_384,seg_512,clip_naflex}` |
| Tables 4–6 (per-task RD) | `make_tables.py` ← `{cls,seg,clip}/eval` |
| Table 7 (throughput) | `make_tables.py` ← `cls/{eval,throughput}` + `encode_complexity/cls_384` |
| Table 8 (deployment tiers) | `make_tables.py` ← `cls/eval` + `encode_complexity/cls_384` |
| Tables 2–3 (Kodak codec) | `make_tables.py` ← `codec_kodak/`, `codec_kodak_cls/` |
| Fig 2 (main results) | `make_figures.py` ← `cls/eval` + `encode_complexity/cls_384` |
| Fig 4 (RD metrics) | `make_extra_figures.py` ← `{cls,seg,clip}/eval` |
| Fig 7 (Kodak codec) | `make_codec_kodak_fig.py` ← `codec_kodak/`, `codec_kodak_cls/` |
| Fig 8 (storage RD) | `make_extra_figures.py` ← `{cls,seg,clip}/eval` (`storage_bpp`) |
| headline accuracy deltas | `cls/eval/eval_{seaft,frp,seab}_cls_n{6,12}.json` |
| Kodak PSNR margins | `codec_kodak/eval_{seaotter,jpeg_sub0}_kodak_*.json` |

## Where the result JSONs come from (producing harnesses)

The `results/` JSONs that the figures/tables read were produced by:

| JSON | producing harness | version |
|------|-------------------|---------|
| `{cls,seg,clip}/eval/eval_*.json` (accuracy + distortion + bpp) | `python -m seaotter.eval.accuracy.eval_accuracy` | `iter6-1` |
| `cls/throughput/throughput_*.json` (CPU consumer decode) | `python -m seaotter.eval.accuracy.cpu_throughput` | `iter11-cpu-2` |
| `encode_complexity/<dataset>/encode_*.json` (encoder-only throughput) | `python -m seaotter.eval.launch_load_bearing` | compressors-style |
| `codec_kodak{,_cls}/eval_*.json` (standalone codec) | `experiments/codec_kodak_eval/run_{eval,cls_eval}.py` | `codec-kodak-v1` / `-cls-v1` |

The accuracy harness (`src/seaotter/eval/accuracy/`) drives each codec
through `eval/accuracy/pipelines/*.py` (SEAOTTER's transcode +
steady-state decode live in `pipelines/seaotter.py` + `pipelines/_base.py`).
Re-running needs a GPU plus the task datasets and teachers (ConvNeXt-Tiny,
UperNet, SigLIP-2) and the FRAPPE/WaLLoC codecs (`compressors`).
**Numerical results are frozen** — re-run for audit, not to overwrite the
committed JSONs.

## Re-run the downstream accuracy / consumer-throughput eval (needs GPU + datasets)

The harness that produced every `eval_*.json` (accuracy + distortion +
bpp) and `throughput_*.json` (consumer decode) is packaged under
`seaotter.eval.accuracy`:

```bash
# one accuracy cell -> eval_<pipeline>_<task>_<op>.json
python -m seaotter.eval.accuracy.eval_accuracy \
    --pipeline seaft --task cls --op '{"type":"n_ch","value":12}' \
    --out_json eval_seaft_cls_n12.json

# steady-state consumer-decode throughput -> throughput_*.json
python -m seaotter.eval.accuracy.cpu_throughput --pipeline seaft --task cls \
    --op '{"type":"n_ch","value":12}' --out_json throughput_seaft_cls_n12.json
```

Pipelines: `avif`, `jpeg`, `jp2`, `webp`, `frp`, `wal`, `seab`, `seaft`,
`walsand`, `walft`, `frp_jpeg`, `wal_jpeg`, `raw` (the per-codec
implementations are in `seaotter/eval/accuracy/pipelines/`).

## Re-run a fine-tune recipe (needs GPU + the task dataset)

Six locked recipes (`{frappe,walloc} × {cls,seg,clip}`):

```bash
python -m seaotter.train.recipes.frappe_cls --frappe_n_ch 12 --out_dir /tmp/run --seed 0
```

The headline cls pipeline (`checkpoint_iter5_cls_n12.pth`, top-1
0.69024) warm-starts from the phase-2 K=3 sandwich at `phase2_k=2`,
arch D, then fine-tunes the FRAPPE decoder + sandwich at λ=0.10,
lr_base=2e-4, 1 epoch ImageNet. The de novo sandwich itself
(`seaotter_jpeg_s3`) was trained on LSDIR at 480² for 4 epochs with
λ=(0.75,0.40,0.22), w=(0.3,0.7,1.5). Per-recipe hyperparameters and the
30-cell pass/fail grid are in the source repo's `recipes.md`.

## Re-run encoder-only throughput (CPU; FRAPPE-reference methodology)

```bash
python -m seaotter.eval.launch_load_bearing
# -> results/encode_complexity/<dataset>/encode_<codec>_<op>.json
```

AMD EPYC 9354 single-process, `torch.inference_mode()`, `n_warmup=1`,
`n_measurement=5` (median per stage) — matches the `compressors` FRAPPE
reference harness to within ~1–3%.
