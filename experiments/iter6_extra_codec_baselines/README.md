# iter-6 — standardized paper-eval harness

Spec: [`../../prompts/iter6_standardized_eval_harness.md`](../../prompts/iter6_standardized_eval_harness.md).

Standardized eval harness producing all numbers needed for the paper's
R-Acc / R-D / throughput tables in one unified pass. Replaces ad-hoc
iter-3 / iter-4 evals with a single harness whose output JSONs share
one schema, share one set of preprocessing rules, and are produced by
one master `eval.sh` recording the exact commands run.

## Files

| path                | role                                                                     |
|---------------------|--------------------------------------------------------------------------|
| `eval.sh`           | **Master record** — re-runnable bash script with every command verbatim. |
| `harness/`          | Shared library: schema, preprocessing, metrics, teacher, pipelines.       |
| `eval_accuracy.py`  | CLI: one (pipeline × op × task) accuracy + distortion cell, full val.     |
| `eval_throughput.py`| CLI: one cell on a 256-image subset for encode + consumer timings.        |
| `launch_iter6.py`   | 4-GPU dispatcher; parses `eval.sh` and runs cells in parallel.            |
| `make_findings.py`  | Aggregates `production/*.json` into `findings.md`.                        |
| `production/`       | Output JSONs (`eval_<pipeline>_<task>_<opid>.json`, etc).                 |
| `logs/`             | Per-cell stdout.                                                          |

## Running

```bash
# Re-run any specific cell:
bash eval.sh                  # full sweep, slow

# Or via the dispatcher (4 GPU accuracy, 1 GPU throughput):
python launch_iter6.py --kind accuracy
python launch_iter6.py --kind throughput

# Then aggregate:
python make_findings.py
```

The launcher sets `LD_LIBRARY_PATH` so torch 2.11+cu130 can load
`libnvrtc-builtins.so.13.0` from the cu13 pip wheel (needed for WaLLoC's
elementwise post-processing kernels).

## Pipelines

| short | label                                       | op-knob          |
|-------|---------------------------------------------|------------------|
| jpeg  | Vanilla JPEG (subsampling=0)                | quality (1..100) |
| jp2   | Vanilla JPEG 2000                           | rate (compr ratio) |
| webp  | Vanilla WebP (lossy)                        | quality          |
| avif  | Vanilla AVIF (libavif default speed)        | quality          |
| avifx | Vanilla AVIF (libavif speed=10, fastest)    | quality          |
| wal   | WaLLoC RGB_16x                              | pixel ratio (q%) |
| frp   | FRAPPE-only (no transcode; iter-3 S3)       | n_ch             |
| seab  | SEA OTTER pipeline, no fine-tuning          | n_ch             |
| seaft | SEA OTTER pipeline, fine-tuned              | n_ch             |

`seaft` checkpoint asymmetry: cls uses iter-5; seg uses iter-1.
