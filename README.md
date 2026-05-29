# SEAOTTER

## Sensor Embedded Autoencoding with One Time Transcode for Efficient Reconstruction

[project page](https://UT-SysML.github.io/seaotter)

[paper](https://danjacobellis.net/_static/seaotter.pdf)

In robotics, wearables, and remote sensing, an incredible wealth of visual data is easily captured at high resolution, high frame rate, and using low cost and low power hardware.
However, limited bandwidth and on-device processing power severely limit our ability to utilize these signals with standard JPEG and MPEG codecs.
Newer standards, like AV1/AVIF, provide better rate-distortion trade-off, but require significantly more resources to encode on device, making them impractical to use without custom ASICs or hardware accelerators.
Recent asymmetric neural network-based autoencoder systems deliver high quality under extreme power and bandwidth constraints, but introduce difficulties for downstream applications due to (1) their prohibitive decoding cost and (2) their use of bespoke formats that ignore decades of infrastructure built around standards like JPEG.
To address these limitations, we introduce a compression framework for cloud robotics based on a **S**ensor **E**mbedded **A**utoencoder paired with a **O**ne-**T**ime **T**ranscode for **E**fficient **R**econstruction (SEAOTTER).
By exploiting heterogeneity of power and bandwidth constraints across processing stages in cloud robotics systems, SEAOTTER bridges the gap between compact, but difficult to utilize, latent representations and higher bitrate, but universally supported, JPEG bitstreams.
SEAOTTER improves the accuracy of global, dense, and VLM/VLA-based remote inference pipelines compared to modern standards like AVIF.
Remarkably, SEAOTTER's additional transcode step *increases* accuracy compared to the same DNN-based autoencoder without it.
At the same time, SEAOTTER seamlessly integrates with the ecosystem of hardware and software systems built around the JPEG standard, and can be optimized end-to-end for specific sensors, environments, or downstream models.

## Code, pretrained models, and reproduction

This repository also ships the code and data to reproduce the paper:

- **`src/seaotter/`** — the `seaotter` v1.0.0 package (JPEG codec, learned color/quantization sandwich, fine-tunable pipeline, training recipes, encoder-throughput harness). Install with `pip install seaotter==1.0.0` or `pip install .` from this repo.
- **Pretrained pipeline, one call:**

  ```python
  from seaotter import load_pipeline_from_hub
  pipe = load_pipeline_from_hub(subdir="seaotter_cls")  # headline ImageNet pipeline
  jpeg = pipe.transcode(image_uint8)   # cloud one-time transcode -> JPEG bytes
  rgb  = pipe.decode(jpeg)             # consumer steady-state decode -> uint8 RGB
  ```

- **`results/`** — every per-operating-point JSON behind the paper's figures and tables (`results.md` schema; `TRACEABILITY.md` number→file audit).
- **`paper_figures/`** — the generators that turn `results/` into the paper figures/tables.
- **`experiments/`** — the late-paper sweep harnesses and findings.

See [`REPRODUCE.md`](REPRODUCE.md) for installation, loading the pretrained models, the dataset list, and a full result→command map.
