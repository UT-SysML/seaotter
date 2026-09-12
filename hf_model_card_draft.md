---
license: mit
library_name: seaotter
pipeline_tag: image-to-image
tags:
- image-compression
- neural-compression
- jpeg
- asymmetric-codec
- split-inference
datasets:
- danjacobellis/LSDIR
- danjacobellis/kodak
- timm/imagenet-1k-wds
- danjacobellis/scene_parse_150
base_model:
- danjacobellis/FRAPPE
- danjacobellis/walloc
---

# SEAOTTER

**Sensor Embedded Autoencoding with One-Time Transcode for Efficient Reconstruction.**
Dan Jacobellis and Neeraja J. Yadwadkar (UT Austin). Paper: [arXiv:2605.28992](https://arxiv.org/abs/2605.28992). Code: [github.com/UT-SysML/seaotter](https://github.com/UT-SysML/seaotter). Package: `pip install seaotter`.

SEAOTTER is a split-inference compression framework for robotics, wearable, and remote-sensing systems. A tiny frozen neural encoder (FRAPPE or WaLLoC) runs on the sensor. A large fine-tuned neural decoder plus a learned "JPEG sandwich" (a learned 3x3 color transform `F`, a learned 8x8 quantization table, and a learned inverse transform `F^-1`) runs **once** in the cloud and transcodes the uplink latents into a standards-compliant JPEG. Every downstream consumer (training-data loaders, viewers, edge devices) then decodes a plain JPEG and applies the light inverse transform. The decoder and sandwich are fine-tuned per downstream task so the stored JPEG preserves what the task model needs, not what a human would rate highly. PSNR against the original is therefore low by design.

## What is in this repo

Two kinds of bundles, both loaded by the `seaotter` package.

### Zero-shot transcoder bundles (schema `seaotter-jpeg-v1`)

The shared color pair `(F, F^-1)` and `K=3` learned qtables from joint training on LSDIR, with no fine-tuned neural decoder. Used as the warm start for every fine-tuned pipeline below and as the zero-shot "SEAOTTER-ZS" rows in the paper.

| subdir | contents |
|---|---|
| `seaotter_jpeg_s3` | default; R16 dual-goal sister checkpoint, lambda = [0.75, 0.40, 0.22]. Warm start for every fine-tuned pipeline. |
| `seaotter_jpeg` | R16 dual-goal champion. |

```python
from seaotter import load_from_hub
bundle = load_from_hub()                      # seaotter_jpeg_s3
jpeg_bytes = bundle.encode(image_uint8, k=1)  # image: (3,H,W) uint8 tensor
rgb = bundle.decode(jpeg_bytes)
```

### Fine-tuned task pipelines (schema `seaotter-pipeline-v1`)

One bundle per `(task, upstream codec, operating point)` evaluated in the paper. Each holds the fine-tuned upstream decoder (about 57 M parameters for FRAPPE, WaLLoC's decoder for the `walloc` bundles), the fine-tuned color pair, and the fine-tuned qtable, stored as fp32 safetensors. The frozen sensor-side encoder is not duplicated. It is fetched from `danjacobellis/FRAPPE` or `danjacobellis/walloc` at load time.

```python
from seaotter import load_pipeline_from_hub
pipe = load_pipeline_from_hub(subdir="seaotter_seg_n12")
jpeg_bytes = pipe.transcode(image_uint8)   # cloud, one-time: encoder -> decoder -> F -> JPEG
rgb = pipe.decode(jpeg_bytes)              # consumer, per fetch: JPEG decode -> F^-1
rgb = pipe.reconstruct(image_uint8)        # both steps
```

Operating point: `n` is the number of FRAPPE latent channels kept (progressive codec), `p` is the WaLLoC pixel ratio in percent. Transmit bpp is the sensor uplink rate, storage bpp is the on-disk JPEG rate. Accuracy is on the frozen downstream backbone from the paper (ConvNeXt-Tiny top-1 on ImageNet-1k for `cls`, UperNet-ConvNeXt-Tiny mIoU on ADE20K for `seg`, SigLIP-2 zero-shot ImageNet top-1 for `clip`). PSNR is against the uncompressed image. All numbers are copied from the paper's evaluation JSONs and are also stored in each bundle's `config.json` under `reference_metrics`.

| subdir | task | operating point | transmit bpp | storage bpp | accuracy | PSNR (dB) |
|---|---|---|---|---|---|---|
| `seaotter_cls` | cls | FRAPPE n=12 | 0.109 | 0.905 | top-1 0.690 | 10.4 |
| `seaotter_cls_n3` | cls | FRAPPE n=3 | 0.012 | 1.616 | top-1 0.173 | 12.2 |
| `seaotter_cls_n6` | cls | FRAPPE n=6 | 0.038 | 1.224 | top-1 0.465 | 12.2 |
| `seaotter_cls_n9` | cls | FRAPPE n=9 | 0.064 | 1.135 | top-1 0.597 | 11.6 |
| `seaotter_cls_n12` | cls | FRAPPE n=12 | 0.109 | 0.905 | top-1 0.690 | 10.4 |
| `seaotter_cls_n15` | cls | FRAPPE n=15 | 0.344 | 0.807 | top-1 0.774 | 9.8 |
| `seaotter_cls_walloc_p4` | cls | WaLLoC p=4 | 0.043 | 2.250 | top-1 0.391 | 17.8 |
| `seaotter_cls_walloc_p16` | cls | WaLLoC p=16 | 0.144 | 2.621 | top-1 0.693 | 21.7 |
| `seaotter_cls_walloc_p36` | cls | WaLLoC p=36 | 0.260 | 2.151 | top-1 0.761 | 20.2 |
| `seaotter_cls_walloc_p80` | cls | WaLLoC p=80 | 0.531 | 1.347 | top-1 0.797 | 11.2 |
| `seaotter_cls_walloc_p100` | cls | WaLLoC p=100 | 0.677 | 1.244 | top-1 0.806 | 10.1 |
| `seaotter_seg_n3` | seg | FRAPPE n=3 | 0.010 | 0.457 | mIoU 0.067 | 10.1 |
| `seaotter_seg_n6` | seg | FRAPPE n=6 | 0.031 | 0.435 | mIoU 0.211 | 10.6 |
| `seaotter_seg_n9` | seg | FRAPPE n=9 | 0.055 | 0.473 | mIoU 0.282 | 10.8 |
| `seaotter_seg_n12` | seg | FRAPPE n=12 | 0.094 | 0.524 | mIoU 0.328 | 10.8 |
| `seaotter_seg_n15` | seg | FRAPPE n=15 | 0.304 | 0.683 | mIoU 0.386 | 12.2 |
| `seaotter_clip_n3` | clip | FRAPPE n=3 | 0.019 | 0.426 | top-1 0.026 | 9.5 |
| `seaotter_clip_n6` | clip | FRAPPE n=6 | 0.054 | 0.524 | top-1 0.200 | 12.7 |
| `seaotter_clip_n9` | clip | FRAPPE n=9 | 0.083 | 0.588 | top-1 0.348 | 13.4 |
| `seaotter_clip_n12` | clip | FRAPPE n=12 | 0.142 | 0.645 | top-1 0.482 | 13.1 |
| `seaotter_clip_n15` | clip | FRAPPE n=15 | 0.410 | 0.798 | top-1 0.613 | 12.8 |

`seaotter_cls` is the headline classification pipeline and is byte-identical to `seaotter_cls_n12`. It is kept under both names so the short name used in the paper's reproduction instructions keeps working.

## Bundle layout

```
<subdir>/
  config.json                        # schema, task, codec, operating point, lambda, reference_metrics, provenance
  <subdir>_pytorch_model.safetensors # decoder.* / fwd.* / inv.* / proxy.* / qtable
```

`qtable` is the deployed integer `(3, 8, 8)` quantization table, rounded once outside the model from the learned continuous parameterization in `proxy.*`. JPEGs are written 4:4:4 with libjpeg's standard Huffman tables, and the three JPEG channels carry the learned transform's output rather than YCbCr.

## Training data

The zero-shot transcoders were trained on LSDIR. Fine-tuned pipelines were trained on LSDIR (`seg`) or the ImageNet-1k training split (`cls`, `clip`), with distillation targets from the frozen downstream backbone. Evaluation used ImageNet-1k validation (50k images), ADE20K validation (2k images), and Kodak. The qtable and color transforms were learned from data alone with no warm start from JPEG's standard tables or the JFIF color matrix.

## Requirements

`seaotter` needs `torch`, `safetensors`, `huggingface_hub`, `pillow`, `gigatorch`, and `compressors` (for the FRAPPE and WaLLoC upstream codecs).

## Citation

```bibtex
@article{jacobellis2026seaotter,
  title={SEAOTTER: Sensor Embedded Autoencoding with One-Time Transcode for Efficient Reconstruction},
  author={Jacobellis, Dan and Yadwadkar, Neeraja J},
  journal={arXiv preprint arXiv:2605.28992},
  year={2026}
}
```
