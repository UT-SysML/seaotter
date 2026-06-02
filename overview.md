# SEAOTTER

## Sensor Embedded Autoencoding with One Time Transcode for Efficient Reconstruction

[code](https://github.com/UT-SysML/seaotter)

[paper](https://danjacobellis.net/_static/seaotter.pdf)

In robotics systems, vast amounts of visual data are easily captured at high resolution using low-cost, low-power hardware.
Yet, limited bandwidth and on-device compute resources prevent full utilization when transmitted via conventional codecs like JPEG/MPEG.
Newer codecs, like AV1/AVIF, improve the rate-distortion trade-off, but demand far more resources for encoding, impractical without custom ASICs.
Recent asymmetric autoencoders deliver high quality under extreme power and bandwidth constraints, but add prohibitive decoding cost and use bespoke formats that ignore decades of infrastructure built around standards like JPEG.
To address these limitations, we introduce a compression framework for cloud robotics based on a **S**ensor **E**mbedded **A**utoencoder paired with a **O**ne-**T**ime **T**ranscode for **E**fficient **R**econstruction (SEAOTTER).
Because the sensor, cloud, and consumer stages face very different power and bandwidth budgets, SEAOTTER combines the compactness of a learned latent with the broad usability of a standard JPEG file.
Since naive transcoding degrades performance, we propose a learnable JPEG color and quantization transform that enables increased accuracy for global, dense, and vision-language-based perception.
Using SEAOTTER, we train both general-purpose and task-aware transcoding pipelines for a pre-trained, frozen encoder.
At a compression ratio of 200:1 and compared to AVIF, we observe 7× faster encoding, 3.5× faster decoding, and +8% ImageNet top-1 accuracy, while retaining compatibility with JPEG infrastructure.

```{figure} img/system.svg
---
name: system
---
Overview of SEAOTTER design and workflow.
```

```{figure} img/radar.svg
---
name: radar
---
Performance trade-offs of SEAOTTER variants vs other codecs.
```
