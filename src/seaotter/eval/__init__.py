"""Encoder-only throughput evaluation harness for the SEAOTTER CoRL paper.

Provides a unified `seaotter.eval.encode_complexity` entry point that
measures **encoder forward + native quant/entropy coding only**, with
the input pre-staged in the encoder's native form (PIL.Image for
libavif/libjpeg/WebP/JP2; float tensor for FRAPPE/WaLLoC; uint8 RGB
tensor for SEAOTTER standalone JPEG).

Methodology mirrors the original FRAPPE harness at
``compressors/src/compressors/frappe/evaluate_encode_complexity.py``:
``torch.inference_mode()``, ``n_warmup=1``, ``n_measurement=5``,
median across all (image x pass) timings, sum of per-stage medians
divides into pixels-per-image to give the reported MPx/s.
"""

"""Public re-exports are lazy: importing submodules is a side-effect of
attribute access, not of importing the parent. This avoids the
"module found in sys.modules before execution" warning under
``python -m seaotter.eval.encode_complexity``."""

__all__ = ["encode_complexity", "codecs", "datasets"]
