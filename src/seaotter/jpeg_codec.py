"""Custom-quantization JPEG codec for already-decorrelated 3-channel data.

Pillow's JPEG encoder is configurable enough to do exactly what we need with
no custom extension:

- the input image is treated as already in YCbCr space (no RGB->YCbCr
  conversion at encode time, no YCbCr->RGB at decode time);
- chroma is **not** subsampled (4:4:4);
- the 8x8 quantization matrix is fully user-supplied (typically a single
  matrix shared across all three components, but per-component is also
  supported).

This module wraps the PIL plumbing in two functions that take and return
3xHxW uint8 torch tensors. Encoding shifts no values; decoding uses
``Image.draft('YCbCr', size)`` so libjpeg emits raw YCbCr samples instead
of running its built-in colorspace transform.
"""

from __future__ import annotations

import io
from typing import Sequence, Union

import numpy as np
import PIL.Image
import torch


# A qtable can be supplied as a torch tensor, numpy array, or a Python sequence
# of ints. Acceptable shapes (after coercion):
#   - (64,)  or (8, 8)         : one matrix shared across all three components
#   - (2, 64) or (2, 8, 8)     : luma + chroma (Y, then Cb=Cr)
#   - (3, 64) or (3, 8, 8)     : independent matrices for Y, Cb, Cr
QTableLike = Union[torch.Tensor, np.ndarray, Sequence]


def _to_qtable_lists(qtable: QTableLike) -> list[list[int]]:
    """Coerce ``qtable`` into the ``list[list[int]]`` form Pillow expects.

    Values are rounded to the nearest integer and clamped to ``[1, 255]``
    (baseline JPEG stores 8-bit qtable entries; 0 would mean "divide by zero"
    and is rejected by libjpeg). Order is raster (left-to-right, top-to-bottom
    within an 8x8 block) — Pillow handles the zigzag conversion internally.
    """
    if isinstance(qtable, torch.Tensor):
        arr = qtable.detach().cpu().numpy()
    else:
        arr = np.asarray(qtable)

    arr = np.rint(arr).astype(np.int64)
    arr = np.clip(arr, 1, 255)

    if arr.ndim == 1:
        if arr.shape[0] != 64:
            raise ValueError(f"1D qtable must have 64 elements, got {arr.shape[0]}")
        tables = [arr]
    elif arr.ndim == 2:
        if arr.shape == (8, 8):
            tables = [arr.reshape(64)]
        elif arr.shape[1] == 64 and arr.shape[0] in (1, 2, 3):
            tables = [arr[i] for i in range(arr.shape[0])]
        else:
            raise ValueError(
                f"2D qtable must be (8,8) or (n,64) with n in {{1,2,3}}, "
                f"got {arr.shape}"
            )
    elif arr.ndim == 3:
        if arr.shape[1:] == (8, 8) and arr.shape[0] in (1, 2, 3):
            tables = [arr[i].reshape(64) for i in range(arr.shape[0])]
        else:
            raise ValueError(
                f"3D qtable must be (n,8,8) with n in {{1,2,3}}, got {arr.shape}"
            )
    else:
        raise ValueError(f"qtable must be 1D, 2D, or 3D; got ndim={arr.ndim}")

    return [t.astype(int).tolist() for t in tables]


def encode(image: torch.Tensor, qtable: QTableLike, *, subsampling: int = 0) -> bytes:
    """Encode a 3xHxW uint8 tensor as JPEG with no color conversion.

    The tensor's three channels are written to the JPEG file as Y, Cb, Cr
    components, byte-for-byte. ``decode`` is the exact inverse modulo the
    DCT round-trip and the ``qtable`` quantization. No RGB conversion is
    performed at any stage.

    Args:
        image: 3xHxW uint8 tensor. Channel order is whatever you want it to
            be — it just gets written into the Y, Cb, Cr slots in order.
        qtable: 8x8 quantization matrix. See :data:`QTableLike` for accepted
            shapes. Values are rounded and clamped to [1, 255].
        subsampling: chroma subsampling code. ``0`` (default) is 4:4:4, i.e.
            no subsampling — required if the channels aren't really chroma.
            ``1`` is 4:2:2, ``2`` is 4:2:0.

    Returns:
        JPEG byte string.
    """
    if image.dtype != torch.uint8:
        raise TypeError(f"image must be uint8, got {image.dtype}")
    if image.dim() != 3 or image.shape[0] != 3:
        raise ValueError(f"image must be 3xHxW, got shape {tuple(image.shape)}")

    tables = _to_qtable_lists(qtable)
    arr = image.detach().cpu().permute(1, 2, 0).contiguous().numpy()
    pil = PIL.Image.fromarray(arr, mode="YCbCr")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", qtables=tables, subsampling=subsampling)
    return buf.getvalue()


def decode(jpeg_bytes: bytes) -> torch.Tensor:
    """Decode JPEG bytes back to a 3xHxW uint8 tensor.

    Uses ``Image.draft('YCbCr', size)`` to keep libjpeg in YCbCr output mode,
    so the returned tensor contains the decoded Y, Cb, Cr samples directly
    with no YCbCr->RGB transform. Round-trip with :func:`encode` is bit-exact
    when the source has only DC content and the qtable's DC entry is 1.
    """
    img = PIL.Image.open(io.BytesIO(jpeg_bytes))
    img.draft("YCbCr", img.size)
    img.load()
    if img.mode != "YCbCr":
        raise RuntimeError(
            f"expected YCbCr-mode load via draft(), got {img.mode}; "
            "this usually means the JPEG was encoded with a non-JFIF color "
            "marker and libjpeg refused to skip color conversion."
        )
    arr = np.array(img)  # HxWx3 uint8, owned (writable) copy
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def read_qtables(jpeg_bytes: bytes) -> dict[int, list[int]]:
    """Return the quantization tables stored in the JPEG, in raster order.

    Useful for verifying that an arbitrary qtable made it into the file
    verbatim (Pillow stores the values exactly as provided).
    """
    img = PIL.Image.open(io.BytesIO(jpeg_bytes))
    return {k: list(v) for k, v in img.quantization.items()}
