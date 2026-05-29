"""8x8 forward / inverse DCT in torch matching JPEG/JFIF normalization.

Adapted from torchjpeg (https://github.com/Queuecumber/torchjpeg) with modifications by Dan Jacobellis.
The author and copyright holder for the the original version is Max Ehrlich (MIT License). 
The following two files in torchjpeg were accessed on 5/8/2026 as the starting point:
torchjpeg/src/torchjpeg/dct/_block.py and
torchjpeg/src/torchjpeg/dct/__init__.py
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def blockify(im: Tensor, size: int) -> Tensor:
    bs = im.shape[0]
    ch = im.shape[1]
    h = im.shape[2]
    w = im.shape[3]

    im = im.reshape(bs * ch, 1, h, w)
    im = F.unfold(im, kernel_size=(size, size), stride=(size, size))
    im = im.transpose(1, 2)
    im = im.reshape(bs, ch, -1, size, size)

    return im


def deblockify(blocks: Tensor, size: Tuple[int, int]) -> Tensor:
    bs = blocks.shape[0]
    ch = blocks.shape[1]
    block_size = blocks.shape[3]

    blocks = blocks.reshape(bs * ch, -1, int(block_size ** 2))
    blocks = blocks.transpose(1, 2)
    blocks = F.fold(
        blocks,
        output_size=size,
        kernel_size=(block_size, block_size),
        stride=(block_size, block_size),
    )
    blocks = blocks.reshape(bs, ch, size[0], size[1])

    return blocks


def _normalize(N: int) -> Tensor:
    n = torch.ones((N, 1))
    n[0, 0] = 1 / math.sqrt(2)
    return n @ n.t()


def _harmonics(N: int) -> Tensor:
    spatial = torch.arange(float(N)).reshape((N, 1))
    spectral = torch.arange(float(N)).reshape((1, N))

    spatial = 2 * spatial + 1
    spectral = (spectral * math.pi) / (2 * N)

    return torch.cos(spatial @ spectral)


def block_dct(blocks: Tensor) -> Tensor:
    N = blocks.shape[3]
    n = _normalize(N).to(blocks.device).to(blocks.dtype)
    h = _harmonics(N).to(blocks.device).to(blocks.dtype)

    return (2 / N) * n * (h.t() @ blocks @ h)


def block_idct(coeff: Tensor) -> Tensor:
    N = coeff.shape[3]
    n = _normalize(N).to(coeff.device).to(coeff.dtype)
    h = _harmonics(N).to(coeff.device).to(coeff.dtype)

    return (2 / N) * (h @ (n * coeff) @ h.t())


def batch_dct(batch: Tensor) -> Tensor:
    size = (batch.shape[2], batch.shape[3])
    im_blocks = blockify(batch, 8)
    dct_blocks = block_dct(im_blocks)
    return deblockify(dct_blocks, size)


def batch_idct(coeff: Tensor) -> Tensor:
    size = (coeff.shape[2], coeff.shape[3])
    dct_blocks = blockify(coeff, 8)
    im_blocks = block_idct(dct_blocks)
    return deblockify(im_blocks, size)


if __name__ == "__main__":
    torch.manual_seed(0)

    # 1. Random round-trip: idct(dct(x)) == x to numerical precision.
    x = torch.randn(2, 3, 64, 64)
    y = batch_idct(batch_dct(x))
    rt_err = (x - y).abs().max().item()
    print(f"[_dct] round-trip max abs err: {rt_err:.3e}")
    assert rt_err < 1e-4, rt_err

    # 2. DC scaling: a constant block of value v -> coeff[0, 0] == 8 * v,
    #    everything else 0. This is N * v with N=8, since (2/N)*alpha(0)^2*N^2*v
    #    = (2/N) * (1/2) * N^2 * v = N v.
    block = torch.full((1, 1, 1, 8, 8), 5.0)
    coeff = block_dct(block)
    assert abs(coeff[0, 0, 0, 0, 0].item() - 8 * 5.0) < 1e-5
    coeff[0, 0, 0, 0, 0] = 0.0
    assert coeff.abs().max().item() < 1e-5
    print("[_dct] DC scaling check passed (block of 5 -> DC == 40, AC == 0)")

    # 3. Zero-mean block: an 8x8 block that's all zeros (i.e. level-shifted
    #    constant 128 in the codec) yields all-zero DCT.
    zero_block = torch.zeros(1, 1, 1, 8, 8)
    z_coeff = block_dct(zero_block)
    assert z_coeff.abs().max().item() < 1e-6
    print("[_dct] zero-block check passed")

    print("[_dct] all smoke tests passed")
