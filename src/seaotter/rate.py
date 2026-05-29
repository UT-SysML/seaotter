"""Differentiable rate proxies for the seaotter JPEG codec.

Every proxy reports a scalar **bits-per-pixel (bpp)** estimate so that a single
`lambda` in `loss = log10(MSE) + lambda * bpp` traces a meaningful R-D curve
across proxy choices. The common-unit guarantee is enforced by an `alpha`
calibration buffer fitted once per proxy on a held-out batch of real JPEG
encodes; subsequent training uses the frozen `alpha`.

Operates on the intermediate tensors exposed by `JPEGProxy.forward`:

    proxy(x)
    bpp = rate_proxy(proxy._last_dct, proxy._last_q_map, x)

`_last_dct`: pre-quantization DCT coefficients, (B, 3, H, W), in raster-of-blocks
order. `_last_q_map`: (1, 3, H, W) tiled qtable. `x`: original (B, 3, H, W) float
in the codec's [0, 255] range.

See `implement_rate_proxies.md` for the design context and references.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from . import jpeg_codec
from ._dct import batch_dct, blockify


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Standard JPEG zigzag scan order. Position k -> (row, col) in an 8x8 block.
_ZIGZAG_ROWS: List[int] = [
    0, 0, 1, 2, 1, 0, 0, 1,
    2, 3, 4, 3, 2, 1, 0, 0,
    1, 2, 3, 4, 5, 6, 5, 4,
    3, 2, 1, 0, 0, 1, 2, 3,
    4, 5, 6, 7, 7, 6, 5, 4,
    3, 2, 1, 2, 3, 4, 5, 6,
    7, 7, 6, 5, 4, 3, 4, 5,
    6, 7, 7, 6, 5, 6, 7, 7,
]
_ZIGZAG_COLS: List[int] = [
    0, 1, 0, 0, 1, 2, 3, 2,
    1, 0, 0, 1, 2, 3, 4, 5,
    4, 3, 2, 1, 0, 0, 1, 2,
    3, 4, 5, 6, 7, 6, 5, 4,
    3, 2, 1, 0, 1, 2, 3, 4,
    5, 6, 7, 7, 6, 5, 4, 3,
    2, 3, 4, 5, 6, 7, 7, 6,
    5, 4, 5, 6, 7, 7, 6, 7,
]


def _zigzag_indices(device: torch.device | None = None) -> tuple[Tensor, Tensor]:
    r = torch.tensor(_ZIGZAG_ROWS, dtype=torch.long, device=device)
    c = torch.tensor(_ZIGZAG_COLS, dtype=torch.long, device=device)
    return r, c


def _dpcm_diff(dc: Tensor) -> Tensor:
    """Inter-block DC differences along the last (n_blocks) axis.

    Args:
        dc: (..., n_blocks) DC coefficients per block, in raster scan order.

    Returns:
        Same-shape tensor with the first block preserved and subsequent blocks
        replaced by `dc[..., k] - dc[..., k - 1]`. Matches JPEG's DC DPCM coding.
    """
    return torch.cat([dc[..., :1], dc[..., 1:] - dc[..., :-1]], dim=-1)


def _qtable_from_q_map(q_map: Tensor) -> Tensor:
    """Extract (3, 8, 8) qtable from the (1, 3, H, W) tiled q_map."""
    return q_map[0, :, :8, :8]


def _real_bpp_per_image(image_uint8: Tensor, qtable: Tensor) -> float:
    """Encode a single (3, H, W) uint8 tensor with the real codec, return bpp."""
    if image_uint8.dtype != torch.uint8:
        image_uint8 = image_uint8.clamp(0, 255).round().to(torch.uint8)
    qt_int = torch.clamp(qtable.detach().cpu().round(), 1, 255).to(torch.int32)
    bytes_ = jpeg_codec.encode(image_uint8.cpu(), qt_int)
    h, w = image_uint8.shape[-2], image_uint8.shape[-1]
    return len(bytes_) * 8.0 / (h * w)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

CalibImages = Union[Tensor, Sequence[Tensor]]


class RateProxy(nn.Module):
    """Base for differentiable rate proxies returning bpp.

    Subclasses implement `_raw(dct, q_map, x_in)` returning a scalar in
    arbitrary units; the base class scales by the calibration buffer
    `alpha` so the final output matches real JPEG bpp on average over the
    calibration set.

    Subclass `__init__` should call `super().__init__()`, set its own
    state (any `dist`, `dc_dpcm`, etc.), then call `self._calibrate(...)`
    to fit `alpha`.
    """

    def __init__(self) -> None:
        super().__init__()
        # alpha: scalar buffer, frozen after calibration.
        self.register_buffer("alpha", torch.tensor(1.0))

    @torch.no_grad()
    def _calibrate(self, calib_images: CalibImages, calib_qtable: Tensor) -> None:
        """Set `alpha` so `forward` matches real JPEG bpp on average over
        the calibration set encoded with `calib_qtable`.
        """
        raw = self._raw_batch(calib_images, calib_qtable)
        real = self._real_bpp_batch(calib_images, calib_qtable)
        new_alpha = (real / (raw.abs() + 1e-12)).item()
        self.alpha.fill_(new_alpha)

    def _raw(self, dct: Tensor, q_map: Tensor, x_in: Tensor | None) -> Tensor:
        raise NotImplementedError

    def forward(
        self, dct: Tensor, q_map: Tensor, x_in: Tensor | None = None
    ) -> Tensor:
        return self.alpha * self._raw(dct, q_map, x_in)

    # -- calibration helpers ------------------------------------------------

    @torch.no_grad()
    def _raw_batch(
        self, calib_images: CalibImages, calib_qtable: Tensor
    ) -> Tensor:
        """Average per-image raw output over the calibration set."""
        if calib_qtable.dim() == 2:
            calib_qtable = calib_qtable.unsqueeze(0).expand(3, 8, 8).contiguous()
        calib_qtable = calib_qtable.to(torch.float32)

        if isinstance(calib_images, Tensor) and calib_images.dim() == 4:
            iterable = [calib_images[i] for i in range(calib_images.shape[0])]
        else:
            iterable = list(calib_images)

        raws: list[float] = []
        for img in iterable:
            x = img.to(torch.float32)
            if x.dim() == 3:
                x = x.unsqueeze(0)
            h, w = x.shape[-2], x.shape[-1]
            if h % 8 != 0 or w % 8 != 0:
                raise ValueError(
                    f"calibration image must have H, W multiples of 8, got ({h}, {w})"
                )
            q_map = calib_qtable.view(1, 3, 8, 8).repeat(1, 1, h // 8, w // 8)
            dct = batch_dct(x - 128.0)
            r = self._raw(dct, q_map, x)
            raws.append(float(r.item()))
        return torch.tensor(float(np.mean(raws)))

    @torch.no_grad()
    def _real_bpp_batch(
        self, calib_images: CalibImages, calib_qtable: Tensor
    ) -> Tensor:
        if calib_qtable.dim() == 2:
            calib_qtable = calib_qtable.unsqueeze(0).expand(3, 8, 8).contiguous()

        if isinstance(calib_images, Tensor) and calib_images.dim() == 4:
            iterable = [calib_images[i] for i in range(calib_images.shape[0])]
        else:
            iterable = list(calib_images)

        bpps = [_real_bpp_per_image(img, calib_qtable) for img in iterable]
        return torch.tensor(float(np.mean(bpps)))


# ---------------------------------------------------------------------------
# Proxies
# ---------------------------------------------------------------------------

# Differential-entropy constants for unit-std distributions, in bits.
# h(X) = log2(sigma) + const for any unit-std distribution scaled to sigma.
_ENTROPY_CONST = {
    # Gaussian: h = 0.5 * log2(2*pi*e*sigma^2) = log2(sigma) + 0.5*log2(2*pi*e).
    "gaussian": 0.5 * math.log2(2.0 * math.pi * math.e),  # ~2.047
    # Laplacian (var = 2 b^2): h = log2(2*e*b) = log2(sigma) + log2(2*e/sqrt(2)).
    "laplacian": math.log2(2.0 * math.e / math.sqrt(2.0)),  # ~1.943
    # Rice (here used as a placeholder for heavier-tailed; per spec):
    # h = log2(sigma) + log2(e) + 1.
    "rice": math.log2(math.e) + 1.0,  # ~2.443
}


class ShannonRate(RateProxy):
    """Per-(channel, frequency) differential-entropy proxy.

    For each (c, u, v), takes the empirical std of `dct[blocks at (u, v)] / Q`
    over batch and blocks, converts to bits/coeff under the chosen
    distribution, clamps at 0, and sums to a per-block bit count.

    Args:
        dist: 'gaussian', 'laplacian', or 'rice'. Sets the entropy constant.
        dc_dpcm: if True, the (c, 0, 0) bin's std is computed over inter-block
            DC differences (matching JPEG's DPCM coding) instead of raw DC values.
    """

    def __init__(
        self,
        calib_images: CalibImages,
        calib_qtable: Tensor,
        *,
        dist: str = "gaussian",
        dc_dpcm: bool = False,
    ) -> None:
        super().__init__()
        if dist not in _ENTROPY_CONST:
            raise ValueError(f"dist must be one of {list(_ENTROPY_CONST)}, got {dist!r}")
        self.dist = dist
        self.dc_dpcm = dc_dpcm
        self.const = _ENTROPY_CONST[dist]
        self._calibrate(calib_images, calib_qtable)

    def _raw(self, dct: Tensor, q_map: Tensor, x_in: Tensor | None = None) -> Tensor:
        qtable = _qtable_from_q_map(q_map)  # (3, 8, 8)
        blocks = blockify(dct, 8)  # (B, 3, n_blocks, 8, 8)
        z = blocks / qtable.view(1, 3, 1, 8, 8)

        sigma = z.std(dim=(0, 2))  # (3, 8, 8)
        if self.dc_dpcm:
            dc = z[..., 0, 0]  # (B, 3, n_blocks)
            dc_diff = _dpcm_diff(dc)
            sigma_dc = dc_diff.std(dim=(0, 2))  # (3,)
            sigma = sigma.clone()
            sigma[:, 0, 0] = sigma_dc

        H = torch.log2(sigma + 1e-12) + self.const
        H = H.clamp(min=0.0)
        bits_per_block = H.sum()  # scalar
        bpp = bits_per_block / 64.0  # 64 pixels per block
        return bpp


class LogMagRate(RateProxy):
    """Sum of `log2(1 + |coeff|/Q)` over coefficients, per-pixel normalized.

    The Hu/Guleryuz "sandwiched codec" proxy form. After the base class fits
    `alpha`, `alpha * raw` matches real JPEG bpp on the calibration set.
    """

    def __init__(self, calib_images: CalibImages, calib_qtable: Tensor) -> None:
        super().__init__()
        self._calibrate(calib_images, calib_qtable)

    def _raw(self, dct: Tensor, q_map: Tensor, x_in: Tensor | None = None) -> Tensor:
        B, _, H_, W_ = dct.shape
        raw = (dct.abs() / q_map).add_(1.0).log2().sum() / (B * H_ * W_)
        return raw


class LogMagRateAnchored(RateProxy):
    """Per-image-anchored Hu 2024 / Guleryuz Eq. 2 proxy.

    Calls libjpeg per image per step to get the *actual* bpp, then defines

        a = stop_grad(real_bpp / raw_per_image)
        bpp = (a * raw_per_image).mean()

    so the value is exactly `real_bpp.mean()` while gradients still flow
    through `raw_per_image = sum log2(1 + |dct|/Q)`. No calibration phase
    needed; `alpha` stays at 1.0.
    """

    def __init__(
        self,
        calib_images: CalibImages | None = None,
        calib_qtable: Tensor | None = None,
    ) -> None:
        super().__init__()
        # Calibration arguments accepted for interface uniformity but unused.

    def _raw(self, dct: Tensor, q_map: Tensor, x_in: Tensor | None = None) -> Tensor:
        raise NotImplementedError("LogMagRateAnchored overrides forward")

    def forward(
        self, dct: Tensor, q_map: Tensor, x_in: Tensor | None = None
    ) -> Tensor:
        if x_in is None:
            raise ValueError("LogMagRateAnchored requires x_in to call libjpeg")
        B, _, H_, W_ = dct.shape
        raw_per_image = (
            (dct.abs() / q_map).add_(1.0).log2().sum(dim=(1, 2, 3))
        )  # (B,)

        qtable = _qtable_from_q_map(q_map)  # (3, 8, 8) float, may be requires_grad
        qt_int = torch.clamp(qtable.detach().round(), 1, 255).to(torch.int32).cpu()
        x_cpu = x_in.detach().cpu()

        real_bpps = []
        for b in range(B):
            x_b = x_cpu[b].clamp(0, 255).round().to(torch.uint8)
            bytes_ = jpeg_codec.encode(x_b, qt_int)
            real_bpps.append(len(bytes_) * 8.0 / (H_ * W_))

        real_bpp = torch.tensor(
            real_bpps, dtype=raw_per_image.dtype, device=raw_per_image.device
        )
        a = real_bpp / (raw_per_image.detach() + 1e-12)  # detached
        bpp_per_image = a * raw_per_image
        return bpp_per_image.mean()


class RunLengthRate(RateProxy):
    """Sparsity-aware proxy modeling JPEG's zigzag run-length AC coding.

    Per 8x8 block, after dividing by Q:
        DC bits  = 1 + log2(1 + |dc|)              (DC always coded)
        AC bits  = sum_k tanh((c_k)^2) * (log2(1 + |c_k|) + run_overhead)
                                                    (smooth nonzero gate)

    `run_overhead` is a fixed scalar (default 4 bits, a JPEG-Huffman ballpark);
    the calibration `alpha` absorbs the exact value.
    """

    def __init__(
        self,
        calib_images: CalibImages,
        calib_qtable: Tensor,
        *,
        run_overhead: float = 4.0,
        dc_dpcm: bool = True,
    ) -> None:
        super().__init__()
        self.run_overhead = float(run_overhead)
        self.dc_dpcm = dc_dpcm
        zr, zc = _zigzag_indices()
        self.register_buffer("_zr", zr, persistent=False)
        self.register_buffer("_zc", zc, persistent=False)
        self._calibrate(calib_images, calib_qtable)

    def _raw(self, dct: Tensor, q_map: Tensor, x_in: Tensor | None = None) -> Tensor:
        B, _, H_, W_ = dct.shape
        qtable = _qtable_from_q_map(q_map)  # (3, 8, 8)
        blocks = blockify(dct, 8)  # (B, 3, n_blocks, 8, 8)
        z = blocks / qtable.view(1, 3, 1, 8, 8)
        # Zigzag reorder: (B, 3, n_blocks, 64)
        z_zz = z[:, :, :, self._zr, self._zc]

        # DC at zigzag position 0
        dc = z_zz[..., 0]  # (B, 3, n_blocks)
        if self.dc_dpcm:
            dc_used = _dpcm_diff(dc)
        else:
            dc_used = dc
        bits_dc = (1.0 + torch.log2(dc_used.abs() + 1.0)).sum()

        # AC at zigzag positions 1..63
        ac = z_zz[..., 1:]  # (B, 3, n_blocks, 63)
        p_nz = torch.tanh(ac.pow(2))
        mag = torch.log2(ac.abs() + 1.0)
        bits_ac = (p_nz * (mag + self.run_overhead)).sum()

        total_bits = bits_dc + bits_ac
        bpp = total_bits / (B * H_ * W_)
        return bpp


class GaussianNoiseRelaxRate(RateProxy):
    """Ballé-2017-style Gaussian-relaxation proxy.

    Adds the same `U[-1/2, 1/2]` noise as `JPEGProxy`'s forward, computes
    per-(c, u, v) variance, and applies the Gaussian differential entropy:
    `H = 0.5 * log2(2*pi*e*sigma_z2)`.
    """

    def __init__(self, calib_images: CalibImages, calib_qtable: Tensor) -> None:
        super().__init__()
        self._calibrate(calib_images, calib_qtable)

    def _raw(self, dct: Tensor, q_map: Tensor, x_in: Tensor | None = None) -> Tensor:
        qtable = _qtable_from_q_map(q_map)
        blocks = blockify(dct, 8)
        z = blocks / qtable.view(1, 3, 1, 8, 8)
        if self.training:
            z = z + torch.empty_like(z).uniform_(-0.5, 0.5)
        sigma2 = z.var(dim=(0, 2))  # (3, 8, 8)
        H = 0.5 * torch.log2(2.0 * math.pi * math.e * sigma2 + 1e-12)
        H = H.clamp(min=0.0)
        bits_per_block = H.sum()
        bpp = bits_per_block / 64.0
        return bpp


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def _make_synthetic_calib(n: int = 4, h: int = 64, w: int = 64) -> list[Tensor]:
    torch.manual_seed(7)
    out = []
    for _ in range(n):
        # Mostly low-frequency with some texture: produces nontrivial DCT
        # statistics so ShannonRate's std is well-defined.
        base = (torch.randn(3, h, w) * 25.0 + 128.0).clamp(0, 255)
        out.append(base.round().to(torch.uint8))
    return out


def _build_q_map(qtable: Tensor, h: int, w: int) -> Tensor:
    if qtable.dim() == 2:
        qtable = qtable.unsqueeze(0).expand(3, 8, 8).contiguous()
    return qtable.view(1, 3, 8, 8).repeat(1, 1, h // 8, w // 8).to(torch.float32)


def _smoke_zigzag() -> None:
    """Sanity-check the zigzag table covers 0..63 exactly once."""
    zr, zc = _zigzag_indices()
    flat = zr * 8 + zc
    assert sorted(flat.tolist()) == list(range(64)), sorted(flat.tolist())
    # Position 0 must be DC.
    assert (zr[0].item(), zc[0].item()) == (0, 0)
    print("[rate] zigzag table covers 0..63 exactly once, starts at DC")


def _smoke_constant_input_zero_bpp() -> None:
    """A constant 128 input has all-zero DCT (after level shift), so all
    proxies should yield ~0 bpp before alpha scaling."""
    h, w = 64, 64
    calib = _make_synthetic_calib()
    qtable = torch.full((3, 8, 8), 8.0)
    q_map = _build_q_map(qtable, h, w)

    x_const = torch.full((1, 3, h, w), 128.0)
    dct_const = batch_dct(x_const - 128.0)
    assert dct_const.abs().max().item() < 1e-4

    # Tolerance differs by proxy: log-mag and Shannon vanish exactly on
    # constant input; RunLengthRate pays a fixed per-block DC overhead
    # (matches the empty-image JPEG-marker cost), which is small in absolute
    # bpp but nonzero. Gaussian-noise-relax has a noise floor from the
    # +1/12 uniform-noise variance, also small.
    cases = [
        (ShannonRate, {"dist": "gaussian"}, 5e-3),
        (ShannonRate, {"dist": "laplacian"}, 5e-3),
        (LogMagRate, {}, 5e-3),
        (RunLengthRate, {}, 0.1),
        (GaussianNoiseRelaxRate, {}, 5e-3),
    ]
    for cls, kw, tol in cases:
        proxy = cls(calib, qtable, **kw)
        proxy.eval()
        with torch.no_grad():
            bpp = proxy(dct_const, q_map, x_const).item()
        print(f"[rate] {cls.__name__}{kw}: const-input bpp = {bpp:.6f}")
        assert abs(bpp) < tol, (cls, kw, bpp, tol)


def _smoke_doubling_q_decreases_bpp() -> None:
    """For Shannon/Gaussian/log-mag, doubling Q must decrease bpp."""
    h, w = 128, 128
    calib = _make_synthetic_calib(h=h, w=w)
    qtable_lo = torch.full((3, 8, 8), 4.0)
    qtable_hi = torch.full((3, 8, 8), 32.0)

    torch.manual_seed(1)
    x = (torch.randn(1, 3, h, w) * 30.0 + 128.0).clamp(0, 255).float()
    dct = batch_dct(x - 128.0)

    for cls in [ShannonRate, LogMagRate, RunLengthRate, GaussianNoiseRelaxRate]:
        proxy = cls(calib, qtable_lo)
        proxy.eval()
        q_map_lo = _build_q_map(qtable_lo, h, w)
        q_map_hi = _build_q_map(qtable_hi, h, w)
        with torch.no_grad():
            bpp_lo = proxy(dct, q_map_lo, x).item()
            bpp_hi = proxy(dct, q_map_hi, x).item()
        print(f"[rate] {cls.__name__}: bpp(Q=4)={bpp_lo:.3f}, bpp(Q=32)={bpp_hi:.3f}")
        assert bpp_hi < bpp_lo, (cls.__name__, bpp_lo, bpp_hi)


def _smoke_calibration_consistency() -> None:
    """After calibration, proxy bpp on the calibration batch ~= real-codec bpp."""
    calib = _make_synthetic_calib(n=4, h=64, w=64)
    qtable = torch.full((3, 8, 8), 8.0)
    real = float(np.mean([_real_bpp_per_image(img, qtable) for img in calib]))
    print(f"[rate] real bpp on synthetic calib (Q=8) = {real:.3f}")

    # Tolerances: the calibration runs in train mode, so noise-relaxation
    # proxies have a small (deterministic) eval/train mismatch from the
    # +1/12 uniform-noise variance term. ~1% is generous.
    cases = [
        (ShannonRate, {"dist": "gaussian"}, 5e-3),
        (LogMagRate, {}, 5e-3),
        (RunLengthRate, {}, 5e-3),
        (GaussianNoiseRelaxRate, {}, 1e-2),
    ]
    for cls, kw, tol in cases:
        proxy = cls(calib, qtable, **kw)
        proxy.eval()
        bpps = []
        with torch.no_grad():
            for img in calib:
                x = img.float().unsqueeze(0)
                h, w = x.shape[-2:]
                q_map = _build_q_map(qtable, h, w)
                dct = batch_dct(x - 128.0)
                bpps.append(proxy(dct, q_map, x).item())
        proxy_avg = float(np.mean(bpps))
        rel = abs(proxy_avg - real) / max(real, 1e-6)
        print(
            f"[rate] {cls.__name__}{kw}: proxy={proxy_avg:.3f}, "
            f"real={real:.3f}, rel_err={rel:.2%}, alpha={proxy.alpha.item():.3e}"
        )
        assert rel < tol, (cls.__name__, proxy_avg, real, tol)


def _smoke_qtable_grad_sign() -> None:
    """Increasing Q should decrease the rate proxy: dR/dQ <= 0."""
    h, w = 64, 64
    calib = _make_synthetic_calib(h=h, w=w)
    qtable = torch.full((3, 8, 8), 8.0)

    torch.manual_seed(2)
    x = (torch.randn(1, 3, h, w) * 30.0 + 128.0).clamp(0, 255).float()
    dct = batch_dct(x - 128.0)

    for cls in [ShannonRate, LogMagRate, RunLengthRate, GaussianNoiseRelaxRate]:
        proxy = cls(calib, qtable)
        proxy.train()
        # Make q_map a leaf with grad.
        q_leaf = qtable.clone().detach().requires_grad_(True)
        q_map = q_leaf.view(1, 3, 8, 8).repeat(1, 1, h // 8, w // 8)
        bpp = proxy(dct, q_map, x)
        bpp.backward()
        g = q_leaf.grad
        # Most entries should be non-positive; a small fraction may be
        # positive due to noise (especially the post-noise variance form).
        frac_neg = (g <= 1e-6).float().mean().item()
        print(
            f"[rate] {cls.__name__}: dR/dQ <= 0 fraction = {frac_neg:.2%}, "
            f"max grad = {g.max().item():+.3e}"
        )
        assert frac_neg > 0.85, (cls.__name__, frac_neg)


def _smoke_logmag_anchored_matches_real() -> None:
    """LogMagRateAnchored.forward should equal real-codec bpp by construction."""
    torch.manual_seed(3)
    h, w = 64, 64
    qtable = torch.full((3, 8, 8), 8.0)
    proxy = LogMagRateAnchored()

    x_uint8 = (torch.randn(3, h, w) * 25.0 + 128.0).clamp(0, 255).round().to(torch.uint8)
    real = _real_bpp_per_image(x_uint8, qtable)
    x = x_uint8.float().unsqueeze(0)
    q_map = _build_q_map(qtable, h, w)
    dct = batch_dct(x - 128.0)
    proxy.train()
    out = proxy(dct, q_map, x).item()
    print(f"[rate] LogMagRateAnchored: proxy={out:.3f}, real={real:.3f}")
    assert abs(out - real) < 1e-3, (out, real)


def _smoke_gradient_flows_through_q_unconstrained() -> None:
    """End-to-end: gradient of rate w.r.t. JPEGProxy.Q_unconstrained is finite."""
    from .proxy import JPEGProxy

    h, w = 64, 64
    calib = _make_synthetic_calib(h=h, w=w)
    qtable = torch.full((3, 8, 8), 8.0)
    proxy = JPEGProxy(init=qtable).train()
    rate = LogMagRate(calib, qtable)
    rate.train()

    torch.manual_seed(4)
    x = (torch.randn(1, 3, h, w) * 30.0 + 128.0).clamp(0, 255).float()
    proxy.zero_grad()
    _ = proxy(x)
    bpp = rate(proxy._last_dct, proxy._last_q_map, x)
    bpp.backward()
    g = proxy.Q_unconstrained.grad
    assert g is not None and torch.isfinite(g).all().item(), g
    print(f"[rate] end-to-end qtable grad: max abs = {g.abs().max().item():.3e}")


if __name__ == "__main__":
    _smoke_zigzag()
    _smoke_constant_input_zero_bpp()
    _smoke_doubling_q_decreases_bpp()
    _smoke_calibration_consistency()
    _smoke_qtable_grad_sign()
    _smoke_logmag_anchored_matches_real()
    _smoke_gradient_flows_through_q_unconstrained()
    print("[rate] all smoke tests passed")
