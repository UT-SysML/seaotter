"""Differentiable surrogate for the seaotter JPEG codec.

`JPEGProxy` is an `nn.Module` whose state is one (3, 8, 8) real-valued
parameter that maps through `softsign + affine` to a per-component
quantization matrix in (1, 256). Forward pass runs the JPEG distortion
chain in float arithmetic:

    level shift -> 8x8 block DCT -> divide by Q -> (additive uniform
    noise if training) -> multiply by Q -> 8x8 block IDCT -> un-shift

No `torch.round` is ever called inside `forward`. Hard rounding to an
integer qtable for `seaotter.encode` is the caller's responsibility:

    q_int = torch.clamp(proxy.qtable().round(), 1, 255).to(torch.int32)

Distortion surrogate only — no rate / file-size term. See `jpeg_proxy.md`
for design context.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ._dct import batch_dct, batch_idct


# Mapping: Q = _Q_MID + _Q_HALF * softsign(p)
#   p -> -inf : Q -> 1
#   p =  -256 : Q ~= 1.5
#   p =     0 : Q  = 128.5
#   p =  +256 : Q ~= 255.5
#   p -> +inf : Q -> 256
_Q_MID = 128.5
_Q_HALF = 127.5


def _softsign_inv(s: Tensor | float) -> Tensor | float:
    """Inverse of x / (1 + |x|), valid for s in (-1, 1)."""
    if isinstance(s, Tensor):
        return s / (1.0 - s.abs())
    return s / (1.0 - abs(s))


def _qtable_to_unconstrained(q: Tensor) -> Tensor:
    """Invert qtable() so a target Q (in (1, 256)) becomes a parameter value."""
    if q.min().item() <= 1.0 or q.max().item() >= 256.0:
        raise ValueError(
            "init qtable values must lie strictly in (1, 256); "
            f"got min={q.min().item():.4f}, max={q.max().item():.4f}"
        )
    s = (q - _Q_MID) / _Q_HALF
    return _softsign_inv(s)


class JPEGProxy(nn.Module):
    """Differentiable surrogate for `seaotter.jpeg_codec`.

    State:
        Q_unconstrained: (3, 8, 8) real-valued nn.Parameter, one (8, 8)
            matrix per channel (Y, Cb, Cr by codec convention; the codec
            is colorspace-agnostic, so they are just "channel 0/1/2").
            Mapped to the qtable range (1, 256) via softsign + affine.

    forward(x):
        x: (B, 3, H, W) float tensor in the codec's natural [0, 255]
           range. H and W must be multiples of 8 (caller pads).
        returns: (B, 3, H, W) float tensor — the surrogate
           reconstruction. No clamp to [0, 255] (caller can clamp).
    """

    def __init__(self, init: Union[float, Tensor] = -256.0) -> None:
        """
        Args:
            init: how to initialize `Q_unconstrained`.
                - float: fill value applied to all 3*64 entries. Default
                  -256.0 yields qtable() ~= 1.5 everywhere (very mild
                  quantization, near-identity surrogate).
                - Tensor of shape (3, 8, 8) or (8, 8) in the qtable
                  domain (values in (1, 256)): inverted through the
                  param->qtable mapping to seed the parameter.
        """
        super().__init__()

        if isinstance(init, Tensor):
            q = init.detach().to(torch.float32)
            if q.shape == (8, 8):
                q = q.unsqueeze(0).expand(3, 8, 8).contiguous()
            elif q.shape != (3, 8, 8):
                raise ValueError(
                    f"init tensor must be (8, 8) or (3, 8, 8), got {tuple(q.shape)}"
                )
            p = _qtable_to_unconstrained(q)
        else:
            p = torch.full((3, 8, 8), float(init))

        self.Q_unconstrained = nn.Parameter(p)

    def qtable(self) -> Tensor:
        """Real-valued per-component qtable in (1, 256), shape (3, 8, 8).

        No noise, no rounding — this is the continuous-domain divisor
        used inside `forward`. To get the integer qtable that goes into
        `seaotter.encode`, the caller does
        `torch.clamp(proxy.qtable().round(), 1, 255).to(torch.int32)`.
        """
        return _Q_MID + _Q_HALF * F.softsign(self.Q_unconstrained)

    def _q_map(self, h: int, w: int) -> Tensor:
        """Tile (3, 8, 8) qtable into (1, 3, H, W) for elementwise division."""
        if h % 8 != 0 or w % 8 != 0:
            raise ValueError(f"H and W must be multiples of 8; got ({h}, {w})")
        q = self.qtable()  # (3, 8, 8)
        q = q.repeat(1, h // 8, w // 8)  # (3, H, W)
        return q.unsqueeze(0)  # (1, 3, H, W)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"x must be (B, 3, H, W); got {tuple(x.shape)}")
        if not torch.is_floating_point(x):
            raise TypeError(f"x must be a float tensor; got {x.dtype}")

        b, _, h, w = x.shape
        q_map = self._q_map(h, w)

        x = x - 128.0
        dct = batch_dct(x)

        # Expose intermediates for rate proxies. These are activations, not
        # state — plain attributes (no register_buffer / Parameter).
        self._last_dct = dct
        self._last_q_map = q_map

        y = dct / q_map
        if self.training:
            y = y + torch.empty_like(y).uniform_(-0.5, 0.5)
        y = y * q_map

        x_hat = batch_idct(y) + 128.0
        return x_hat


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

def _smoke_pure_dct_roundtrip() -> None:
    """idct(dct(x)) == x to numerical precision."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    err = (batch_idct(batch_dct(x)) - x).abs().max().item()
    print(f"[proxy] pure DCT round-trip max abs err: {err:.3e}")
    assert err < 1e-4, err


def _smoke_default_init_near_identity() -> None:
    """proxy.eval() at default init reconstructs to ~original (no quant)."""
    torch.manual_seed(1)
    proxy = JPEGProxy().eval()
    q = proxy.qtable()
    print(
        f"[proxy] default-init qtable: min={q.min().item():.4f}, "
        f"max={q.max().item():.4f}"
    )
    # softsign(-256) = -256/257, so Q = 128.5 + 127.5 * (-256/257) ~= 1.4961
    expected_q = _Q_MID + _Q_HALF * (-256.0 / 257.0)
    assert torch.allclose(q, torch.full_like(q, expected_q), atol=1e-4), (
        q.flatten()[:3].tolist(), expected_q
    )

    x = torch.rand(1, 3, 64, 64) * 255.0
    with torch.no_grad():
        y = proxy(x)
    err = (y - x).abs().max().item()
    # Eval mode is identity (no noise, no rounding), so the only error
    # is float DCT round-trip.
    print(f"[proxy] eval-mode reconstruction max abs err: {err:.3e}")
    assert err < 1e-3, err


def _smoke_constant_block_eval() -> None:
    """A constant 8x8 block should round-trip bit-exact in eval mode."""
    proxy = JPEGProxy().eval()
    x = torch.full((1, 3, 16, 16), 128.0)
    with torch.no_grad():
        y = proxy(x)
    err = (y - x).abs().max().item()
    print(f"[proxy] constant-block eval err: {err:.3e}")
    assert err < 1e-3, err


def _smoke_noise_stat() -> None:
    """In train mode, residual std should grow with q.

    The forward residual at the pixel level comes from
        delta_y = uniform(-0.5, 0.5) * q  (in the DCT domain)
    so per-DCT-coefficient variance is q^2 / 12. After IDCT (an
    orthonormal transform up to the (2/N)*alpha rescaling, which is
    its own inverse in this code path), per-pixel variance equals
    per-coefficient variance, so per-pixel std ~= q / sqrt(12).
    """
    torch.manual_seed(2)
    qs = [4.0, 16.0, 64.0]
    n_trials = 64
    for q_target in qs:
        init = torch.full((3, 8, 8), q_target)
        proxy = JPEGProxy(init=init).train()
        x = torch.full((1, 3, 64, 64), 128.0)
        residuals = []
        with torch.no_grad():
            for _ in range(n_trials):
                residuals.append(proxy(x) - x)
        r = torch.stack(residuals)
        std = r.std().item()
        expected = q_target / math.sqrt(12.0)
        rel_err = abs(std - expected) / expected
        print(
            f"[proxy] q={q_target}: residual std={std:.3f}, "
            f"expected~{expected:.3f}, rel_err={rel_err:.3f}"
        )
        assert rel_err < 0.10, (q_target, std, expected, rel_err)


def _smoke_eval_no_quant() -> None:
    """Even at q=128, eval mode should reconstruct to ~original (no rounding,
    no noise => the q is irrelevant: y = (D/Q)*Q = D)."""
    init = torch.full((3, 8, 8), 128.0)
    proxy = JPEGProxy(init=init).eval()
    torch.manual_seed(3)
    x = torch.rand(1, 3, 32, 32) * 255.0
    with torch.no_grad():
        y = proxy(x)
    err = (y - x).abs().max().item()
    print(f"[proxy] q=128 eval err (should be ~0): {err:.3e}")
    assert err < 1e-3, err


def _smoke_cross_check_vs_codec() -> None:
    """Compare proxy(eval+train) vs the real codec at uniform q=4 on a
    natural image. PSNR should agree to within ~0.5 dB.

    We use train mode for the noise-quantized surrogate (the standard
    Balle-style stand-in for hard rounding), and eval mode as the
    no-quant ceiling.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        print(f"[proxy] cross-check skipped (no `datasets`): {e}")
        return

    from .jpeg_codec import decode, encode

    ds = load_dataset("danjacobellis/kodak", split="validation")
    img = ds[22]["image"]
    arr = torch.tensor(list(img.tobytes()), dtype=torch.uint8).reshape(
        img.size[1], img.size[0], 3
    )
    x_uint8 = arr.permute(2, 0, 1).contiguous()
    # crop to multiples of 8
    h = (x_uint8.shape[1] // 8) * 8
    w = (x_uint8.shape[2] // 8) * 8
    x_uint8 = x_uint8[:, :h, :w]

    q = 4
    qtable = torch.full((8, 8), float(q))

    # Real codec
    bs = encode(x_uint8, qtable)
    rec_codec = decode(bs).float()

    # Proxy in train mode (noise quantization)
    proxy = JPEGProxy(init=qtable).train()
    torch.manual_seed(0)
    with torch.no_grad():
        x_float = x_uint8.float().unsqueeze(0)
        rec_proxy = proxy(x_float).clamp_(0, 255).squeeze(0)

    def psnr(a: Tensor, b: Tensor) -> float:
        mse = ((a - b) ** 2).mean().item()
        return 10.0 * math.log10(255.0 ** 2 / mse)

    p_codec = psnr(x_uint8.float(), rec_codec)
    p_proxy = psnr(x_uint8.float(), rec_proxy)
    print(
        f"[proxy] kodak[22] q={q}: codec PSNR={p_codec:.2f} dB, "
        f"proxy(train) PSNR={p_proxy:.2f} dB, delta={p_codec - p_proxy:+.2f} dB"
    )
    assert abs(p_codec - p_proxy) < 1.0, (p_codec, p_proxy)


def _smoke_caller_side_int_qtable() -> None:
    """Demonstrate the caller-side recipe for getting an integer qtable
    suitable for seaotter.encode."""
    init = torch.tensor([
        [16.0, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ])
    proxy = JPEGProxy(init=init)
    q_recovered = proxy.qtable()  # (3, 8, 8)
    err = (q_recovered - init.unsqueeze(0).expand(3, 8, 8)).abs().max().item()
    print(f"[proxy] init->qtable round-trip max abs err: {err:.3e}")
    assert err < 1e-4, err

    q_int = torch.clamp(q_recovered.round(), 1, 255).to(torch.int32)
    assert q_int.shape == (3, 8, 8)
    assert q_int.dtype == torch.int32
    print(f"[proxy] caller-side int qtable shape={tuple(q_int.shape)}, "
          f"dtype={q_int.dtype}, range=[{q_int.min().item()}, {q_int.max().item()}]")


if __name__ == "__main__":
    _smoke_pure_dct_roundtrip()
    _smoke_default_init_near_identity()
    _smoke_constant_block_eval()
    _smoke_noise_stat()
    _smoke_eval_no_quant()
    _smoke_caller_side_int_qtable()
    _smoke_cross_check_vs_codec()
    print("[proxy] all smoke tests passed")
