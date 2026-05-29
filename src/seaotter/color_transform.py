"""Learnable color transforms that sandwich the JPEG codec.

`ForwardTransform` / `InverseTransform` wrap the existing `JPEGProxy` /
`seaotter.encode`-`decode` to replace the fixed JFIF YCbCr round-trip with
a small learned color pair. Both modules are parameterized by an `arch`
flag selecting one of several architectural variants:

  A  — Softsign (forward) + SoftsignExpand (inverse). The inverse
       nonlinearity is the algebraic inverse of the forward Softsign
       (`g(v) = σ·v / (r + ε - |v|)`), so the composition is exactly the
       identity at random or algebraic-identity init. ε small (default
       0.5) so the singularity at |v| = r is regularized.
  B  — Hardtanh (forward) + Hardtanh (inverse). No companding; the
       analysis just clips to the codec range. The composition is the
       identity for non-saturating inputs.
  D  — As variant A but with 3×3 (instead of 1×1) convolutions on both
       sides, mirroring FRAPPE's patch-sized convolution.
  E  — The round-4 default. Forward `SoftsignCompanding(qat=True)`,
       inverse `Hardtanh` (no inverse companding). Capped at ~21 dB at
       algebraic-identity init per `experiments/sandwich_lr_findings.md`.
       Kept as the baseline so the new diagnostic harness can reproduce
       round-4's measured ceiling.
  F  — As variant A but with smaller `bits` (default 6 → `r = 31`).
       Gentler companding at the cost of fewer codec levels.

Boundary noise (`UniformTrainingNoise`) lives inside each module and is
gated by `self.training`. Eval-mode helpers `fwd.codec_input_uint8(x)`
and `inv.output_uint8(z)` clamp + round to uint8 — call sites are
responsible for `module.eval()` before invoking them.
"""

from __future__ import annotations

import math

import torch
from gigatorch.ops import (
    ChannelAffine,
    Softsign,
    SoftsignCompanding,
    UniformTrainingNoise,
)
from torch import Tensor, nn


_ARCH_CHOICES = ("A", "B", "D", "E", "F", "G")


class SoftsignExpand(nn.Module):
    """Algebraic inverse of `gigatorch.ops.Softsign`.

    Forward Softsign: `f(u) = r·u / (σ + |u|)`,  range `(-r, r)`.
    Inverse:         `g(v) = σ·v / (r + ε - |v|)`,  domain `(-r-ε, r+ε)`
    (regularized at `|v| = r`).

    Composition `g(f(u)) → u` exactly when `ε = 0`; small `ε > 0` adds a
    minor companding bias but avoids the division-by-zero at the boundary
    where the codec's uint8 quantization can push `|v|` arbitrarily close
    to `r`.

    The learnable σ uses the same parameterization (`|_σ| + 1e-6`) as the
    forward `Softsign` so the two can be initialized symmetrically.
    """

    def __init__(
        self,
        dim: int,
        num_channels: int,
        bits: int = 8,
        eps: float = 0.5,
        sigma_init: float | None = None,
    ) -> None:
        super().__init__()
        self.shape = [1, -1] + [1] * dim
        self.r = 2 ** (bits - 1) - 1  # 127 for bits=8
        self.eps = float(eps)
        if sigma_init is None:
            sigma_init = float(self.r - 1)
        self._σ = nn.Parameter(torch.full((num_channels,), float(sigma_init)))

    def forward(self, x: Tensor) -> Tensor:
        σ = (self._σ.abs() + 1e-6).view(self.shape)
        denom = (self.r + self.eps) - x.abs()
        denom = denom.clamp(min=1e-3)
        return σ * x / denom


# ---------------------------------------------------------------------------
# Helpers shared across variants
# ---------------------------------------------------------------------------

def _conv_set_identity(conv: nn.Conv2d, bias_value: float = 0.0) -> None:
    """Zero `conv.weight` and set the center tap to identity per channel.
    Sets `conv.bias` to `bias_value` (per-channel constant).
    """
    with torch.no_grad():
        conv.weight.zero_()
        kh, kw = conv.kernel_size
        ic, jc = kh // 2, kw // 2
        c_out, c_in = conv.weight.shape[:2]
        assert c_out == c_in, "identity init requires square channel count"
        for i in range(c_out):
            conv.weight[i, i, ic, jc] = 1.0
        if conv.bias is not None:
            conv.bias.fill_(bias_value)


# JFIF YCbCr matrices (full-range, uncentered RGB[0,255] → JFIF YCbCr[0,255]).
# Forward:  codec_input_Y  = 0.299R + 0.587G + 0.114B               (no bias)
#           codec_input_Cb = -0.169R - 0.331G + 0.5B + 128
#           codec_input_Cr = 0.5R - 0.419G - 0.081B + 128
# Inverse:  R = Y + 1.402·(Cr - 128)
#           G = Y - 0.344·(Cb - 128) - 0.714·(Cr - 128)
#           B = Y + 1.772·(Cb - 128)
_JFIF_FWD = torch.tensor([
    [ 0.299,  0.587,  0.114],
    [-0.168736, -0.331264, 0.5],
    [ 0.5, -0.418688, -0.081312],
])
# Centered fwd bias: codec_input = analysis_transform(x) + 128, so we want
# `analysis_transform(x) = JFIF(x) - 128`. Per-channel offsets:
#   Y row:  b_Y = -128 (Y' has no +128 offset, so we subtract 128 to center)
#   Cb row: b_Cb = 0  (Cb' = ... + 128, so JFIF - 128 has no offset on this row)
#   Cr row: b_Cr = 0
_JFIF_FWD_BIAS_CENTERED = torch.tensor([-128.0, 0.0, 0.0])
_JFIF_INV = torch.tensor([
    [1.0,  0.0,    1.402],
    [1.0, -0.344136, -0.714136],
    [1.0,  1.772,  0.0],
])


def _conv_set_jfif(conv: nn.Conv2d, kind: str) -> None:
    """Initialize a 1×1 or 3×3 Conv2d to the JFIF Y/Cb/Cr (`kind='fwd'`) or
    Y/Cb/Cr→RGB (`kind='inv'`) matrix. Centered convention:
      - Fwd: `analysis_transform(x) + 128 = codec_input` with codec_input
             ≈ JFIF YCbCr. The conv biases are per-channel `(-128, 0, 0)` so
             the +128 outside the Sequential recovers (Y, Cb', Cr').
      - Inv: `synthesis_transform(z - 128) + 128 = rgb_hat`. Conv bias all
             zeros — the +128 outside recovers RGB.
    """
    if kind == "fwd":
        W_mat = _JFIF_FWD
        b_vec = _JFIF_FWD_BIAS_CENTERED
    elif kind == "inv":
        W_mat = _JFIF_INV
        b_vec = torch.zeros(3)
    else:
        raise ValueError(f"unknown kind={kind!r}, expected 'fwd' or 'inv'")
    with torch.no_grad():
        conv.weight.zero_()
        kh, kw = conv.kernel_size
        ic, jc = kh // 2, kw // 2
        # weight shape: (3, 3, kh, kw). Want weight[c_out, c_in, ic, jc] = W_mat[c_out, c_in].
        for i in range(3):
            for j in range(3):
                conv.weight[i, j, ic, jc] = float(W_mat[i, j])
        if conv.bias is not None:
            for c in range(3):
                conv.bias[c] = float(b_vec[c])


def _conv_random_init(conv: nn.Conv2d, init_sigma: float) -> None:
    with torch.no_grad():
        conv.weight.normal_(mean=0.0, std=init_sigma)
        if conv.bias is not None:
            conv.bias.zero_()


# ---------------------------------------------------------------------------
# ForwardTransform
# ---------------------------------------------------------------------------


class ForwardTransform(nn.Module):
    """Analysis transform: RGB float in `[0, 255]` → codec-domain float
    near `[0, 255]`.

    Internal Sequential differs per `arch`. The `+128` post-shift lives
    outside the Sequential for variants A/B/D/F so the Sequential output is
    centered near zero; variant E keeps the `+128` outside as well (matches
    round-4's exact layout) so existing analysis still applies.
    """

    def __init__(
        self,
        arch: str = "A",
        init_sigma: float | None = None,
        bits: int | None = None,
    ) -> None:
        super().__init__()
        if arch not in _ARCH_CHOICES:
            raise ValueError(f"unknown arch={arch!r}, expected one of {_ARCH_CHOICES}")
        self.arch = arch

        if arch == "A":
            self.analysis_transform = nn.Sequential(
                nn.Conv2d(3, 3, kernel_size=1, bias=True),
                Softsign(dim=2, num_channels=3, bits=8),
                UniformTrainingNoise(k=0),
                ChannelAffine(dim=2, num_channels=3, bias=False),
            )
        elif arch == "B":
            self.analysis_transform = nn.Sequential(
                nn.Conv2d(3, 3, kernel_size=1, bias=True),
                nn.Hardtanh(min_val=-128.0, max_val=127.0),
                UniformTrainingNoise(k=0),
            )
        elif arch == "D":
            self.analysis_transform = nn.Sequential(
                nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=True),
                Softsign(dim=2, num_channels=3, bits=8),
                UniformTrainingNoise(k=0),
                ChannelAffine(dim=2, num_channels=3, bias=False),
            )
        elif arch == "E":
            # Round-4 default. Matches state_dict layout of prior checkpoints.
            self.analysis_transform = nn.Sequential(
                nn.Conv2d(3, 3, kernel_size=1, bias=True),
                SoftsignCompanding(
                    dim=2, num_channels=3,
                    qat=True, bits=8, k=0,
                    affine=True, bias=False,
                ),
            )
        elif arch == "F":
            b = 6 if bits is None else int(bits)
            self.analysis_transform = nn.Sequential(
                nn.Conv2d(3, 3, kernel_size=1, bias=True),
                Softsign(dim=2, num_channels=3, bits=b),
                UniformTrainingNoise(k=0),
                ChannelAffine(dim=2, num_channels=3, bias=False),
            )
        elif arch == "G":
            # Same analysis as variant A: Conv → Softsign(σ_a) → noise → γ_a.
            # The synthesis (in InverseTransform) is "stacked" — one compand
            # at the codec output to bound noisy inputs, plus two expands.
            self.analysis_transform = nn.Sequential(
                nn.Conv2d(3, 3, kernel_size=1, bias=True),
                Softsign(dim=2, num_channels=3, bits=8),
                UniformTrainingNoise(k=0),
                ChannelAffine(dim=2, num_channels=3, bias=False),
            )

        # Random init of the conv. Default σ matches round-4. Algebraic-identity
        # init is opt-in via `init_algebraic_identity()`.
        if init_sigma is None:
            init_sigma = (1.0 / math.sqrt(3.0)) / 128.0
        _conv_random_init(self.analysis_transform[0], init_sigma)

    def forward(self, x: Tensor) -> Tensor:
        """Float codec-domain output centered near 128. Boundary noise applied
        iff `self.training`.
        """
        return self.analysis_transform(x) + 128.0

    def codec_input_uint8(self, x: Tensor) -> Tensor:
        """Eval-only: clamp + round to the uint8 tensor that goes into
        `seaotter.encode`. Caller must set `self.eval()`."""
        z = self.forward(x)
        return z.clamp(0.0, 255.0).round().to(torch.uint8)

    def init_algebraic_identity(self) -> None:
        """Set parameters to the algebraic-identity configuration.

        For arch A/D/E/F: Conv = I, bias = -128 (centers `[0, 255]` →
        `[-128, 127]`). Softsign σ keeps its default (`r - 1`). ChannelAffine
        γ keeps its default (1).
        For arch B: Conv = I, bias = -128. Hardtanh is parameter-free.
        """
        conv = self.analysis_transform[0]
        _conv_set_identity(conv, bias_value=-128.0)
        # The Softsign σ and ChannelAffine γ remain at their constructor defaults
        # (126 and 1.0 respectively for arch A/D; 30 and 1.0 for arch F bits=6).
        # For arch E the SoftsignCompanding's inner Softsign / ChannelAffine
        # also start at these defaults.

    def init_jfif(self) -> None:
        """Set parameters to the JFIF Y/Cb/Cr forward configuration.

        Conv weight set to the JFIF YCbCr forward matrix; Conv bias set to -128
        so that `analysis_transform(x) + 128 ≈ codec_input` lies near Y'/Cb'/Cr'
        (full-range JFIF). Softsign σ / ChannelAffine γ stay at defaults — for
        variant B (no companding), this gives near-bit-exact JFIF round-trip
        when paired with `InverseTransform(arch="B").init_jfif()`. For variants
        A/D/F (softsign companding), the companding distorts JFIF values, so
        this init is approximate.
        """
        conv = self.analysis_transform[0]
        _conv_set_jfif(conv, kind="fwd")


# ---------------------------------------------------------------------------
# InverseTransform
# ---------------------------------------------------------------------------


class InverseTransform(nn.Module):
    """Synthesis transform: codec-domain float in `[0, 255]` → RGB float in
    `[0, 255]`.

    Internal Sequential differs per `arch`. The pre-shift `(z - 128)` and
    post-shift `(+128)` for variants A/B/D/F are applied around the
    Sequential. Variant E uses round-4's exact `(z - 128)/126` /
    `127.5·y + 127.5` normalization.
    """

    SCALE_E: float = 126.0  # variant E pre-normalization scale

    def __init__(self, arch: str = "A", bits: int | None = None) -> None:
        super().__init__()
        if arch not in _ARCH_CHOICES:
            raise ValueError(f"unknown arch={arch!r}, expected one of {_ARCH_CHOICES}")
        self.arch = arch

        if arch == "A":
            self.synthesis_transform = nn.Sequential(
                UniformTrainingNoise(k=0),
                ChannelAffine(dim=2, num_channels=3, bias=False),
                SoftsignExpand(dim=2, num_channels=3, bits=8, eps=0.5),
                nn.Conv2d(3, 3, kernel_size=1, bias=True),
            )
        elif arch == "B":
            self.synthesis_transform = nn.Sequential(
                UniformTrainingNoise(k=0),
                nn.Conv2d(3, 3, kernel_size=1, bias=True),
                nn.Hardtanh(min_val=-128.0, max_val=127.0),
            )
        elif arch == "D":
            self.synthesis_transform = nn.Sequential(
                UniformTrainingNoise(k=0),
                ChannelAffine(dim=2, num_channels=3, bias=False),
                SoftsignExpand(dim=2, num_channels=3, bits=8, eps=0.5),
                nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=True),
            )
        elif arch == "E":
            # Round-4 default. State_dict matches prior checkpoints.
            self.synthesis_transform = nn.Sequential(
                UniformTrainingNoise(k=0),
                nn.Conv2d(3, 3, kernel_size=1, bias=True),
                nn.Hardtanh(),
            )
        elif arch == "F":
            b = 6 if bits is None else int(bits)
            self.synthesis_transform = nn.Sequential(
                UniformTrainingNoise(k=0),
                ChannelAffine(dim=2, num_channels=3, bias=False),
                SoftsignExpand(dim=2, num_channels=3, bits=b, eps=0.5),
                nn.Conv2d(3, 3, kernel_size=1, bias=True),
            )
        elif arch == "G":
            # Stacked design (user proposal): synthesis-side compand bounds the
            # potentially out-of-domain codec output (saturates noisy values at
            # ±r, with smooth gradients) and carries the QAT noise modeling
            # codec uint8 quantization. Two SoftsignExpands then invert,
            # respectively, the synthesis compand and the analysis compand.
            #   z → Softsign(σ_s) → noise → γ_s   (synth compand)
            #     → γ_s_inv → SoftsignExpand(σ_s)  (invert synth compand)
            #     → γ_a_inv → SoftsignExpand(σ_a)  (invert analysis compand)
            #     → Conv → +128
            self.synthesis_transform = nn.Sequential(
                Softsign(dim=2, num_channels=3, bits=8),              # [0] synth Softsign
                UniformTrainingNoise(k=0),                            # [1] codec quant noise
                ChannelAffine(dim=2, num_channels=3, bias=False),     # [2] γ_s
                ChannelAffine(dim=2, num_channels=3, bias=False),     # [3] γ_s_inv
                SoftsignExpand(dim=2, num_channels=3, bits=8, eps=0.5),  # [4] invert synth
                ChannelAffine(dim=2, num_channels=3, bias=False),     # [5] γ_a_inv
                SoftsignExpand(dim=2, num_channels=3, bits=8, eps=0.5),  # [6] invert analysis
                nn.Conv2d(3, 3, kernel_size=1, bias=True),            # [7] final Conv
            )

        # Random init of the conv. Algebraic-identity init opt-in via
        # `init_algebraic_identity()`.
        conv = self._inverse_conv()
        sigma = 1.0 / math.sqrt(3.0)
        _conv_random_init(conv, sigma)

    def _inverse_conv(self) -> nn.Conv2d:
        """Return the (only) Conv2d in the synthesis Sequential."""
        if self.arch in ("A", "D", "F"):
            return self.synthesis_transform[3]
        if self.arch == "B":
            return self.synthesis_transform[1]
        if self.arch == "E":
            return self.synthesis_transform[1]
        if self.arch == "G":
            return self.synthesis_transform[7]
        raise RuntimeError("unreachable")

    def _continuous(self, z: Tensor) -> Tensor:
        if self.arch == "E":
            z_centered = (z - 128.0) / self.SCALE_E
            y_hat = self.synthesis_transform(z_centered)
            return 127.5 * y_hat + 127.5
        # Variants A/B/D/F: pre-shift -128, post-shift +128
        return self.synthesis_transform(z - 128.0) + 128.0

    def forward(self, z: Tensor) -> Tensor:
        """RGB float output (clamped to `[0, 255]` only in `output_uint8`).
        Boundary noise applied iff `self.training`.
        """
        return self._continuous(z)

    def output_uint8(self, z: Tensor) -> Tensor:
        """Eval-only: clamp + round to uint8 RGB. Caller must set `self.eval()`."""
        x = self._continuous(z)
        return x.clamp(0.0, 255.0).round().to(torch.uint8)

    def init_algebraic_identity(self) -> None:
        """Set parameters to the algebraic-identity configuration.

        For arch A/D/F: Conv = I, bias = 0. ChannelAffine γ = 1 (default).
                       SoftsignExpand σ = `r - 1` (default).
        For arch B: Conv = I, bias = +128 (so post-Hardtanh +128 ≡ identity).
                    The +128 is folded into the outer post-shift; net is
                    `Conv_inv(I) → Hardtanh → +128`, so bias = 0.
                    [Actually: pre-shift is `z - 128`, Conv_inv (I, 0) keeps
                    it, Hardtanh(-128, 127) is identity on this range, +128
                    post-shift recovers z. So bias = 0 here too.]
        For arch E: Conv = I, bias = 0. The +128 shift is replaced by the
                    `127.5·y + 127.5` post-affine; algebraic identity here is
                    the round-4 measured 20.92 dB ceiling.
        """
        conv = self._inverse_conv()
        _conv_set_identity(conv, bias_value=0.0)

    def init_jfif(self) -> None:
        """Set parameters to the JFIF YCbCr → RGB inverse configuration.

        Conv weight set to the JFIF inverse matrix; Conv bias set to 0 so that
        `synthesis_transform(z - 128) + 128 ≈ rgb_hat`. ChannelAffine γ /
        SoftsignExpand σ stay at defaults.
        """
        if self.arch == "E":
            raise NotImplementedError("init_jfif() is not defined for arch=E "
                                      "(round-4 default uses different normalization).")
        conv = self._inverse_conv()
        _conv_set_jfif(conv, kind="inv")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    torch.manual_seed(0)
    x = torch.randint(0, 256, (1, 3, 16, 16)).float()
    for arch in _ARCH_CHOICES:
        fwd = ForwardTransform(arch=arch).eval()
        inv = InverseTransform(arch=arch).eval()
        fwd.init_algebraic_identity()
        inv.init_algebraic_identity()
        z_u8 = fwd.codec_input_uint8(x)  # (1, 3, 16, 16) uint8
        x_hat = inv.output_uint8(z_u8.float())  # (1, 3, 16, 16) uint8
        mse = ((x.float() / 255.0 - x_hat.float() / 255.0) ** 2).mean().item()
        psnr = -10 * math.log10(max(mse, 1e-12))
        print(f"arch={arch}  algebraic-identity no-codec  PSNR={psnr:.2f} dB")

    print()
    for arch in _ARCH_CHOICES:
        if arch == "E":
            continue  # arch E doesn't support init_jfif
        fwd = ForwardTransform(arch=arch).eval()
        inv = InverseTransform(arch=arch).eval()
        fwd.init_jfif()
        inv.init_jfif()
        z_u8 = fwd.codec_input_uint8(x)
        x_hat = inv.output_uint8(z_u8.float())
        mse = ((x.float() / 255.0 - x_hat.float() / 255.0) ** 2).mean().item()
        psnr = -10 * math.log10(max(mse, 1e-12))
        print(f"arch={arch}  JFIF                 no-codec  PSNR={psnr:.2f} dB")
