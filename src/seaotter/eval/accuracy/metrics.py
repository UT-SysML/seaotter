"""Distortion metric helpers + bpp + dB conventions.

- piq.psnr / piq.ssim consumed in [0, 1] (data_range=1.0).
- piq.LPIPS / piq.DISTS networks resident on device; called in [0, 1].
- dB convention for perceptual metrics: `-10 * log10(metric)`.
- bpp: `8 * n_bytes / (H * W)` per image.

Accumulator class so 50k images don't have to all be kept in memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import piq
import torch


def uint8_to_unit(x_uint8: torch.Tensor) -> torch.Tensor:
    return x_uint8.to(torch.float32) / 255.0


def bpp_for_image(n_bytes: int, h: int, w: int) -> float:
    return 8.0 * n_bytes / (h * w)


def db_from_metric(value: float) -> float:
    """dB convention used in iter-3/iter-4 R-D tables for LPIPS/DISTS.

    -10 * log10(value). Clamps `value` to a tiny positive number so an
    exact-match recon (metric = 0) doesn't blow up.
    """
    return -10.0 * math.log10(max(value, 1e-12))


@dataclass
class DistortionAccumulator:
    """Running aggregator for PSNR / SSIM / LPIPS / DISTS over the val set.

    PSNR is computed per-image (not per-batch) so single-image variation is
    preserved. SSIM/LPIPS/DISTS run on a per-batch basis to keep the
    learned-metric forwards efficient; the per-image figure is the batch
    mean, weighted by batch size at the aggregate step.
    """

    device: torch.device
    psnr_sum: float = 0.0
    ssim_sum: float = 0.0
    lpips_sum: float = 0.0
    dists_sum: float = 0.0
    n: int = 0
    _lpips_net: piq.LPIPS = field(default=None, init=False, repr=False)
    _dists_net: piq.DISTS = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._lpips_net = piq.LPIPS().to(self.device).eval()
        self._dists_net = piq.DISTS().to(self.device).eval()

    @torch.no_grad()
    def update(
        self,
        recon_uint8: torch.Tensor,
        ref_uint8: torch.Tensor,
    ) -> None:
        """recon, ref both (B, 3, H, W) uint8 on `self.device`."""
        if recon_uint8.shape != ref_uint8.shape:
            raise ValueError(
                f"shape mismatch: recon={tuple(recon_uint8.shape)} "
                f"ref={tuple(ref_uint8.shape)}"
            )
        recon_f = uint8_to_unit(recon_uint8)
        ref_f = uint8_to_unit(ref_uint8)
        B = recon_f.shape[0]

        # PSNR per image; sum so we can average at the end.
        mse_per_img = (recon_f - ref_f).pow(2).mean(dim=(1, 2, 3))
        psnr_per_img = -10.0 * torch.log10(mse_per_img.clamp_min(1e-12))
        self.psnr_sum += float(psnr_per_img.sum().item())

        # SSIM via piq.ssim on the full batch, then multiply by B.
        ssim_batch = piq.ssim(recon_f, ref_f, data_range=1.0, reduction="mean")
        self.ssim_sum += float(ssim_batch.item()) * B

        lpips_batch = self._lpips_net(recon_f, ref_f)
        self.lpips_sum += float(lpips_batch.item()) * B

        dists_batch = self._dists_net(recon_f, ref_f)
        self.dists_sum += float(dists_batch.item()) * B

        self.n += B

    def finalize(self) -> dict[str, float]:
        if self.n == 0:
            return {"psnr_db": None, "ssim": None, "lpips_db": None, "dists_db": None}
        return {
            "psnr_db": self.psnr_sum / self.n,
            "ssim": self.ssim_sum / self.n,
            "lpips_db": db_from_metric(self.lpips_sum / self.n),
            "dists_db": db_from_metric(self.dists_sum / self.n),
        }
