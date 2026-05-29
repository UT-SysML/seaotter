"""Load the published SEA OTTER phase-2 JPEG transcoder bundle.

The bundle is a single shared `(ForwardTransform, InverseTransform)` pair
plus `K` independent qtables (and the matching `JPEGProxy` instances for
fine-tune warm-start). Layout on disk mirrors the FRAPPE Hugging Face
release: a named subdir with `config.json` and one safetensors file.

Two variants are published under `danjacobellis/seaotter`:

- `seaotter_jpeg_s3` (default) — R16 dual-goal sister checkpoint;
  λ = [0.75, 0.40, 0.22]. This is the warm-start used by every phase-4
  production pipeline (seaft / seab / walft / walsand at `phase2_k=2`).
- `seaotter_jpeg` — R16 dual-goal champion (S2); λ = [0.65, 0.40, 0.22].
  Same shared `(fwd, inv)` family and identical k=1 / k=2 target
  lambdas; only the k=0 row differs structurally (lower-bpp anchor).

Public entry points:

    from seaotter import load_from_hub
    bundle = load_from_hub()                              # S3 (default)
    bundle = load_from_hub(subdir="seaotter_jpeg")        # S2 champion
    bundle = load_from_local("~/hf/seaotter/seaotter_jpeg_s3")

`bundle.encode(image_uint8, k)` is the zero-shot path; `bundle.proxy_at(λ)`
hands back a `JPEGProxy` ready to be plugged into fine-tuning.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor

from .color_transform import ForwardTransform, InverseTransform
from .jpeg_codec import decode as _jpeg_decode
from .jpeg_codec import encode as _jpeg_encode
from .proxy import JPEGProxy


_CONFIG_NAME = "config.json"
_WEIGHTS_NAME = "seaotter_jpeg_pytorch_model.safetensors"


@dataclass
class SEAOTTERJPEGBundle:
    """Loaded phase-2 JPEG bundle (one shared color pair + K qtables)."""

    lambdas: list[float]
    qtables: list[Tensor]           # K × (3, 8, 8) int32, ready for jpeg_codec.encode
    fwd: ForwardTransform           # eval mode, CPU
    inv: InverseTransform           # eval mode, CPU
    proxies: list[JPEGProxy]        # K × JPEGProxy (for fine-tune warm-start)
    config: dict = field(default_factory=dict)

    @property
    def K(self) -> int:
        return len(self.lambdas)

    # -- λ lookup ---------------------------------------------------------
    def _index_of(self, lam: float) -> int:
        for i, l in enumerate(self.lambdas):
            if abs(l - lam) <= 1e-6:
                return i
        raise KeyError(
            f"lambda={lam!r} not in bundle.lambdas={self.lambdas!r}"
        )

    def qtable_at(self, lam: float) -> Tensor:
        return self.qtables[self._index_of(lam)]

    def proxy_at(self, lam: float) -> JPEGProxy:
        return self.proxies[self._index_of(lam)]

    # -- zero-shot transcode ---------------------------------------------
    def encode(self, image_uint8: Tensor, k: int, *, subsampling: int = 0) -> bytes:
        """Encode a 3×H×W uint8 RGB tensor as a JPEG byte string.

        Runs `fwd` to map RGB → codec domain, hard-rounds to uint8, and
        calls `jpeg_codec.encode` with the k-th qtable.
        """
        if image_uint8.dtype != torch.uint8:
            raise TypeError(f"image_uint8 must be uint8, got {image_uint8.dtype}")
        if image_uint8.dim() != 3 or image_uint8.shape[0] != 3:
            raise ValueError(
                f"image_uint8 must be 3×H×W, got {tuple(image_uint8.shape)}"
            )
        x = image_uint8.float().unsqueeze(0)  # (1, 3, H, W)
        with torch.inference_mode():
            z = self.fwd.codec_input_uint8(x)[0]  # (3, H, W) uint8
        return _jpeg_encode(z, self.qtables[k], subsampling=subsampling)

    def decode(self, jpeg_bytes: bytes) -> Tensor:
        """Decode a JPEG byte string back to a 3×H×W uint8 RGB tensor."""
        z_uint8 = _jpeg_decode(jpeg_bytes)  # (3, H, W) uint8
        z = z_uint8.float().unsqueeze(0)
        with torch.inference_mode():
            x = self.inv.output_uint8(z)[0]  # (3, H, W) uint8
        return x


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _build_from_artifacts(cfg: dict, weights: dict[str, Tensor]) -> SEAOTTERJPEGBundle:
    K = int(cfg["K"])
    arch = str(cfg["color_xform_arch"])
    lambdas = [float(x) for x in cfg["lambdas"]]

    fwd = ForwardTransform(arch=arch)
    inv = InverseTransform(arch=arch)
    fwd_sd = {k[len("fwd."):]: v for k, v in weights.items() if k.startswith("fwd.")}
    inv_sd = {k[len("inv."):]: v for k, v in weights.items() if k.startswith("inv.")}
    fwd.load_state_dict(fwd_sd)
    inv.load_state_dict(inv_sd)
    fwd.eval()
    inv.eval()

    qtables = [weights[f"qtable.{i}"].to(torch.int32).contiguous() for i in range(K)]

    proxies: list[JPEGProxy] = []
    for i in range(K):
        proxy = JPEGProxy()
        sd = {
            k[len(f"proxy.{i}."):]: v
            for k, v in weights.items()
            if k.startswith(f"proxy.{i}.")
        }
        proxy.load_state_dict(sd)
        proxies.append(proxy)

    return SEAOTTERJPEGBundle(
        lambdas=lambdas,
        qtables=qtables,
        fwd=fwd,
        inv=inv,
        proxies=proxies,
        config=cfg,
    )


def load_from_local(path: str | os.PathLike) -> SEAOTTERJPEGBundle:
    """Load the bundle from a local directory containing
    `config.json` + `seaotter_jpeg_pytorch_model.safetensors`.
    """
    from safetensors.torch import load_file
    base = Path(path).expanduser()
    with open(base / _CONFIG_NAME) as f:
        cfg = json.load(f)
    weights = load_file(str(base / _WEIGHTS_NAME))
    return _build_from_artifacts(cfg, weights)


def load_from_hub(
    repo_id: str = "danjacobellis/seaotter",
    subdir: str = "seaotter_jpeg_s3",
    *,
    local_files_only: bool = False,
) -> SEAOTTERJPEGBundle:
    """Download (or look up cached) `config.json` + safetensors from the
    Hugging Face hub and return a loaded `SEAOTTERJPEGBundle`.

    Default `subdir="seaotter_jpeg_s3"` is the R16 dual-goal sister
    checkpoint used by every phase-4 production pipeline. Pass
    `subdir="seaotter_jpeg"` for the S2 champion (or any future
    variant published under the same repo).
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{subdir}/{_CONFIG_NAME}",
        local_files_only=local_files_only,
    )
    weights_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{subdir}/{_WEIGHTS_NAME}",
        local_files_only=local_files_only,
    )
    with open(config_path) as f:
        cfg = json.load(f)
    weights = load_file(weights_path)
    return _build_from_artifacts(cfg, weights)


# ===========================================================================
# Full task-specific pipeline bundle (schema "seaotter-pipeline-v1")
# ===========================================================================
#
# `SEAOTTERJPEGBundle` above is the *warm-start transcoder only* — the shared
# color pair + K qtables, with no FRAPPE decoder. A task-specific SEAOTTER
# pipeline additionally carries the fine-tuned FRAPPE decoder G_S, so a single
# bundle reconstructs the full headline pipeline of Eq. (1) end-to-end:
#
#     transcode (cloud, one-time):  x --G_A--> int8 latents --C--> G_S --> F --J_Q encode--> JPEG
#     decode    (consumer, steady): JPEG --J_Q decode--> [F^{-1}] --> RGB
#
# The frozen FRAPPE encoder G_A is *referenced* (frappe_encoder_repo /
# frappe_encoder_subdir + frappe_n_ch), never duplicated — it is shared and
# un-fine-tuned across every operating point, so the bundle ships only the
# ~57M-parameter fine-tuned decoder plus the tiny color pair / qtable.


def _check_chw_uint8(image_uint8: Tensor) -> Tensor:
    if image_uint8.dtype != torch.uint8:
        raise TypeError(f"image must be uint8, got {image_uint8.dtype}")
    if image_uint8.dim() != 3 or image_uint8.shape[0] != 3:
        raise ValueError(f"image must be 3xHxW, got {tuple(image_uint8.shape)}")
    return image_uint8


@dataclass
class PipelinePackage:
    """A full task-specific SEAOTTER pipeline loaded from a pipeline bundle.

    Bundles the frozen FRAPPE encoder + fine-tuned decoder (G_A / G_S), the
    learned color pair (F, F^{-1}), and the single-op JPEG qtable. Exposes the
    deployed one-time transcode and the steady-state consumer decode.

        from seaotter import load_pipeline_from_hub
        pipe = load_pipeline_from_hub(subdir="seaotter_cls")
        jpeg = pipe.transcode(image_uint8)        # cloud one-time transcode
        rgb  = pipe.decode(jpeg)                   # consumer steady-state decode
        rgb  = pipe.reconstruct(image_uint8)       # end-to-end (== decode(transcode(.)))
    """

    frappe: object                  # seaotter.train.codecs.FrappeBundle (encoder frozen)
    fwd: ForwardTransform           # eval mode
    inv: InverseTransform           # eval mode
    proxy: JPEGProxy                # carries Q_unconstrained (qtable below is derived)
    qtable: Tensor                  # (3, 8, 8) int32, ready for jpeg_codec.encode
    config: dict = field(default_factory=dict)
    device: str = "cpu"

    @property
    def frappe_n_ch(self) -> int:
        return int(self.config["frappe_n_ch"])

    # -- sensor side -----------------------------------------------------
    def sensor_encode(self, image_uint8: Tensor) -> list:
        """G_A only: RGB uint8 (3,H,W) -> list of int8 latent tensors (what
        crosses the wireless uplink). The encoder is frozen."""
        from .train.codecs import _frappe_encode_int8
        x = _check_chw_uint8(image_uint8).to(self.device).float().unsqueeze(0)
        with torch.inference_mode():
            return _frappe_encode_int8(self.frappe.model, x / 127.5 - 1.0)

    # -- cloud one-time transcode ----------------------------------------
    def transcode(self, image_uint8: Tensor, *, subsampling: int = 0) -> bytes:
        """One-time cloud transcode G_A -> C -> G_S -> F -> JPEG encode.

        Returns the on-disk, standards-compliant JPEG byte string. Mirrors
        `train.codecs.frappe_pipeline_forward` up to the JPEG-encode boundary
        (deployed path: hard-rounded uint8 into a real libjpeg encode, not the
        differentiable proxy)."""
        from .train.codecs import _frappe_encode_int8
        x = _check_chw_uint8(image_uint8).to(self.device).unsqueeze(0)
        with torch.inference_mode():
            latents = _frappe_encode_int8(self.frappe.model, x.float() / 127.5 - 1.0)
            rgb_decoded = self.frappe.model.decode(latents).clamp(-1.0, 1.0)
            rgb_codec_in = (rgb_decoded + 1.0) * 127.5
            z = self.fwd.codec_input_uint8(rgb_codec_in)[0]   # (3,H,W) uint8
        return _jpeg_encode(z.cpu(), self.qtable, subsampling=subsampling)

    # -- consumer steady-state decode ------------------------------------
    def decode(self, jpeg_bytes: bytes, *, apply_inverse: bool = True) -> Tensor:
        """Steady-state consumer decode: vanilla JPEG decode (+ optional
        F^{-1}). `apply_inverse=False` returns the raw codec-domain uint8 for
        JPEG-domain consumers that fold F^{-1} into their first layer."""
        z_uint8 = _jpeg_decode(jpeg_bytes)            # (3,H,W) uint8, CPU
        if not apply_inverse:
            return z_uint8
        z = z_uint8.to(self.device).float().unsqueeze(0)
        with torch.inference_mode():
            return self.inv.output_uint8(z)[0]

    # -- end-to-end ------------------------------------------------------
    def reconstruct(self, image_uint8: Tensor, *, subsampling: int = 0) -> Tensor:
        """End-to-end transcode + steady-state decode -> uint8 RGB (3,H,W).
        This is the input a downstream consumer backbone sees; run your
        task model on `pipe.reconstruct(x)` to reproduce the downstream eval."""
        return self.decode(self.transcode(image_uint8, subsampling=subsampling))


def _build_pipeline_from_artifacts(cfg: dict, weights: dict, device: str = "cpu") -> PipelinePackage:
    from .train.codecs import load_frappe

    arch = str(cfg["color_xform_arch"])
    n_ch = int(cfg["frappe_n_ch"])

    # Frozen FRAPPE encoder (referenced, not in this bundle) + fine-tuned decoder.
    frappe = load_frappe(n_ch, device)
    dec_sd = {k[len("decoder."):]: v for k, v in weights.items() if k.startswith("decoder.")}
    frappe.model.decoder.load_state_dict(dec_sd)
    frappe.model.eval()

    fwd = ForwardTransform(arch=arch)
    inv = InverseTransform(arch=arch)
    proxy = JPEGProxy()
    fwd.load_state_dict({k[len("fwd."):]: v for k, v in weights.items() if k.startswith("fwd.")})
    inv.load_state_dict({k[len("inv."):]: v for k, v in weights.items() if k.startswith("inv.")})
    proxy.load_state_dict({k[len("proxy."):]: v for k, v in weights.items() if k.startswith("proxy.")})
    fwd.to(device).eval()
    inv.to(device).eval()
    proxy.to(device).eval()

    if "qtable" in weights:
        qtable = weights["qtable"].to(torch.int32).contiguous()
    else:
        qtable = torch.clamp(proxy.qtable().round(), 1, 255).to(torch.int32)

    return PipelinePackage(
        frappe=frappe, fwd=fwd, inv=inv, proxy=proxy,
        qtable=qtable, config=cfg, device=str(device),
    )


def load_pipeline_from_local(path: str | os.PathLike, *, device: str = "cpu") -> PipelinePackage:
    """Load a full SEAOTTER pipeline from a local directory containing
    `config.json` (schema `seaotter-pipeline-v1`) + the safetensors named by
    its `weights_file` field. The frozen FRAPPE encoder is pulled from the
    repo/subdir recorded in the config."""
    from safetensors.torch import load_file
    base = Path(path).expanduser()
    with open(base / _CONFIG_NAME) as f:
        cfg = json.load(f)
    weights = load_file(str(base / cfg["weights_file"]))
    return _build_pipeline_from_artifacts(cfg, weights, device=device)


def load_pipeline_from_hub(
    repo_id: str = "danjacobellis/seaotter",
    subdir: str = "seaotter_cls",
    *,
    device: str = "cpu",
    local_files_only: bool = False,
) -> PipelinePackage:
    """Download (or look up cached) a full SEAOTTER pipeline bundle from the
    Hugging Face hub and return a loaded `PipelinePackage`.

    Default `subdir="seaotter_cls"` is the headline ImageNet-classification
    pipeline (FRAPPE n=12, fine-tuned decoder + sandwich; reported top-1
    0.69024). The frozen FRAPPE encoder is fetched separately from the
    repo/subdir recorded in the bundle config."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    config_path = hf_hub_download(
        repo_id=repo_id, filename=f"{subdir}/{_CONFIG_NAME}",
        local_files_only=local_files_only,
    )
    with open(config_path) as f:
        cfg = json.load(f)
    weights_path = hf_hub_download(
        repo_id=repo_id, filename=f"{subdir}/{cfg['weights_file']}",
        local_files_only=local_files_only,
    )
    weights = load_file(weights_path)
    return _build_pipeline_from_artifacts(cfg, weights, device=device)
