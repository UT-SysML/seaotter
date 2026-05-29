"""Per-codec encoder adapters for the encode-complexity harness.

Each adapter exposes::

    name              short slug used in output JSON filenames
    family            "pillow" | "frappe" | "walloc" | "seaotter_jpeg"
    stages            list of stage names timed inside the loop (FRAPPE
                      uses analysis/transfer/store; WaLLoC uses analysis
                      /store; Pillow uses encode; SEAOTTER standalone JPEG
                      uses fwd/store).
    prepare(pil_img)  pre-stage the encoder's native input (PIL.Image
                      for Pillow codecs; float tensor for FRAPPE/WaLLoC;
                      uint8 RGB tensor for SEAOTTER standalone JPEG).
                      Excluded from timing.
    encode(input)     timed by the caller via wallclock contexts; the
                      adapter exposes a `run_stages(wallclock, input)`
                      method that contains the per-stage breakdown.
    config_block()    dict for the output JSON's ``config`` field.
    op_slug()         filename slug (n12 / q5 / r100 / p16 / k0).

The harness is responsible for the warmup/measurement loop; the adapter
just performs one timed encode of one input.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from typing import Any, Callable

import PIL.Image
import torch
from torchvision.transforms.v2.functional import pil_to_tensor


# ----------------------------------------------------------------------
# Base adapter
# ----------------------------------------------------------------------

class CodecAdapter:
    """Common interface for all encoder adapters."""

    name: str
    family: str
    stages: tuple[str, ...]

    def __init__(self, op: dict, *, device: str = "cpu"):
        self.op = op
        self.device = device

    def prepare(self, pil_img: PIL.Image.Image) -> Any:
        """Pre-stage one input in the encoder's native form."""
        raise NotImplementedError

    def run_stages(self, wallclock, native_input) -> bytes:
        """Run the encoder forward + native quant/entropy coding for one
        input, wrapped in ``wallclock(<stage>)`` contexts. Returns the
        encoded bytes (used only for sanity checks / blob size logging).
        """
        raise NotImplementedError

    def run_stages_untimed(self, native_input) -> bytes:
        """Same body as :meth:`run_stages` but without ``wallclock`` calls.
        Used for warmup passes."""
        raise NotImplementedError

    def config_block(self) -> dict:
        return {}

    def op_slug(self) -> str:
        raise NotImplementedError


# ----------------------------------------------------------------------
# Pillow-family adapter (AVIF / AVIFx / JPEG / JPEG sub0 / WebP / JP2)
# ----------------------------------------------------------------------

class PillowAdapter(CodecAdapter):
    """Encoder native input is a `PIL.Image`; the timed block is
    `Image.save(BytesIO, **save_kwargs)`."""

    family = "pillow"
    stages = ("encode",)

    def __init__(self, *, op: dict, save_kwargs: dict, codec_name: str, slug: str,
                 device: str = "cpu"):
        super().__init__(op=op, device=device)
        self.name = codec_name
        self._save_kwargs = dict(save_kwargs)
        self._slug = slug

    def prepare(self, pil_img: PIL.Image.Image) -> PIL.Image.Image:
        # Load the pixels off disk once so the timed `save` call only
        # exercises the encoder. The PIL JPEG decoder is lazy by default.
        rgb = pil_img.convert("RGB")
        rgb.load()
        return rgb

    def run_stages(self, wallclock, native_input: PIL.Image.Image) -> bytes:
        with wallclock("encode"):
            buf = io.BytesIO()
            native_input.save(buf, **self._save_kwargs)
        return buf.getvalue()

    def run_stages_untimed(self, native_input: PIL.Image.Image) -> bytes:
        buf = io.BytesIO()
        native_input.save(buf, **self._save_kwargs)
        return buf.getvalue()

    def config_block(self) -> dict:
        return {
            "backend": "pillow",
            **self._save_kwargs,
        }

    def op_slug(self) -> str:
        return self._slug


def make_pillow_adapter(short: str, op: dict, *, device: str = "cpu") -> PillowAdapter:
    """Construct the right Pillow adapter from a (short, op) pair.

    Supported shorts: avif, avifx, jpeg, jpeg_sub0, webp, jp2.
    """
    t = op["type"]
    v = op["value"]
    if short == "avif":
        if t != "quality":
            raise ValueError(f"avif: expected op.type=quality, got {t}")
        q = int(v)
        return PillowAdapter(
            op=op,
            save_kwargs={"format": "AVIF", "quality": q},
            codec_name="avif",
            slug=f"q{q}",
            device=device,
        )
    if short == "avifx":
        if t != "quality":
            raise ValueError(f"avifx: expected op.type=quality, got {t}")
        q = int(v)
        speed = int(op.get("extras", {}).get("avif_speed", 10))
        return PillowAdapter(
            op=op,
            save_kwargs={"format": "AVIF", "quality": q, "speed": speed},
            codec_name="avifx",
            slug=f"q{q}_s{speed}",
            device=device,
        )
    if short == "jpeg":
        # iter-6 vanilla JPEG: subsampling=0 (4:4:4) is the iter-6
        # convention shipped in `harness/pipelines/jpeg.py`. The prompt
        # asks for both `jpeg` (4:2:0 default) and `jpeg_sub0` (4:4:4)
        # so we honour the prompt: short `jpeg` => 4:2:0 (default).
        if t != "quality":
            raise ValueError(f"jpeg: expected op.type=quality, got {t}")
        q = int(v)
        return PillowAdapter(
            op=op,
            save_kwargs={"format": "JPEG", "quality": q},
            codec_name="jpeg",
            slug=f"q{q}",
            device=device,
        )
    if short == "jpeg_sub0":
        if t != "quality":
            raise ValueError(f"jpeg_sub0: expected op.type=quality, got {t}")
        q = int(v)
        return PillowAdapter(
            op=op,
            save_kwargs={"format": "JPEG", "quality": q, "subsampling": 0},
            codec_name="jpeg_sub0",
            slug=f"q{q}",
            device=device,
        )
    if short == "webp":
        if t != "quality":
            raise ValueError(f"webp: expected op.type=quality, got {t}")
        q = int(v)
        return PillowAdapter(
            op=op,
            save_kwargs={"format": "WEBP", "quality": q},
            codec_name="webp",
            slug=f"q{q}",
            device=device,
        )
    if short == "jp2":
        if t != "rate":
            raise ValueError(f"jp2: expected op.type=rate, got {t}")
        r = float(v)
        return PillowAdapter(
            op=op,
            save_kwargs={
                "format": "JPEG2000",
                "quality_mode": "rates",
                "quality_layers": [r],
            },
            codec_name="jp2",
            slug=f"r{int(r) if abs(r - int(r)) < 1e-9 else r}",
            device=device,
        )
    raise ValueError(f"unsupported pillow short {short!r}")


# ----------------------------------------------------------------------
# FRAPPE adapter
# ----------------------------------------------------------------------

def _load_hybrid_v2_entropy():
    """Load the hybrid-v2 entropy coder (jpegls + zstd fallback) from
    `compressors/experiments/encoder_optimization/hybrid_v2.py`.

    This is the same entropy module pinned in the reference FRAPPE
    encode-complexity JSON at
    ``compressors/results/frappe/encode_1777342722.json``. The
    pillow_jpls reference path in ``compressors.frappe.entropy_coding``
    is ~4x slower on this CPU. Bpp is within ~1-2 % of the pillow_jpls
    path at typical operating points; bytes are not bit-identical but
    decode round-trip is bit-exact.
    """
    import importlib.util
    path = (
        "/home/dgj335/danjacobellis/compressors/experiments/"
        "encoder_optimization/hybrid_v2.py"
    )
    spec = importlib.util.spec_from_file_location("_seaotter_eval_hybrid_v2", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load hybrid_v2 from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FrappeAdapter(CodecAdapter):
    """FRAPPE encoder: analysis (forward + int8 quantize), transfer
    (device->host; no-op on CPU), store (arrange + entropy coding via
    the hybrid-v2 module = JPEG-LS for big scales + zstd for tiny ones).

    Uses the production hybrid-v2 entropy coder
    (``compressors/experiments/encoder_optimization/hybrid_v2.py``),
    matching the FRAPPE reference encode-complexity JSON
    (``compressors/results/frappe/encode_1777342722.json``). The
    default pillow_jpls path in ``compressors.frappe.entropy_coding``
    is ~4x slower on this CPU and is the slow reference
    implementation; the SEAOTTER paper's encoder bytes are identical
    modulo the entropy-coder choice (decode round-trips bit-exactly
    in both).
    """

    family = "frappe"
    stages = ("analysis", "transfer", "store")

    def __init__(self, *, op: dict, device: str = "cpu"):
        super().__init__(op=op, device=device)
        if op["type"] != "n_ch":
            raise ValueError(f"frp: expected op.type=n_ch, got {op['type']!r}")
        self.n_ch = int(op["value"])
        self._slug = f"n{self.n_ch}"

        from compressors.frappe.model import load_from_hub, load_progressive_model
        cfg, weights, n_trained = load_from_hub()
        if not 1 <= self.n_ch <= n_trained:
            raise ValueError(f"n_ch must be in [1, {n_trained}], got {self.n_ch}")
        self._cfg = cfg
        self._linear_input = bool(getattr(cfg, "linear_input", False))
        self._model = load_progressive_model(weights, cfg, self.n_ch, device).eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

        # Entropy coder: hybrid-v2 (JPEG-LS + zstd fallback). Matches the
        # FRAPPE reference harness JSON.
        self._ec = _load_hybrid_v2_entropy()

        self.name = "frp"

    def prepare(self, pil_img: PIL.Image.Image) -> torch.Tensor:
        rgb = pil_img.convert("RGB")
        rgb.load()
        x_u8 = pil_to_tensor(rgb).unsqueeze(0)  # (1, 3, H, W)
        x = x_u8.to(torch.float32) / 127.5 - 1.0
        if self._linear_input:
            from compressors.frappe.quantize import srgb_to_linear
            x = srgb_to_linear(x)
        return x.to(self.device).contiguous()

    def run_stages(self, wallclock, native_input: torch.Tensor) -> bytes:
        ec = self._ec
        with wallclock("analysis"):
            with torch.inference_mode():
                latents = self._model.encode(native_input)
                latents_q = [z.round().clamp(-127, 127).to(torch.int8)
                             for z in latents]
        with wallclock("transfer"):
            latents_cpu = [z.cpu() for z in latents_q]
        with wallclock("store"):
            arranged = ec.arrange_latents(latents_cpu)
            blob = ec.encode_latents(arranged)
        return blob

    def run_stages_untimed(self, native_input: torch.Tensor) -> bytes:
        ec = self._ec
        with torch.inference_mode():
            latents = self._model.encode(native_input)
            latents_q = [z.round().clamp(-127, 127).to(torch.int8) for z in latents]
        latents_cpu = [z.cpu() for z in latents_q]
        arranged = ec.arrange_latents(latents_cpu)
        return ec.encode_latents(arranged)

    def config_block(self) -> dict:
        return {
            "backend": "frappe",
            "n_ch": self.n_ch,
            "linear_input": self._linear_input,
            "entropy_coding": "hybrid_v2 (JPEG-LS + zstd fallback)",
        }

    def op_slug(self) -> str:
        return self._slug


# ----------------------------------------------------------------------
# WaLLoC adapter
# ----------------------------------------------------------------------

class WallocAdapter(CodecAdapter):
    """WaLLoC encoder: analysis (wavelet + encoder forward + integer
    round), transfer (device->host; no-op on CPU), store (latent
    packed to WebP-lossless bytes via the bundled recipe).

    op.type=pixel_ratio; op.value in (0, 100] (percentage of full
    spatial resolution, controlling the snap shape).
    """

    family = "walloc"
    stages = ("analysis", "transfer", "store")

    def __init__(self, *, op: dict, device: str = "cpu"):
        super().__init__(op=op, device=device)
        if op["type"] != "pixel_ratio":
            raise ValueError(f"wal: expected op.type=pixel_ratio, got {op['type']!r}")
        self.q = float(op["value"])
        if not 0 < self.q <= 100:
            raise ValueError(f"wal: pixel_ratio in (0, 100], got {self.q}")
        # Slug rule: integer for integer-valued ops, otherwise dotted
        # (e.g. 10p5). Matches iter-6 convention.
        if abs(self.q - round(self.q)) < 1e-9:
            self._slug = f"p{int(round(self.q))}"
        else:
            self._slug = f"p{str(self.q).replace('.', 'p')}"

        from compressors.walloc._codec import load_codec, SNAP_MULTIPLE
        self._SNAP_MULTIPLE = SNAP_MULTIPLE
        self._codec, self._info = load_codec(device=device, torch_dtype=torch.float32)

        self.name = "wal"

    def _snap_hw(self, h: int, w: int) -> tuple[int, int]:
        from compressors.walloc._codec import snap_shape
        if h % self._SNAP_MULTIPLE or w % self._SNAP_MULTIPLE:
            raise ValueError(
                f"wal: H, W must be multiples of {self._SNAP_MULTIPLE}, got {h}x{w}"
            )
        return snap_shape(h, w, self.q)

    def prepare(self, pil_img: PIL.Image.Image) -> torch.Tensor:
        from compressors.walloc._codec import resize_bicubic, to_model_input
        rgb = pil_img.convert("RGB")
        rgb.load()
        x_u8 = pil_to_tensor(rgb).unsqueeze(0)  # (1, 3, H, W)
        x = x_u8.to(torch.float32) / 255.0
        _, _, H, W = x.shape
        snap_h, snap_w = self._snap_hw(H, W)
        x_r = resize_bicubic(x, (snap_h, snap_w)).clamp(0, 1)
        x_in = to_model_input(x_r)
        return x_in.to(self.device).contiguous()

    def run_stages(self, wallclock, native_input: torch.Tensor) -> bytes:
        from compressors.walloc._codec import encode_to_latent, latent_to_webp_bytes
        with wallclock("analysis"):
            with torch.inference_mode():
                z_hat = encode_to_latent(self._codec, native_input)
        with wallclock("transfer"):
            z_hat_cpu = z_hat.cpu()
        with wallclock("store"):
            blob = latent_to_webp_bytes(z_hat_cpu, self._info.latent_bits)
        return blob

    def run_stages_untimed(self, native_input: torch.Tensor) -> bytes:
        from compressors.walloc._codec import encode_to_latent, latent_to_webp_bytes
        with torch.inference_mode():
            z_hat = encode_to_latent(self._codec, native_input)
        z_hat_cpu = z_hat.cpu()
        return latent_to_webp_bytes(z_hat_cpu, self._info.latent_bits)

    def config_block(self) -> dict:
        return {
            "backend": "walloc",
            "checkpoint": "RGB_16x",
            "pixel_ratio": self.q,
            "latent_dim": self._info.latent_dim,
            "latent_bits": self._info.latent_bits,
            "J": self._info.J,
        }

    def op_slug(self) -> str:
        return self._slug


# ----------------------------------------------------------------------
# SEAOTTER standalone JPEG adapter
# ----------------------------------------------------------------------

class SeaOtterJpegAdapter(CodecAdapter):
    """SEAOTTER standalone learned JPEG: forward color transform (uint8
    input -> codec_input_uint8 via `bundle.fwd`) + jpeg_codec.encode
    with the k-th learned qtable.

    op.type=k; op.value in {0, 1, 2} (which entry of the bundle to use).
    Native input: 3xHxW uint8 RGB tensor on CPU. The forward transform
    is run on CPU.
    """

    family = "seaotter_jpeg"
    stages = ("fwd", "store")

    def __init__(self, *, op: dict, device: str = "cpu"):
        super().__init__(op=op, device=device)
        if op["type"] != "k":
            raise ValueError(f"seaotter_jpeg: expected op.type=k, got {op['type']!r}")
        self.k = int(op["value"])
        self._slug = f"k{self.k}"
        self._subdir = op.get("extras", {}).get("subdir", "seaotter_jpeg_s3")

        from seaotter.hub import load_from_hub as sea_load_from_hub
        self._bundle = sea_load_from_hub(subdir=self._subdir)
        if not 0 <= self.k < self._bundle.K:
            raise ValueError(
                f"seaotter_jpeg: k must be in [0, {self._bundle.K}), got {self.k}"
            )
        # Pin transforms to the target device (CPU here) and verify.
        self._bundle.fwd.to(device).eval()
        self._bundle.inv.to(device).eval()
        for p in self._bundle.fwd.parameters():
            p.requires_grad_(False)
        for p in self._bundle.inv.parameters():
            p.requires_grad_(False)
        self._qtable = self._bundle.qtables[self.k].to(torch.int32)

        self.name = "seaotter_jpeg"

    def prepare(self, pil_img: PIL.Image.Image) -> torch.Tensor:
        rgb = pil_img.convert("RGB")
        rgb.load()
        x_u8 = pil_to_tensor(rgb)  # (3, H, W) uint8
        return x_u8.contiguous()

    def run_stages(self, wallclock, native_input: torch.Tensor) -> bytes:
        from seaotter.jpeg_codec import encode as jpeg_encode
        with wallclock("fwd"):
            with torch.inference_mode():
                x = native_input.float().unsqueeze(0)
                z = self._bundle.fwd.codec_input_uint8(x)[0]
        with wallclock("store"):
            blob = jpeg_encode(z, self._qtable, subsampling=0)
        return blob

    def run_stages_untimed(self, native_input: torch.Tensor) -> bytes:
        from seaotter.jpeg_codec import encode as jpeg_encode
        with torch.inference_mode():
            x = native_input.float().unsqueeze(0)
            z = self._bundle.fwd.codec_input_uint8(x)[0]
        return jpeg_encode(z, self._qtable, subsampling=0)

    def config_block(self) -> dict:
        return {
            "backend": "seaotter_jpeg",
            "subdir": self._subdir,
            "k": self.k,
            "lambda": self._bundle.lambdas[self.k],
        }

    def op_slug(self) -> str:
        return self._slug


# ----------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------

PILLOW_SHORTS = {"avif", "avifx", "jpeg", "jpeg_sub0", "webp", "jp2"}


def make_adapter(short: str, op: dict, *, device: str = "cpu") -> CodecAdapter:
    """Construct a codec adapter from a short code + op-point dict."""
    if short in PILLOW_SHORTS:
        return make_pillow_adapter(short, op, device=device)
    if short == "frp":
        return FrappeAdapter(op=op, device=device)
    if short == "wal":
        return WallocAdapter(op=op, device=device)
    if short == "seaotter_jpeg":
        return SeaOtterJpegAdapter(op=op, device=device)
    raise ValueError(f"unknown codec short {short!r}")
