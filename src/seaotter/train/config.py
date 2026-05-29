"""TrainerConfig — frozen dataclass capturing every recipe + runtime knob.

The six recipes under `seaotter.train.recipes` instantiate this with their
locked values; the Trainer reads it as a closed-world record.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional


PHASE2_K3_PATH_DEFAULT = (
    "/home/dgj335/danjacobellis/seaotter/shared_color/experiments/"
    "round16_dual_goal/iter8/S3_K3_lams_0p75_0p4_0p22_w_0p3_0p7_1p5.pth"
)
PHASE2_K_DEFAULT = 2

CLIP_NAFLEX_MAX_PATCHES = 256
CLIP_NAFLEX_PATCH_SIZE = 16
CLIP_NAFLEX_SNAP = 32

CROP_FOR_TASK: dict[str, Optional[int]] = {"cls": 384, "seg": 512, "clip": None}
NUM_CLASSES_FOR_TASK = {"cls": 1000, "seg": 150, "clip": 1000}


@dataclass(frozen=True)
class TrainerConfig:
    # ---- recipe knobs ---------------------------------------------------
    codec: Literal["frappe", "walloc"]
    task: Literal["cls", "seg", "clip"]

    # Operating point — exactly one of these is set per codec.
    frappe_n_ch: Optional[int] = None
    walloc_pixel_ratio: Optional[float] = None

    # Sandwich warm-start
    phase2_init: str = PHASE2_K3_PATH_DEFAULT
    phase2_k: int = PHASE2_K_DEFAULT

    # Loss / LR
    lam: float = 0.10
    lr_base: float = 2e-4
    lr_ratio_q: float = 0.5
    lr_ratio_x: float = 1.0
    alpha_clip: Optional[float] = None  # required iff task=="clip"

    # Schedule / data
    batch_size: int = 8
    epochs: int = 1
    train_ds: str = "timm/imagenet-1k-wds"
    calib_ds: str = "danjacobellis/kodak"
    crop_size: Optional[int] = None  # None ⇒ task default (clip = variable)
    num_workers: int = 8
    calib_n: int = 16
    dataset_samples: Optional[int] = None

    # Optimizer / schedule shape
    optimizer: Literal["adan_caution"] = "adan_caution"
    schedule: Literal["raised_cosine"] = "raised_cosine"
    lr_pow: float = 4.0
    min_lr: float = 1e-8
    grad_clip: float = 5.0

    # Runtime
    device: str = "cuda:0"
    exp_name: str = "unnamed"
    out_dir: str = "."
    seed: int = 0
    print_every: int = 500

    # Aux
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.codec not in ("frappe", "walloc"):
            raise ValueError(f"codec={self.codec!r} must be 'frappe' or 'walloc'")
        if self.task not in ("cls", "seg", "clip"):
            raise ValueError(f"task={self.task!r} must be cls/seg/clip")

        # Codec ↔ operating-point cross-check.
        if self.codec == "frappe":
            if self.frappe_n_ch is None:
                raise ValueError("frappe codec requires frappe_n_ch")
            if self.walloc_pixel_ratio is not None:
                raise ValueError("walloc_pixel_ratio set but codec=frappe")
        else:
            if self.walloc_pixel_ratio is None:
                raise ValueError("walloc codec requires walloc_pixel_ratio")
            if self.frappe_n_ch is not None:
                raise ValueError("frappe_n_ch set but codec=walloc")

        # Crop ↔ task cross-check.
        default_crop = CROP_FOR_TASK[self.task]
        if self.task == "clip":
            if self.crop_size is not None:
                raise ValueError(
                    "clip task uses variable per-image (H', W'); crop_size must be None"
                )
        else:
            expected = default_crop
            if self.crop_size is None:
                object.__setattr__(self, "crop_size", expected)
            elif self.crop_size != expected:
                # Allow override but only with a multiple of 32 for FRAPPE max_ps.
                if self.crop_size % 32 != 0:
                    raise ValueError(
                        f"crop_size={self.crop_size} must be a multiple of 32"
                    )

        # alpha_clip iff clip.
        if self.task == "clip":
            if self.alpha_clip is None:
                raise ValueError("task=clip requires alpha_clip (default 0.7)")
            if not 0.0 <= self.alpha_clip <= 1.0:
                raise ValueError(f"alpha_clip={self.alpha_clip} must be in [0, 1]")
            if self.batch_size != 1:
                raise ValueError(
                    f"task=clip forces batch_size=1; got {self.batch_size}"
                )
        else:
            if self.alpha_clip is not None:
                raise ValueError(f"alpha_clip set on non-clip task={self.task}")

    # Convenience
    def to_dict(self) -> dict:
        return asdict(self)
