__version__ = "1.0.0"

from .jpeg_codec import encode, decode, read_qtables
from .hub import (
    SEAOTTERJPEGBundle,
    load_from_hub,
    load_from_local,
    PipelinePackage,
    load_pipeline_from_hub,
    load_pipeline_from_local,
)

__all__ = [
    "__version__",
    # JPEG codec primitives
    "encode",
    "decode",
    "read_qtables",
    # phase-2 warm-start transcoder bundle (shared color pair + K qtables)
    "SEAOTTERJPEGBundle",
    "load_from_hub",
    "load_from_local",
    # full task-specific pipeline (adds the fine-tuned FRAPPE decoder)
    "PipelinePackage",
    "load_pipeline_from_hub",
    "load_pipeline_from_local",
]
