"""Shared `Pipeline` interface every codec/pipeline implements.

Two execution paths share this API:

1. **Accuracy + distortion** (full val) — collate-side encode runs in
   DataLoader workers (`collate_encode`), GPU-side decode batches all
   blobs together (`decode_blobs_batch`). This pipelines CPU codec
   throughput with GPU teacher forwards. Distortion is computed by
   comparing the recon batch to the same uint8 batch fed to the codec
   (returned alongside the blobs by the worker).

2. **Throughput** (256-image subset, single-thread per-image) —
   `encode_only_cpu` and `decode_only_consumer` measure sensor and
   consumer cost respectively.

All recon tensors are uint8 (B, 3, H, W) on `self.device`. Vanilla
codecs return `storage_bytes=None` (since transmission == storage);
the SEA OTTER rows return separate `transmit_bytes` and `storage_bytes`.
"""

from __future__ import annotations

import PIL.Image
import torch


class Pipeline:
    short: str = "<base>"

    def __init__(self, op: dict, task: str, device: torch.device):
        self.op = op
        self.task = task
        self.device = device

    # ---- accuracy path (workers + main loop) ----------------------------

    @torch.no_grad()
    def collate_encode(self, pil_img: PIL.Image.Image) -> bytes:
        """Worker-side encode for one image — returns the transmission blob.

        Called from the DataLoader collate_fn. Must be safe to call from
        a fork-spawned worker process.
        """
        raise NotImplementedError

    @torch.no_grad()
    def decode_blobs_batch(
        self, blobs: list[bytes],
    ) -> tuple[torch.Tensor, list[int], list[int] | None]:
        """Main-loop decode for a batch of transmission blobs.

        Returns
        -------
        recon_uint8_batch : (B, 3, H, W) uint8 on `self.device`
        transmit_bytes    : list[int] of length B = [len(b) for b in blobs]
        storage_bytes     : list[int] of length B or None (None for vanilla
                            codecs; populated for SEA OTTER rows).
        """
        raise NotImplementedError

    # ---- throughput path ------------------------------------------------

    @torch.no_grad()
    def encode_only_cpu(self, pil_img: PIL.Image.Image) -> bytes:
        """Per-image sensor-side encode (default: same path as collate_encode)."""
        return self.collate_encode(pil_img)

    @torch.no_grad()
    def decode_only_consumer(self, blob_bytes: bytes) -> torch.Tensor:
        """Per-image consumer-side decode → uint8 (3, H, W) on `self.device`.

        For SEAOTTER-family pipelines (sandwich-bearing), this method runs
        the full first-receive transcode path. Use the
        ``transcode_to_storage_blob`` / ``decode_steady_state_consumer``
        split below to measure deployed steady-state consumer cost.
        """
        raise NotImplementedError

    @torch.no_grad()
    def transcode_to_storage_blob(self, sensor_blob: bytes) -> bytes:
        """Cloud-side one-time transcode: sensor uplink blob → on-disk
        storage blob. Default is identity for pipelines without a
        transcode step (the sensor uplink artifact == the storage artifact).
        SEAOTTER-family pipelines override this to run the FRAPPE/WaLLoC
        neural decode + sandwich + JPEG encode chain that produces the
        on-disk JPEG file.
        """
        return sensor_blob

    @torch.no_grad()
    def decode_steady_state_consumer(self, storage_blob: bytes) -> torch.Tensor:
        """Per-image deployed steady-state consumer-side decode.

        Takes the on-disk storage blob (output of
        ``transcode_to_storage_blob``) and returns the consumer-side uint8
        (3, H, W) reconstruction. For pipelines without a transcode this
        is identical to ``decode_only_consumer``. SEAOTTER-family pipelines
        override this to be JPEG decode + F^{-1} only (no transcode work).
        """
        return self.decode_only_consumer(storage_blob)

    # ---- introspection --------------------------------------------------

    def config_block(self) -> dict:
        return {}
