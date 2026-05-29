"""SEA OTTER sandwich: ForwardTransform + JPEGProxy + InverseTransform.

Warm-starts from a phase-2 K-tier checkpoint; arch is auto-detected from
the checkpoint metadata. The phase-2 K=3 R16 checkpoint at
PHASE2_K3_PATH_DEFAULT (phase2_k=2) is the canonical warm-start for the
phase-4 training family.
"""

from __future__ import annotations

import torch

from seaotter.color_transform import ForwardTransform, InverseTransform
from seaotter.proxy import JPEGProxy


def load_sandwich(
    phase2_init: str,
    phase2_k: int,
    device: torch.device | str,
) -> tuple[ForwardTransform, JPEGProxy, InverseTransform, dict]:
    """Build (fwd, proxy, inv, meta) warm-started from a phase-2 K-tier
    checkpoint. meta carries arch + lambdas + which tier was picked."""
    ckpt = torch.load(phase2_init, map_location=device, weights_only=False)
    arch = ckpt.get("config", {}).get("arch")
    if arch not in ("A", "B", "D", "E", "F", "G"):
        raise ValueError(f"phase2 checkpoint arch={arch!r} not supported")
    proxy_state_dicts = ckpt["proxy_state_dicts"]
    if not 0 <= phase2_k < len(proxy_state_dicts):
        raise ValueError(
            f"phase2_k={phase2_k} out of range [0, {len(proxy_state_dicts) - 1}]"
        )

    fwd = ForwardTransform(arch=arch).to(device)
    inv = InverseTransform(arch=arch).to(device)
    proxy = JPEGProxy(init=torch.full((3, 8, 8), 8.0)).to(device)
    fwd.load_state_dict(ckpt["fwd_state_dict"])
    inv.load_state_dict(ckpt["inv_state_dict"])
    proxy.load_state_dict(proxy_state_dicts[phase2_k])

    meta = {
        "arch": arch,
        "lambdas": ckpt.get("lambdas", []),
        "phase2_k": phase2_k,
        "phase2_init": phase2_init,
    }
    return fwd, proxy, inv, meta


def set_sandwich_freeze(
    fwd: ForwardTransform,
    proxy: JPEGProxy,
    inv: InverseTransform,
    hot: bool,
) -> None:
    """Toggle requires_grad on the sandwich. eval()/train() mode is the
    caller's responsibility — it gates JPEGProxy noise + boundary noise."""
    for m in (fwd, proxy, inv):
        for p in m.parameters():
            p.requires_grad_(bool(hot))
