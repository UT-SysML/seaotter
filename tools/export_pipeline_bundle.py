"""Export a full task-specific SEAOTTER pipeline as a loadable bundle.

Serializes a phase-4 production checkpoint (fine-tuned FRAPPE decoder G_S +
learned color pair F/F^{-1} + JPEG qtable) into the `seaotter-pipeline-v1`
bundle layout consumed by `seaotter.load_pipeline_from_local` /
`load_pipeline_from_hub`:

    <out_dir>/
    ├── config.json                          # schema seaotter-pipeline-v1
    └── <task>_pytorch_model.safetensors     # decoder.* / fwd.* / inv.* / proxy.* / qtable

The frozen FRAPPE encoder G_A is *not* serialized — it is shared and
un-fine-tuned, so the bundle only references it (frappe_encoder_repo /
frappe_encoder_subdir + frappe_n_ch) and the loader pulls it from the FRAPPE
Hub release. Weights are stored fp32 for bit-faithful reproduction.

Usage (headline ImageNet-classification bundle, the default):

    python -m tools.export_pipeline_bundle            # writes ~/hf/seaotter/seaotter_cls/

Run from the seaotter repo root (so the relative checkpoint path resolves).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from seaotter.color_transform import ForwardTransform, InverseTransform
from seaotter.proxy import JPEGProxy

# FRAPPE encoder reference (resolved by compressors.frappe.model.load_from_hub).
FRAPPE_ENCODER_REPO = "danjacobellis/FRAPPE"
FRAPPE_ENCODER_SUBDIR = "FRAPPE"

# Default = the headline cls operating point pinned from the eval-JSON
# provenance (config.checkpoint of eval_seaft_cls_n12.json).
DEFAULTS = {
    "task": "cls",
    "checkpoint": "pre_trained_convnext/experiments/iter5_imagenet_gt_squash/production/checkpoint_iter5_cls_n12.pth",
    "frappe_n_ch": 12,
    "eval_json": "/home/dgj335/UT-SysML/seaotter/results/cls/eval/eval_seaft_cls_n12.json",
    "out_dir": "/home/dgj335/hf/seaotter/seaotter_cls",
}


def _strip_prefix_free(sd: dict) -> dict:
    """Detach + contiguous-clone every tensor so safetensors can serialize it."""
    return {k: v.detach().to(torch.float32).contiguous().clone() for k, v in sd.items()}


def export_bundle(task: str, checkpoint: str, frappe_n_ch: int,
                  eval_json: str, out_dir: str) -> Path:
    from safetensors.torch import save_file

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    arch = str(ckpt["phase2_arch"])
    cfg_src = ckpt.get("config", {}) or {}

    # --- assemble the safetensors payload --------------------------------
    weights: dict[str, torch.Tensor] = {}
    for k, v in ckpt["decoder_state_dict"].items():
        weights[f"decoder.{k}"] = v.detach().to(torch.float32).contiguous().clone()
    for k, v in ckpt["fwd_state_dict"].items():
        weights[f"fwd.{k}"] = v.detach().to(torch.float32).contiguous().clone()
    for k, v in ckpt["inv_state_dict"].items():
        weights[f"inv.{k}"] = v.detach().to(torch.float32).contiguous().clone()
    for k, v in ckpt["proxy_state_dict"].items():
        weights[f"proxy.{k}"] = v.detach().to(torch.float32).contiguous().clone()

    # Derive the deployed integer qtable exactly as the codec does.
    proxy = JPEGProxy()
    proxy.load_state_dict(ckpt["proxy_state_dict"])
    proxy.eval()
    qtable = torch.clamp(proxy.qtable().round(), 1, 255).to(torch.int32).contiguous().clone()
    weights["qtable"] = qtable

    # Sanity: fwd/inv state dicts must match a fresh arch module.
    ForwardTransform(arch=arch).load_state_dict(ckpt["fwd_state_dict"])
    InverseTransform(arch=arch).load_state_dict(ckpt["inv_state_dict"])

    # --- reference metrics (authoritative, from the frozen eval JSON) ----
    ev = json.loads(Path(eval_json).read_text())
    m = ev.get("metrics") or {}
    reference_metrics = {
        "top1": m.get("top1"),
        "top5": m.get("top5"),
        "miou": m.get("miou"),
        "transmit_bpp_mean": ev.get("transmit_bpp_mean"),
        "storage_bpp_mean": ev.get("storage_bpp_mean"),
        "psnr_db": m.get("psnr_db"),
        "source_eval_json": Path(eval_json).name,
        "harness_version": ev.get("harness_version"),
    }

    weights_file = f"{Path(out_dir).name}_pytorch_model.safetensors"
    config = {
        "schema": "seaotter-pipeline-v1",
        "task": task,
        "frappe_n_ch": frappe_n_ch,
        "color_xform_arch": arch,
        "lambda": cfg_src.get("lam"),
        "rate_alpha": float(ckpt.get("rate_alpha")) if ckpt.get("rate_alpha") is not None else None,
        "subsampling": 0,
        "frappe_encoder_repo": FRAPPE_ENCODER_REPO,
        "frappe_encoder_subdir": FRAPPE_ENCODER_SUBDIR,
        "weights_file": weights_file,
        "source_checkpoint": checkpoint,
        "phase2_init": cfg_src.get("phase2_init"),
        "phase2_k": cfg_src.get("phase2_k"),
        "seed": cfg_src.get("seed"),
        "reference_metrics": reference_metrics,
        "decoder_num_params": int(sum(v.numel() for v in ckpt["decoder_state_dict"].values())),
    }

    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(out / weights_file), metadata={"schema": "seaotter-pipeline-v1"})
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    size_mb = (out / weights_file).stat().st_size / 1e6
    print(f"WROTE {out/weights_file}  ({size_mb:.1f} MB)")
    print(f"WROTE {out/'config.json'}")
    print(f"  task={task} n_ch={frappe_n_ch} arch={arch} "
          f"top1={reference_metrics['top1']} qtable=[{int(qtable.min())},{int(qtable.max())}]")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default=DEFAULTS["task"])
    ap.add_argument("--checkpoint", default=DEFAULTS["checkpoint"])
    ap.add_argument("--frappe_n_ch", type=int, default=DEFAULTS["frappe_n_ch"])
    ap.add_argument("--eval_json", default=DEFAULTS["eval_json"])
    ap.add_argument("--out_dir", default=DEFAULTS["out_dir"])
    args = ap.parse_args()
    export_bundle(args.task, args.checkpoint, args.frappe_n_ch,
                  args.eval_json, args.out_dir)


if __name__ == "__main__":
    main()
