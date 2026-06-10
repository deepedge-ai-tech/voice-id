#!/usr/bin/env python3
"""Export the vendored vblinkf PyTorch model to ONNX format.

Run once from the Voice-ID directory:

    uv run python scripts/export_onnx_model.py

Produces: src/wespeaker_deep_edge/_models/vblinkf/model.onnx
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

# Add vendored wespeaker to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "wespeaker_deep_edge" / "_wespeaker"))
from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.checkpoint import load_checkpoint


MODEL_DIR = Path(__file__).resolve().parent.parent / "src" / "wespeaker_deep_edge" / "_models" / "vblinkf"


class WrapperModel(nn.Module):
    """Wraps the speaker model so ONNX export gets a clean forward signature."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        outputs = self.model(feats)
        embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
        return embeds


def main() -> None:
    config_path = MODEL_DIR / "config.yaml"
    ckpt_path = MODEL_DIR / "avg_model.pt"
    out_path = MODEL_DIR / "model.onnx"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_cls_name = config["model"]
    model_args = config["model_args"]
    print(f"Building {model_cls_name}({model_args})...")
    model = get_speaker_model(model_cls_name)(**model_args)
    load_checkpoint(model, str(ckpt_path))
    model.eval()

    wrapped = WrapperModel(model).eval()

    # Dynamic time dimension: input [1, T, 80], output [1, 256]
    num_frms = config["dataset_args"].get("num_frms", 200)
    dummy = torch.randn(1, num_frms, model_args.get("feat_dim", 80))

    torch.onnx.export(
        wrapped,
        dummy,
        str(out_path),
        do_constant_folding=True,
        opset_version=17,
        input_names=["feats"],
        output_names=["embs"],
        dynamic_axes={
            "feats": {0: "B", 1: "T"},
            "embs": {0: "B"},
        },
    )
    print(f"ONNX model saved to {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
