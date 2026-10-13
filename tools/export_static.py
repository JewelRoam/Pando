"""Export Laya to a STATIC-shape ONNX (one file per shape bucket).

Round-32 design: the dynamic graph cannot be inference-time padded (internal
Reshape folding), so each bucket gets its own static export. The OV shim
selects the smallest bucket >= input and pads (marker_mask makes marker
padding semantically safe).

Usage: python tools/export_static.py --seq 256 --markers 8
"""

from __future__ import annotations

import argparse
import os

import torch

from laya.agent import Agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="convaiinnovations/laya")
    parser.add_argument("--output", default="artifacts/laya-en-static.onnx")
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--markers", type=int, default=8)
    args = parser.parse_args()

    print(f"loading {args.model} ...", flush=True)
    agent = Agent(args.model, compile=False, device="cpu")

    batch, seq_len, num_markers = 1, args.seq, args.markers
    dummy = (
        torch.ones((batch, seq_len), dtype=torch.long),
        torch.ones((batch, seq_len), dtype=torch.long),
        torch.zeros((batch, num_markers), dtype=torch.long),
        torch.zeros((batch, num_markers), dtype=torch.bool),
        torch.zeros(batch, dtype=torch.long),
    )

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"exporting static shape batch={batch} seq={seq_len} "
          f"markers={num_markers} ...", flush=True)
    torch.onnx.export(
        agent.model,
        dummy,
        args.output,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask", "marker_pos",
                     "marker_mask", "qtype"],
        output_names=["logits", "act_logits"],
        # NO dynamic_axes - the whole point of this export.
    )
    print("saved", args.output, flush=True)


if __name__ == "__main__":
    main()
