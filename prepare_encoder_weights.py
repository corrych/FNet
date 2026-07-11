"""Run once on a networked machine to save ImageNet-pretrained encoder weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from segmentation_models_pytorch.encoders import get_encoder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and save encoder weights for offline FNet training."
    )
    parser.add_argument("--encoder-name", default="tu-caformer_s36")
    parser.add_argument("--encoder-depth", type=int, default=5)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--weights", default="imagenet")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("weights/tu-caformer_s36_imagenet_encoder.pth"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    encoder = get_encoder(
        args.encoder_name,
        in_channels=args.in_channels,
        depth=args.encoder_depth,
        weights=args.weights,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder_name": args.encoder_name,
            "encoder_depth": args.encoder_depth,
            "in_channels": args.in_channels,
            "weights": args.weights,
            "encoder_state_dict": encoder.state_dict(),
        },
        args.output,
    )
    print(f"Saved encoder weights to {args.output}")


if __name__ == "__main__":
    main()
