#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch
from torch import nn


IMAGE_SIZE = 299
NUM_CLASSES = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-check a saved ad-safety model contract")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.model_path, map_location=device, weights_only=False)
    if not isinstance(model, nn.Module):
        raise TypeError(f"{args.model_path} does not contain a torch.nn.Module")

    model = model.to(device).eval()
    sample = torch.rand(args.batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    with torch.inference_mode():
        logits = model(sample)

    expected_shape = (args.batch_size, NUM_CLASSES)
    if tuple(logits.shape) != expected_shape:
        raise ValueError(f"Expected logits shape {expected_shape}, got {tuple(logits.shape)}")
    if not torch.is_floating_point(logits):
        raise TypeError(f"Expected floating-point logits, got {logits.dtype}")

    print(f"ok model={args.model_path} input={tuple(sample.shape)} output={tuple(logits.shape)}")


if __name__ == "__main__":
    main()
