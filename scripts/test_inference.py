#!/usr/bin/env python
"""Quick inference test for converted DDPM (image generation)."""

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from PIL import Image

from src.pipelines import DDPMCDPipeline


def main():
    model_path = _PROJECT_ROOT / "models" / "BiliSakura" / "BiliSakura" / "ddpm-cd-pretrained-256"
    out_dir = _PROJECT_ROOT / "experiments" / "inference_test"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading pipeline from {model_path}...")
    pipe = DDPMCDPipeline.from_pretrained(str(model_path))

    batch_size = 2
    image_size = 256
    num_steps = 2  # Quick test (full=2000)
    print(f"Generating {batch_size} images ({image_size}x{image_size}), {num_steps} steps...")
    images = pipe.generate(batch_size=batch_size, image_size=image_size, num_inference_steps=num_steps)

    # images: (B, C, H, W) in [-1, 1]
    images = (images + 1) / 2
    images = images.clamp(0, 1).cpu()
    images = (images.permute(0, 2, 3, 1) * 255).byte().numpy()

    for i, arr in enumerate(images):
        path = out_dir / f"sample_{i}.png"
        Image.fromarray(arr).save(path)
        print(f"  Saved {path}")

    print("Done.")


if __name__ == "__main__":
    main()
