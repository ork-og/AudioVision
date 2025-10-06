#!/usr/bin/env python3
# sdxl_mps_wh_only.py
import argparse
from diffusers import StableDiffusionXLPipeline
import torch

def main():
    parser = argparse.ArgumentParser(description="SDXL on MPS with only width/height configurable")
    parser.add_argument("--width", type=int, default=1024, help="Ширина изображения")
    parser.add_argument("--height", type=int, default=1024, help="Высота изображения")
    args = parser.parse_args()

    device = "mps"  # для Apple Silicon
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to(device)

    prompt = "lofi night scene, distant figure watching city lights, glowing street lamps, calm atmosphere, cinematic mood, soft lighting, detailed reflections, cozy melancholy"
    image = pipe(
        prompt=prompt,
        num_inference_steps=30,
        width=args.width,
        height=args.height,
    ).images[0]

    out_name = f"output_{args.width}x{args.height}.png"
    image.save(out_name)
    print("Saved:", out_name)

if __name__ == "__main__":
    main()
