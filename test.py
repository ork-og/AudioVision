from diffusers import StableDiffusionXLPipeline
import torch

device = "mps"  # для Apple Silicon
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to(device)

prompt = "lofi city at night, neon reflections, cinematic, cozy atmosphere"
image = pipe(prompt=prompt, num_inference_steps=30).images[0]
image.save("output.png")
