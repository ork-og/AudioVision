import torch
from diffusers import StableDiffusionXLPipeline
import time

class ImageAI:
    pipe = None

    def __init__(self):
        pass

    def load(self, device):
        try:
            # dtype/variant подбираем под устройство
            use_fp16 = device in ("mps", "cuda")
            dtype = torch.float16 if use_fp16 else torch.float32

            pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=dtype,
                variant="fp16" if use_fp16 else None,
            ).to(device)

            self.pipe = pipe

            return pipe

        except Exception as e:
            print(e)
            return False

    def run(self, steps, prompt, w, h):
        try:
            torch.manual_seed(int(time.time()))
            img = self.pipe(
                prompt=prompt,
                num_inference_steps=int(steps),
                width=int(w),
                height=int(h),
            ).images[0]

            out_name = f"output_{w}x{h}.png"
            self.img = img
            img.save(out_name)
            result = (out_name, None)
        except Exception as e:
            result = (None, str(e))

        return result

        
