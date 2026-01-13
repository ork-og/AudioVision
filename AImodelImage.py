import inspect
import os
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline

class ImageAI:
    pipe = None

    def __init__(self):
        pass

    def load(self, device):
        try:
            # dtype/variant подбираем под устройство
            use_fp16 = device in ("mps", "cuda")
            dtype = torch.float16 if use_fp16 else torch.float32

            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=dtype,
                variant="fp16" if use_fp16 else None,
            ).to(device)

            return self.pipe

        except Exception as e:
            print(e)
            return False

    def run(self, steps, prompt, w, h, progress_callback=None):
        try:
            torch.manual_seed(int(time.time()))
            call_kwargs = dict(
                prompt=prompt,
                num_inference_steps=int(steps),
                width=int(w),
                height=int(h),
            )

            if progress_callback is not None:
                def _cb(step_index, *args):
                    try:
                        progress_callback(step_index, int(steps))
                    except Exception:
                        pass

                try:
                    sig = inspect.signature(self.pipe.__call__)
                except (TypeError, ValueError):
                    sig = None

                if sig and "callback" in sig.parameters:
                    call_kwargs["callback"] = _cb
                    call_kwargs["callback_steps"] = 1
                elif sig and "callback_on_step_end" in sig.parameters:
                    def _on_step_end(pipe, step_index, timestep, callback_kwargs):
                        _cb(step_index, timestep, None)
                        return callback_kwargs

                    call_kwargs["callback_on_step_end"] = _on_step_end
                else:
                    call_kwargs["callback"] = _cb
                    call_kwargs["callback_steps"] = 1

            img = self.pipe(**call_kwargs).images[0]

            out_dir = self._default_output_dir()
            out_name = f"output_{w}x{h}.png"
            out_path = os.path.join(out_dir, out_name)
            self.img = img
            img.save(out_path)
            result = (out_path, None)
        except Exception as e:
            result = (None, str(e))

        return result

    def _default_output_dir(self):
        home = Path.home()
        pictures = home / "Pictures"
        if pictures.exists():
            base = pictures / "AudioVision"
        else:
            base = home / "AudioVision"
        out_dir = base / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir)

        
