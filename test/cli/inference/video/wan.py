import os
from typing import Any, Dict, Optional, Callable
import torch
from diffusers import WanPipeline
try:
    from diffusers import WanImageToVideoPipeline
except ImportError:
    WanImageToVideoPipeline = None

from diffusers.utils import export_to_video
from PIL import Image

from ..base import ModelRunner

class WanT2VRunner(ModelRunner):
    vram_required_gb = 4.0

    def load(self) -> None:
        if self.is_loaded:
            return
        model_path = self._get_model_path()
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.pipeline = WanPipeline.from_pretrained(model_path, torch_dtype=dtype)
        if self.device == "cuda":
            if hasattr(self.pipeline, "enable_vae_tiling"):
                self.pipeline.enable_vae_tiling()
            if hasattr(self.pipeline, "enable_vae_slicing"):
                self.pipeline.enable_vae_slicing()
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb >= 12 and hasattr(self.pipeline, "enable_model_cpu_offload"):
                self.pipeline.enable_model_cpu_offload()
            else:
                self.pipeline.enable_sequential_cpu_offload()
        self.is_loaded = True

    def unload(self) -> None:
        self._cleanup_gpu()
        self.is_loaded = False

    def generate(self, parameters: Dict[str, Any], progress_callback: Optional[Callable[[float], None]] = None) -> str:
        prompt = parameters.get("prompt", "")
        default_neg = "low quality, blurry, distorted, artifact, glitch, static, low resolution, noise, flickering"
        negative_prompt = parameters.get("negative_prompt") or default_neg
        num_frames = parameters.get("num_frames", 81)
        height = parameters.get("height", 480)
        width = parameters.get("width", 832)
        num_inference_steps = parameters.get("num_inference_steps", 40)
        guidance_scale = parameters.get("guidance_scale", 6.0)
        seed = parameters.get("seed", 42)
        output_path = parameters.get("output_path", "output_t2v.mp4")

        generator = torch.Generator("cpu").manual_seed(seed)

        def step_callback(pipe, step, timestep, callback_kwargs):
            if progress_callback:
                progress_callback(step / num_inference_steps)
            return callback_kwargs

        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            callback_on_step_end=step_callback if progress_callback else None
        )
        export_to_video(output.frames[0], output_path, fps=16)
        if progress_callback:
            progress_callback(1.0)
        return output_path


class WanI2VRunner(ModelRunner):
    vram_required_gb = 4.0

    def load(self) -> None:
        if self.is_loaded:
            return
        if WanImageToVideoPipeline is None:
            raise ImportError("WanImageToVideoPipeline is not available.")
        model_path = self._get_model_path()
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.pipeline = WanImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        if self.device == "cuda":
            if hasattr(self.pipeline, "enable_vae_tiling"):
                self.pipeline.enable_vae_tiling()
            if hasattr(self.pipeline, "enable_vae_slicing"):
                self.pipeline.enable_vae_slicing()
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb >= 12 and hasattr(self.pipeline, "enable_model_cpu_offload"):
                self.pipeline.enable_model_cpu_offload()
            else:
                self.pipeline.enable_sequential_cpu_offload()
        self.is_loaded = True

    def unload(self) -> None:
        self._cleanup_gpu()
        self.is_loaded = False

    def generate(self, parameters: Dict[str, Any], progress_callback: Optional[Callable[[float], None]] = None) -> str:
        prompt = parameters.get("prompt", "")
        default_neg = "low quality, blurry, distorted, artifact, glitch, static, low resolution, noise, flickering"
        negative_prompt = parameters.get("negative_prompt") or default_neg
        image_path = parameters.get("image")
        if not image_path or not os.path.exists(image_path):
            raise ValueError("Valid image path is required for I2V")
        
        num_frames = parameters.get("num_frames", 81)
        height = parameters.get("height", 480)
        width = parameters.get("width", 832)
        num_inference_steps = parameters.get("num_inference_steps", 40)
        guidance_scale = parameters.get("guidance_scale", 6.0)
        seed = parameters.get("seed", 42)
        output_path = parameters.get("output_path", "output_i2v.mp4")

        init_image = Image.open(image_path).convert("RGB").resize((width, height))
        generator = torch.Generator("cpu").manual_seed(seed)

        def step_callback(pipe, step, timestep, callback_kwargs):
            if progress_callback:
                progress_callback(step / num_inference_steps)
            return callback_kwargs

        output = self.pipeline(
            prompt=prompt,
            image=init_image,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            callback_on_step_end=step_callback if progress_callback else None
        )
        export_to_video(output.frames[0], output_path, fps=16)
        if progress_callback:
            progress_callback(1.0)
        return output_path
