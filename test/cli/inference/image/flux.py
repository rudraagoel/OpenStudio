import os
from typing import Any, Dict, Optional, Callable
import torch
from diffusers import FluxPipeline
from ..base import ModelRunner

class FluxSchnellRunner(ModelRunner):
    vram_required_gb = 8.0

    def load(self) -> None:
        if self.is_loaded:
            return
        model_path = self._get_model_path()
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.pipeline = FluxPipeline.from_pretrained(model_path, torch_dtype=dtype)
        if self.device == "cuda":
            # Enable VAE optimizations for higher resolution without OOM
            if hasattr(self.pipeline, "enable_vae_tiling"):
                self.pipeline.enable_vae_tiling()
            if hasattr(self.pipeline, "enable_vae_slicing"):
                self.pipeline.enable_vae_slicing()
            # Smart offloading: model_cpu_offload is faster but needs more VRAM
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb >= 16 and hasattr(self.pipeline, "enable_model_cpu_offload"):
                self.pipeline.enable_model_cpu_offload()
            else:
                self.pipeline.enable_sequential_cpu_offload()
        self.is_loaded = True

    def unload(self) -> None:
        self._cleanup_gpu()
        self.is_loaded = False

    def generate(self, parameters: Dict[str, Any], progress_callback: Optional[Callable[[float], None]] = None) -> str:
        prompt = parameters.get("prompt", "")
        height = parameters.get("height", 1024)
        width = parameters.get("width", 1024)
        num_inference_steps = parameters.get("num_inference_steps", 4)
        guidance_scale = parameters.get("guidance_scale", 0.0)
        seed = parameters.get("seed", 42)
        output_path = parameters.get("output_path", "output_flux.png")

        generator = torch.Generator("cpu").manual_seed(seed)

        def step_callback(pipe, step, timestep, callback_kwargs):
            if progress_callback:
                progress_callback(step / num_inference_steps)
            return callback_kwargs

        output = self.pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            callback_on_step_end=step_callback if progress_callback else None
        )

        image = output.images[0]
        image.save(output_path)

        if progress_callback:
            progress_callback(1.0)

        return output_path


class FluxDevRunner(ModelRunner):
    """FLUX.1-dev runner with higher step count and CFG for maximum quality."""
    vram_required_gb = 12.0

    def load(self) -> None:
        if self.is_loaded:
            return
        model_path = self._get_model_path()
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.pipeline = FluxPipeline.from_pretrained(model_path, torch_dtype=dtype)
        if self.device == "cuda":
            if hasattr(self.pipeline, "enable_vae_tiling"):
                self.pipeline.enable_vae_tiling()
            if hasattr(self.pipeline, "enable_vae_slicing"):
                self.pipeline.enable_vae_slicing()
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb >= 16 and hasattr(self.pipeline, "enable_model_cpu_offload"):
                self.pipeline.enable_model_cpu_offload()
            else:
                self.pipeline.enable_sequential_cpu_offload()
        self.is_loaded = True

    def unload(self) -> None:
        self._cleanup_gpu()
        self.is_loaded = False

    def generate(self, parameters: Dict[str, Any], progress_callback: Optional[Callable[[float], None]] = None) -> str:
        prompt = parameters.get("prompt", "")
        height = parameters.get("height", 1024)
        width = parameters.get("width", 1024)
        num_inference_steps = parameters.get("num_inference_steps", 25)
        guidance_scale = parameters.get("guidance_scale", 3.5)
        seed = parameters.get("seed", 42)
        output_path = parameters.get("output_path", "output_flux_dev.png")

        generator = torch.Generator("cpu").manual_seed(seed)

        def step_callback(pipe, step, timestep, callback_kwargs):
            if progress_callback:
                progress_callback(step / num_inference_steps)
            return callback_kwargs

        output = self.pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            callback_on_step_end=step_callback if progress_callback else None
        )

        image = output.images[0]
        image.save(output_path)

        if progress_callback:
            progress_callback(1.0)

        return output_path

