import os
from typing import Any, Dict, Optional, Callable
import torch
import soundfile as sf
from ..base import ModelRunner

class ChatterboxRunner(ModelRunner):
    vram_required_gb = 4.0

    def load(self) -> None:
        if self.is_loaded:
            return
        from chatterbox.tts import ChatterboxTTS
        self.pipeline = ChatterboxTTS.from_pretrained(device=self.device)
        self.is_loaded = True

    def unload(self) -> None:
        self._cleanup_gpu()
        self.is_loaded = False

    def generate(self, parameters: Dict[str, Any], progress_callback: Optional[Callable[[float], None]] = None) -> str:
        text = parameters.get("text", "")
        reference_audio = parameters.get("reference_audio")
        output_path = parameters.get("output_path", "output_voice_clone.wav")

        if progress_callback:
            progress_callback(0.1)

        wav = self.pipeline.generate(
            text=text,
            audio_prompt_path=reference_audio
        )

        if progress_callback:
            progress_callback(0.8)

        sf.write(output_path, wav.cpu().numpy(), getattr(self.pipeline, "sr", 24000))

        if progress_callback:
            progress_callback(1.0)

        return output_path
