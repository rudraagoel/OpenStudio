import os
import subprocess
from typing import Any, Dict, Optional, Callable
from ..base import ModelRunner

class Wav2LipRunner(ModelRunner):
    vram_required_gb = 2.0

    def load(self) -> None:
        if self.is_loaded:
            return
        model_path = self._get_model_path()
        checkpoint = os.path.join(model_path, "wav2lip_gan.pth")
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Wav2Lip checkpoint not found at {checkpoint}")
        self.is_loaded = True

    def unload(self) -> None:
        self.is_loaded = False

    def generate(self, parameters: Dict[str, Any], progress_callback: Optional[Callable[[float], None]] = None) -> str:
        face_video = parameters.get("face_video")
        audio = parameters.get("audio")
        output_path = parameters.get("output_path", "output_lipsync.mp4")

        if not face_video or not os.path.exists(face_video):
            raise ValueError("Valid face_video path is required")
        if not audio or not os.path.exists(audio):
            raise ValueError("Valid audio path is required")

        if progress_callback:
            progress_callback(0.1)

        model_path = self._get_model_path()
        checkpoint = os.path.join(model_path, "wav2lip_gan.pth")

        cmd = [
            "python", os.path.join(model_path, "inference.py"),
            "--checkpoint_path", checkpoint,
            "--face", face_video,
            "--audio", audio,
            "--outfile", output_path
        ]

        if progress_callback:
            progress_callback(0.3)

        subprocess.run(cmd, check=True)

        if progress_callback:
            progress_callback(1.0)

        return output_path
