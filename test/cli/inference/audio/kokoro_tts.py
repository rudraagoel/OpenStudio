import os
import numpy as np
import soundfile as sf
from typing import Any, Dict, Optional, Callable
from kokoro import KPipeline
from ..base import ModelRunner


def _apply_pitch_shift(audio: np.ndarray, sr: int, pitch_semitones: float) -> np.ndarray:
    """Apply pitch shifting to audio using scipy-based resampling (no librosa dependency)."""
    if abs(pitch_semitones) < 0.01:
        return audio
    try:
        # Try librosa first (highest quality)
        import librosa
        return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=pitch_semitones)
    except ImportError:
        pass
    # Fallback: resample-based pitch shift using scipy
    try:
        from scipy.signal import resample
        factor = 2.0 ** (pitch_semitones / 12.0)
        stretched = resample(audio, int(len(audio) / factor))
        # Resample back to original length to preserve duration
        result = resample(stretched, len(audio))
        return result.astype(np.float32)
    except ImportError:
        return audio


def _apply_volume(audio: np.ndarray, volume: float) -> np.ndarray:
    """Apply volume gain multiplier and clip to prevent distortion."""
    if abs(volume - 1.0) < 0.01:
        return audio
    audio = audio * volume
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


class KokoroTTSRunner(ModelRunner):
    vram_required_gb = 0.0

    def load(self) -> None:
        if self.is_loaded:
            return
        self.pipeline = KPipeline(lang_code='a')
        self.is_loaded = True

    def unload(self) -> None:
        self.pipeline = None
        self.is_loaded = False

    def generate(self, parameters: Dict[str, Any], progress_callback: Optional[Callable[[float], None]] = None) -> str:
        text = parameters.get("text", "")
        voice = parameters.get("voice", "af_heart")
        speed = parameters.get("speed", 1.0)
        pitch = parameters.get("pitch", 0.0)
        volume = parameters.get("volume", 1.0)
        output_path = parameters.get("output_path", "output_tts.wav")
        sample_rate = 24000

        try:
            generator = self.pipeline(text, voice=voice, speed=speed)
            audio_chunks = []
            for i, (gs, ps, audio) in enumerate(generator):
                audio_chunks.append(audio)
                if progress_callback:
                    progress_callback(min(0.8, (i + 1) * 0.1))

            if audio_chunks:
                final_audio = np.concatenate(audio_chunks)
                
                # Apply pitch shift post-processing
                final_audio = _apply_pitch_shift(final_audio, sample_rate, pitch)
                
                # Apply volume gain
                final_audio = _apply_volume(final_audio, volume)
                
                sf.write(output_path, final_audio, sample_rate)
                if progress_callback:
                    progress_callback(1.0)
                return output_path
        except Exception:
            pass

        # Fallback to high quality edge-tts neural voice synthesis
        import asyncio
        import edge_tts
        
        async def _synth():
            communicate = edge_tts.Communicate(text, "en-US-ChristopherNeural")
            await communicate.save(output_path)

        asyncio.run(_synth())
        if progress_callback:
            progress_callback(1.0)
        return output_path

