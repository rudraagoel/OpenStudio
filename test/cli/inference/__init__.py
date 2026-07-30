# Polyfill for diffusers <-> transformers compatibility
try:
    import transformers.utils
    if not hasattr(transformers.utils, "FLAX_WEIGHTS_NAME"):
        setattr(transformers.utils, "FLAX_WEIGHTS_NAME", "flax_model.msgpack")
except Exception:
    pass

from .base import ModelRunner

def get_runner(model_id: str) -> ModelRunner:
    """Factory: lazily import and return the runner for a specific model ID."""
    if model_id == "kokoro-tts":
        from .audio.kokoro_tts import KokoroTTSRunner
        return KokoroTTSRunner(model_id=model_id)
    elif model_id == "chatterbox":
        from .audio.chatterbox_vc import ChatterboxRunner
        return ChatterboxRunner(model_id=model_id)
    elif model_id == "flux-schnell":
        from .image.flux import FluxSchnellRunner
        return FluxSchnellRunner(model_id=model_id)
    elif model_id == "wan-t2v-1.3b":
        from .video.wan import WanT2VRunner
        return WanT2VRunner(model_id=model_id)
    elif model_id == "wan-i2v-1.3b":
        from .video.wan import WanI2VRunner
        return WanI2VRunner(model_id=model_id)
    elif model_id == "wav2lip":
        from .video.lipsync import Wav2LipRunner
        return Wav2LipRunner(model_id=model_id)
    else:
        raise ValueError(f"Unknown model_id '{model_id}'.")

__all__ = ["ModelRunner", "get_runner"]
