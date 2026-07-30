import abc
import os
import gc
from typing import Any, Dict, Optional, Callable

class ModelRunner(abc.ABC):
    vram_required_gb: float = 0.0
    
    def __init__(self, model_id: str, device: str = "auto"):
        self.model_id = model_id
        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device
        self.pipeline = None
        self.is_loaded = False
    
    def _get_model_path(self) -> str:
        base = os.path.join(os.path.expanduser("~"), ".opencanon", "models", self.model_id)
        if not os.path.exists(base):
            raise FileNotFoundError(f"Model '{self.model_id}' not found at {base}. Run: opencanon models install {self.model_id}")
        return base
    
    @abc.abstractmethod
    def load(self) -> None:
        pass
    
    @abc.abstractmethod
    def unload(self) -> None:
        pass
    
    @abc.abstractmethod
    def generate(self, parameters: Dict[str, Any], progress_callback: Optional[Callable[[float], None]] = None) -> str:
        pass
    
    def __enter__(self):
        self.load()
        return self
    
    def __exit__(self, *args):
        self.unload()
    
    def _cleanup_gpu(self):
        self.pipeline = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
