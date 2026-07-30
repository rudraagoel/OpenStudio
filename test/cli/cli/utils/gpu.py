from dataclasses import dataclass
from typing import Optional

@dataclass
class GPUInfo:
    name: str
    vram_total_gb: float
    vram_free_gb: float
    cuda_version: Optional[str] = None
    driver_version: Optional[str] = None

def detect_gpu() -> Optional[GPUInfo]:
    """Detect NVIDIA GPU using pynvml, falling back to torch.cuda."""
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            driver_ver = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver_ver, bytes):
                driver_ver = driver_ver.decode("utf-8")
            
            return GPUInfo(
                name=name,
                vram_total_gb=round(mem_info.total / (1024**3), 2),
                vram_free_gb=round(mem_info.free / (1024**3), 2),
                driver_version=driver_ver
            )
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory
            return GPUInfo(
                name=name,
                vram_total_gb=round(total_mem / (1024**3), 2),
                vram_free_gb=round(total_mem / (1024**3), 2),
                cuda_version=torch.version.cuda
            )
    except Exception:
        pass

    return None
