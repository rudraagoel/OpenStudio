import gc
import sys
import torch
from rich.console import Console

console = Console()

def clear_cuda_memory():
    """Force garbage collection and clear PyTorch CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()

def get_memory_stats() -> dict:
    """Get current RAM, VRAM, and GPU utilization stats."""
    import psutil
    stats = {
        "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "ram_used_gb": round(psutil.virtual_memory().used / (1024 ** 3), 2),
        "ram_free_gb": round(psutil.virtual_memory().available / (1024 ** 3), 2),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": None,
        "vram_total_gb": 0.0,
        "vram_used_gb": 0.0,
        "vram_free_gb": 0.0,
    }
    
    if torch.cuda.is_available():
        stats["gpu_name"] = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        stats["vram_total_gb"] = round(total / (1024 ** 3), 2)
        stats["vram_used_gb"] = round(reserved / (1024 ** 3), 2)
        stats["vram_free_gb"] = round((total - reserved) / (1024 ** 3), 2)
        
    return stats

def configure_pipeline_memory(pipeline, enable_cpu_offload: bool = True, enable_sequential_offload: bool = True, enable_attention_slicing: bool = True, enable_vae_tiling: bool = True):
    """Apply memory saving optimizations to diffusers pipeline."""
    if not torch.cuda.is_available():
        return pipeline

    try:
        if enable_sequential_offload and hasattr(pipeline, "enable_sequential_cpu_offload"):
            pipeline.enable_sequential_cpu_offload()
        elif enable_cpu_offload and hasattr(pipeline, "enable_model_cpu_offload"):
            pipeline.enable_model_cpu_offload()
            
        if enable_attention_slicing and hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing("auto")
            
        if enable_vae_tiling and hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()
    except Exception as e:
        console.print(f"[yellow]Notice during memory optimization:[/yellow] {e}")
        
    return pipeline
