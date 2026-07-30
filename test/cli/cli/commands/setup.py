import click
import platform
import shutil
from ..utils.gpu import detect_gpu
from ..utils.display import print_system_info
from ..utils.config import get_config_dir, save_config

@click.command()
@click.option('--check', is_flag=True, help="Only display info without saving")
def setup_cmd(check):
    """Check GPU, Python version, CUDA, and torch installation."""
    os_name = platform.system()
    py_version = platform.python_version()
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        ram_gb = "Unknown"
    
    gpu_info = detect_gpu()
    
    try:
        import torch
        torch_installed = True
        cuda_available = torch.cuda.is_available()
    except ImportError:
        torch_installed = False
        cuda_available = False
        
    ffmpeg_available = shutil.which("ffmpeg") is not None
    
    system_info = {
        "os": os_name,
        "python": py_version,
        "ram_gb": ram_gb,
        "torch_installed": torch_installed,
        "cuda_available": cuda_available,
        "ffmpeg_available": ffmpeg_available
    }
    
    print_system_info(gpu_info, system_info)
    
    if not check:
        config_dir = get_config_dir()
        config_dir.mkdir(parents=True, exist_ok=True)
        config_data = {
            "system": system_info,
            "gpu": gpu_info.__dict__ if gpu_info else None
        }
        save_config(config_data)
        click.echo(f"Saved config to {config_dir / 'config.json'}")
