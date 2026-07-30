from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from typing import Optional

console = Console()

BANNER = r"""
   ____                     ____                       
  / __ \____  ___  ____    / __ \____ _____  ____  ____ 
 / / / / __ \/ _ \/ __ \  / / / / __ `/ __ \/ __ \/ __ \
/ /_/ / /_/ /  __/ / / / / /_/ / /_/ / / / / /_/ / / / /
\____/ .___/\___/_/ /_/  \____/\__,_/_/ /_/\____/_/ /_/ 
    /_/                                                 
"""

def print_banner():
    console.print(BANNER, style="bold cyan")

def print_system_info(gpu_info: Optional[object], system_info: dict):
    print_banner()
    table = Table(title="System Information", border_style="cyan")
    table.add_column("Component", style="bold")
    table.add_column("Status / Value")

    table.add_row("OS", str(system_info.get("os", "Unknown")))
    table.add_row("Python", str(system_info.get("python", "Unknown")))
    table.add_row("RAM (GB)", str(system_info.get("ram_gb", "Unknown")))
    table.add_row("PyTorch", "Installed" if system_info.get("torch_installed") else "Not Installed")
    table.add_row("CUDA Available", "Yes" if system_info.get("cuda_available") else "No")
    table.add_row("FFmpeg", "Available" if system_info.get("ffmpeg_available") else "Not Found")

    console.print(table)

    if gpu_info:
        gpu_table = Table(title="GPU Detected", border_style="green")
        gpu_table.add_column("Property", style="bold")
        gpu_table.add_column("Value")
        gpu_table.add_row("Name", gpu_info.name)
        gpu_table.add_row("Total VRAM", f"{gpu_info.vram_total_gb} GB")
        gpu_table.add_row("Free VRAM", f"{gpu_info.vram_free_gb} GB")
        if gpu_info.driver_version:
            gpu_table.add_row("Driver", gpu_info.driver_version)
        console.print(gpu_table)
    else:
        console.print("[yellow]No GPU detected.[/yellow]")

def print_models_table(models: dict, installed_map: dict):
    table = Table(title="Model Registry", border_style="cyan")
    table.add_column("ID", style="bold yellow")
    table.add_column("Name")
    table.add_column("Category")
    table.add_column("Size (GB)")
    table.add_column("Min VRAM (GB)")
    table.add_column("Status")

    for mid, info in models.items():
        is_inst = installed_map.get(mid, False)
        status_str = "[green]Installed[/green]" if is_inst else "[dim]Not Installed[/dim]"
        table.add_row(
            mid,
            info["name"],
            info["category"],
            str(info["size_gb"]),
            str(info["vram_min_gb"]),
            status_str
        )

    console.print(table)

def create_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    )
