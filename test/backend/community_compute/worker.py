import time
import json
from rich.console import Console
from rich.panel import Panel
from ..local_compute_mgmt import memory

console = Console()

class ComputeWorkerDaemon:
    def __init__(self, node_name: str = "local_gpu_worker", server_url: str = "http://127.0.0.1:8000"):
        self.node_name = node_name
        self.server_url = server_url
        self.is_running = False

    def get_node_telemetry(self) -> dict:
        import psutil
        import torch
        return {
            "node_name": self.node_name,
            "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
            "ram_free_gb": round(psutil.virtual_memory().available / (1024 ** 3), 2),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2) if torch.cuda.is_available() else 0.0,
        }

    def run_worker_loop(self, max_iterations: int = 5):
        self.is_running = True
        console.print(f"[bold cyan]Worker Daemon '{self.node_name}' registered.[/bold cyan]")
        console.print(f"Connecting to compute queue at {self.server_url}...")
        
        for i in range(max_iterations):
            console.print(f"  • Polling queue for render jobs... Iteration {i+1}/{max_iterations}")
            time.sleep(0.05)
            
        self.is_running = False
        console.print("[green]✓[/green] Compute worker completed processing queue.")
