import click
from rich.console import Console
from rich.panel import Panel
from ..utils.memory import get_memory_stats

console = Console()

@click.group(name="compute")
def compute_cmd():
    """Community & Local Compute Engine Commands"""
    pass

@compute_cmd.command(name="status")
def compute_status():
    """Display local RAM, VRAM, GPU hardware & active compute jobs status"""
    stats = get_memory_stats()
    
    console.print(Panel(
        f"[bold]System RAM:[/bold] {stats['ram_used_gb']} / {stats['ram_total_gb']} GB (Free: {stats['ram_free_gb']} GB)\n"
        f"[bold]GPU Accelerator:[/bold] {stats['gpu_name'] or 'CPU Only'}\n"
        f"[bold]VRAM Allocated:[/bold] {stats['vram_used_gb']} / {stats['vram_total_gb']} GB (Free: {stats['vram_free_gb']} GB)\n"
        f"[bold]CUDA Compute:[/bold] {'Available' if stats['cuda_available'] else 'Not Available'}",
        title="Open Canon Compute Hardware Telemetry", border_style="cyan"
    ))

@compute_cmd.command(name="worker")
@click.option("--node-name", default="local_gpu_worker", help="Name of this compute worker node")
@click.option("--server-url", default="http://127.0.0.1:8000", help="URL of central render queue server")
@click.option("--dry-run", is_flag=True, help="Test compute worker registration without running infinite daemon")
def compute_worker(node_name, server_url, dry_run):
    """Run local GPU worker daemon to process queued video generation tasks"""
    console.print(Panel(
        f"[bold]Worker Node:[/bold] {node_name}\n"
        f"[bold]Target Queue Server:[/bold] {server_url}\n"
        f"[bold]Mode:[/bold] {'Dry Run (Verification)' if dry_run else 'Active GPU Worker Loop'}",
        title="Compute Worker Daemon", border_style="magenta"
    ))

    try:
        from pathlib import Path
        import sys
        project_root = str(Path(__file__).resolve().parents[3])
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            
        from backend.community_compute.worker import ComputeWorkerDaemon
        worker = ComputeWorkerDaemon(node_name=node_name, server_url=server_url)
        telemetry = worker.get_node_telemetry()
        console.print(f"[bold]Detected GPU:[/bold] {telemetry['gpu_name']} | [bold]VRAM:[/bold] {telemetry['vram_total_gb']} GB")
        
        worker.run_worker_loop(max_iterations=3 if dry_run else 100)
    except Exception as e:
        console.print(f"[red]Error starting compute worker:[/red] {e}")
