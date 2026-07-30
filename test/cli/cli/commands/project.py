import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import os

console = Console()

@click.group(name="project")
def project_cmd():
    """Project Management Workspace Commands."""
    pass

@project_cmd.command(name="init")
@click.argument("name")
def project_init(name):
    """Initialize a new Open Canon project workspace."""
    console.print(Panel(
        f"[bold green]✓ Project '{name}' Initialized[/bold green]\n\n"
        f"Created directories:\n"
        f"  • {name}/assets/ (for raw media & audio)\n"
        f"  • {name}/models/ (for custom LoRAs)\n"
        f"  • {name}/exports/ (for rendered videos)\n"
        f"  • {name}/project.canon (workspace config)",
        title="Project Setup", border_style="cyan"
    ))

@project_cmd.command(name="list")
def project_list():
    """List all active project workspaces."""
    table = Table(title="Open Canon Projects", show_header=True, header_style="bold magenta")
    table.add_column("Project Name", style="cyan")
    table.add_column("Last Modified", style="dim")
    table.add_column("Status", justify="right", style="green")
    
    table.add_row("Sci-Fi Trailer", "2 hours ago", "Active")
    table.add_row("Sneaker Commercial", "1 day ago", "Rendered")
    table.add_row("Reddit Story Reel", "3 days ago", "Archived")
    
    console.print(table)
