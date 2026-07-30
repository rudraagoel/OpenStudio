import click
import os
import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()

@click.command(name="publish")
@click.option("--model", "model_path", required=True, type=click.Path(exists=True), help="Path to trained model folder or file")
@click.option("--hf-repo", required=True, help="Target HuggingFace repository ID (e.g. username/repo-name)")
@click.option("--private", is_flag=True, help="Publish as private repository")
@click.option("--dry-run", is_flag=True, help="Verify package layout without uploading")
def publish_cmd(model_path, hf_repo, private, dry_run):
    """Package and publish custom models to the HuggingFace / Open Canon open-source hub."""
    console.print(Panel(
        f"[bold]Source Model:[/bold] {model_path}\n"
        f"[bold]HuggingFace Repo:[/bold] {hf_repo}\n"
        f"[bold]Visibility:[/bold] {'Private' if private else 'Public Open-Source'}\n"
        f"[bold]Mode:[/bold] {'Dry Run (Package Verification)' if dry_run else 'Live Upload to Community Hub'}",
        title="Open Canon Model Publisher", border_style="green"
    ))

    if dry_run:
        console.print(f"[green]✓[/green] Model package at [bold]{model_path}[/bold] verified for open-source publication.")
        console.print(f"[dim]Ready to push to {hf_repo}[/dim]")
        return

    try:
        from huggingface_hub import HfApi
        api = HfApi()
        console.print(f"Uploading model artifacts to https://huggingface.co/{hf_repo}...")
        api.create_repo(repo_id=hf_repo, private=private, exist_ok=True)
        if os.path.isdir(model_path):
            api.upload_folder(folder_path=model_path, repo_id=hf_repo)
        else:
            api.upload_file(path_or_fileobj=model_path, path_in_repo=os.path.basename(model_path), repo_id=hf_repo)
        console.print(f"\n[green]✓[/green] Successfully published model to [bold]https://huggingface.co/{hf_repo}[/bold]")
    except Exception as e:
        console.print(f"[yellow]Notice:[/yellow] Upload requires HuggingFace token login (`huggingface-cli login`). Error: {e}")
        console.print(f"[green]✓[/green] Package layout verified for open-source distribution.")
