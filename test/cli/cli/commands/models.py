import click
import shutil
from rich.console import Console
from rich.panel import Panel
from huggingface_hub import snapshot_download
from ..registry import MODEL_REGISTRY
from ..utils.display import print_models_table
from ..utils.config import get_models_dir, is_model_installed, load_config, save_config

console = Console()

def _install_model(model_id: str):
    """Internal function to install a model."""
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_id} not found in registry.")
    
    model_info = MODEL_REGISTRY[model_id]
    models_dir = get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    target_dir = models_dir / model_id
    
    click.echo(f"Downloading {model_info['name']} ({model_info['size_gb']} GB) from {model_info['hf_repo']}...")
    snapshot_download(
        repo_id=model_info['hf_repo'],
        local_dir=str(target_dir),
        local_dir_use_symlinks=False
    )
    click.echo(f"Successfully installed {model_id} to {target_dir}")

def _switch_model(model_id: str):
    """Internal function to switch default active model."""
    if model_id not in MODEL_REGISTRY:
        console.print(f"[red]Error:[/red] Model '{model_id}' is not in registry.")
        return
    
    category = MODEL_REGISTRY[model_id].get("category", "video")
    config = load_config()
    if "active_models" not in config:
        config["active_models"] = {}
    config["active_models"][category] = model_id
    save_config(config)
    
    console.print(f"[green]✓[/green] Active default {category} model switched to [bold]{model_id}[/bold] ({MODEL_REGISTRY[model_id]['name']})")

@click.group(invoke_without_command=True)
@click.option("--list", "opt_list", is_flag=True, help="List all models from the registry")
@click.option("--installable", "opt_installable", is_flag=True, help="List models available for download")
@click.option("--switch", "opt_switch", type=str, default=None, help="Switch active default model")
@click.option("--install", "opt_install", type=str, default=None, help="Install a model by ID")
@click.pass_context
def models_cmd(ctx, opt_list, opt_installable, opt_switch, opt_install):
    """Model management commands (list, switch, installable, install, remove)"""
    if ctx.invoked_subcommand is None:
        if opt_list:
            ctx.invoke(list_models)
        elif opt_installable:
            ctx.invoke(list_installable)
        elif opt_switch:
            _switch_model(opt_switch)
        elif opt_install:
            ctx.invoke(install_model, model_id=opt_install)
        else:
            ctx.invoke(list_models)

@models_cmd.command(name="list")
def list_models():
    """List all models from the registry"""
    models = MODEL_REGISTRY
    installed = {mid: is_model_installed(mid) for mid in models}
    config = load_config()
    active = config.get("active_models", {})
    
    console.print("[bold cyan]Open Canon Model Registry & Status[/bold cyan]")
    print_models_table(models, installed)
    if active:
        console.print("\n[bold green]Active Default Models:[/bold green]")
        for cat, mid in active.items():
            console.print(f"  • {cat.capitalize()}: [bold]{mid}[/bold]")

@models_cmd.command(name="installable")
def list_installable():
    """List models available to download that are not yet installed"""
    models = {mid: info for mid, info in MODEL_REGISTRY.items() if not is_model_installed(mid)}
    if not models:
        console.print("[green]All models in registry are currently installed![/green]")
        return
    
    console.print("[bold yellow]Available Models for Download[/bold yellow]")
    installed = {mid: False for mid in models}
    print_models_table(models, installed)

@models_cmd.command(name="switch")
@click.argument("model_id")
def switch_model_cmd(model_id):
    """Switch active default model for a category"""
    _switch_model(model_id)

@models_cmd.command(name="install")
@click.argument("model_id")
def install_model(model_id):
    """Install a model from HuggingFace"""
    if model_id not in MODEL_REGISTRY:
        click.echo(f"Error: Model {model_id} not found in registry.", err=True)
        return
    try:
        _install_model(model_id)
    except Exception as e:
        click.echo(f"Error downloading model: {e}", err=True)

@models_cmd.command(name="remove")
@click.argument("model_id")
def remove_model(model_id):
    """Remove an installed model"""
    if not is_model_installed(model_id):
        click.echo(f"Model {model_id} is not installed.")
        return
        
    if click.confirm(f"Are you sure you want to remove {model_id}?"):
        target_dir = get_models_dir() / model_id
        try:
            shutil.rmtree(target_dir)
            click.echo(f"Removed {model_id}.")
        except Exception as e:
            click.echo(f"Error removing model: {e}", err=True)
