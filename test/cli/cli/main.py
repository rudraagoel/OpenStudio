import warnings
warnings.filterwarnings("ignore", message=".*pynvml.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")

# Polyfill for diffusers <-> transformers compatibility
try:
    import transformers.utils
    if not hasattr(transformers.utils, "FLAX_WEIGHTS_NAME"):
        setattr(transformers.utils, "FLAX_WEIGHTS_NAME", "flax_model.msgpack")
except Exception:
    pass

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import click
from .commands.setup import setup_cmd
from .commands.models import models_cmd
from .commands.generate import generate_cmd
from .commands.caption import caption_cmd
from .commands.serve import serve_cmd
from .commands.compute import compute_cmd
from .commands.train import train_cmd
from .commands.publish import publish_cmd
from .commands.interactive import interactive_cmd

from .commands.edit import edit_cmd

from .commands.project import project_cmd
from .commands.plugins import plugins_cmd
from .commands.benchmark import benchmark_cmd
from .commands.character import character_cmd

@click.group()
def cli():
    """Open Canon — Local AI Generation Suite (CLI Edition)"""
    pass

cli.add_command(setup_cmd, name="setup")
cli.add_command(models_cmd, name="models")
cli.add_command(generate_cmd, name="generate")
cli.add_command(character_cmd, name="character")
cli.add_command(caption_cmd, name="caption")
cli.add_command(edit_cmd, name="edit")
cli.add_command(serve_cmd, name="serve")
cli.add_command(compute_cmd, name="compute")
cli.add_command(train_cmd, name="train")
cli.add_command(publish_cmd, name="publish")
cli.add_command(interactive_cmd, name="chat")
cli.add_command(interactive_cmd, name="interactive")
cli.add_command(project_cmd, name="project")
cli.add_command(plugins_cmd, name="plugins")
cli.add_command(benchmark_cmd, name="benchmark")

if __name__ == "__main__":
    cli()
