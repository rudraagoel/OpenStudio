import os
import re
import sys
import click
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.syntax import Syntax

console = Console()

def _extract_attached_images(text: str) -> tuple[str, list[str]]:
    """Extract image file paths attached using @path/to/image.png syntax."""
    pattern = r'@([^\s]+\.(?:png|jpg|jpeg|webp|bmp|gif))'
    attached_images = re.findall(pattern, text, re.IGNORECASE)
    clean_text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
    
    # Filter valid existing files
    valid_images = [img for img in attached_images if Path(img).exists()]
    return clean_text, valid_images

@click.command(name="chat")
@click.option("--image", "-i", "image_inputs", multiple=True, help="Attach image file(s) to session")
@click.option("--dry-run", is_flag=True, help="Run interactive shell in baseline test mode")
def interactive_cmd(image_inputs, dry_run):
    """Launch Claude Code-Style Interactive Terminal AI Studio (`opencanon chat`)"""
    console.clear()
    console.print(Panel(
        "[bold cyan]Open Canon AI Studio — Interactive Terminal Shell[/bold cyan]\n"
        "[dim]Type your prompt, attach images using [bold]@image.png[/bold], or use slash commands like [bold]/help[/bold].[/dim]\n\n"
        "[bold green]Slash Commands:[/bold green]\n"
        "  • [bold]/generate video <prompt>[/bold]  - Generate video clip\n"
        "  • [bold]/generate image <prompt>[/bold]  - Generate 1024x1024 image\n"
        "  • [bold]/generate tts <text>[/bold]     - Generate neural speech audio\n"
        "  • [bold]/models[/bold]                   - Model registry & status\n"
        "  • [bold]/compute[/bold]                  - GPU VRAM & hardware telemetry\n"
        "  • [bold]/clear[/bold]                    - Clear screen\n"
        "  • [bold]/exit[/bold]                     - Exit interactive shell",
        title="🤖 Open Canon Interactive CLI", border_style="cyan"
    ))

    session_attached_images = list(image_inputs)
    if session_attached_images:
        console.print(f"[bold yellow]Attached Session Images:[/bold yellow] {', '.join(session_attached_images)}")

    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]opencanon>[/bold cyan]").strip()
            if not user_input:
                continue

            if user_input.lower() in ["/exit", "exit", "quit", "/quit"]:
                console.print("[yellow]Exiting Open Canon Interactive Shell. Goodbye![/yellow]")
                break

            if user_input.lower() == "/clear":
                console.clear()
                continue

            if user_input.lower() == "/help":
                console.print(Panel(
                    "[bold]Interactive Help Guide[/bold]\n\n"
                    "1. [bold]Prompting with Images:[/bold]\n"
                    "   You can attach images directly in your text using the [bold]@filename.png[/bold] syntax:\n"
                    "   [dim]opencanon> Create a video of this robot exploring Mars @robot.png[/dim]\n\n"
                    "2. [bold]Slash Commands:[/bold]\n"
                    "   /models   - List models and switch active model\n"
                    "   /compute  - View GPU VRAM allocation & system RAM\n"
                    "   /serve    - Launch local REST/WebSocket server",
                    title="Help", border_style="blue"
                ))
                continue

            if user_input.lower() == "/models":
                from pathlib import Path
                import sys
                from ..commands.models import list_models
                list_models()
                continue

            if user_input.lower() == "/compute":
                from ..commands.compute import compute_status
                compute_status()
                continue

            # Extract @image attachments from text prompt
            clean_prompt, prompt_attached_images = _extract_attached_images(user_input)
            all_images = list(set(session_attached_images + prompt_attached_images))

            if prompt_attached_images:
                console.print(f"[bold yellow]Detected Image Attachments:[/bold yellow] {', '.join(prompt_attached_images)}")

            # Process prompt & trigger generation
            if user_input.startswith("/generate video") or "video" in user_input.lower():
                prompt_text = clean_prompt.replace("/generate video", "").strip() or clean_prompt
                console.print(f"[cyan]Dispatching video generation task...[/cyan]")
                from ..commands.generate import gen_video
                ctx = click.get_current_context()
                ctx.invoke(
                    gen_video,
                    prompt=prompt_text or "A futuristic cinematic scene",
                    image_path=all_images[0] if all_images else None,
                    dry_run=True if dry_run else False
                )
            elif user_input.startswith("/generate image") or "image" in user_input.lower():
                prompt_text = clean_prompt.replace("/generate image", "").strip() or clean_prompt
                console.print(f"[magenta]Dispatching image generation task...[/magenta]")
                from ..commands.generate import gen_image
                ctx = click.get_current_context()
                ctx.invoke(
                    gen_image,
                    prompt=prompt_text or "A high-tech digital artwork",
                    dry_run=True if dry_run else False
                )
            else:
                # Default conversational video prompt dispatch
                console.print(f"[cyan]Processing prompt:[/cyan] {clean_prompt}")
                if all_images:
                    console.print(f"[yellow]Using image input:[/yellow] {all_images[0]}")
                from ..commands.generate import gen_video
                ctx = click.get_current_context()
                ctx.invoke(
                    gen_video,
                    prompt=clean_prompt,
                    image_path=all_images[0] if all_images else None,
                    dry_run=True if dry_run else False
                )
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Exiting Open Canon Interactive Shell.[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
