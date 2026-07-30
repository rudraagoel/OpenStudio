import click
import os
import sys
from pathlib import Path
from rich.console import Console

console = Console()

@click.group(name="train")
def train_cmd():
    """Custom Model Fine-Tuning Commands (Video, Image, Voice)"""
    pass

def _run_trainer(model_type: str, base_model: str, dataset: str, output: str, steps: int, rank: int, lr: float, dry_run: bool):
    project_root = str(Path(__file__).resolve().parents[3])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    from models.fine_tune import CustomModelTrainer
    trainer = CustomModelTrainer(model_type=model_type, base_model=base_model)
    trainer.train_lora(dataset_path=dataset, output_dir=output, num_steps=steps, lora_rank=rank, learning_rate=lr, dry_run=dry_run)

@train_cmd.command(name="video")
@click.option("--dataset", required=True, help="Path to video/clip dataset folder")
@click.option("--base-model", default="wan-t2v-1.3b", help="Base model ID to fine-tune")
@click.option("--output", default="./models/custom_video_lora", help="Output directory for LoRA weights")
@click.option("--steps", default=500, type=int, help="Training steps")
@click.option("--rank", default=16, type=int, help="LoRA rank dimension")
@click.option("--lr", default=1e-4, type=float, help="Learning rate")
@click.option("--dry-run", is_flag=True, help="Simulate training pipeline without GPU weight allocation")
def train_video(dataset, base_model, output, steps, rank, lr, dry_run):
    """Fine-tune a video model with custom video clips or style datasets."""
    _run_trainer("video", base_model, dataset, output, steps, rank, lr, dry_run)

@train_cmd.command(name="image")
@click.option("--dataset", required=True, help="Path to image dataset folder")
@click.option("--base-model", default="flux-schnell", help="Base image model ID")
@click.option("--output", default="./models/custom_image_lora", help="Output directory for LoRA weights")
@click.option("--steps", default=500, type=int, help="Training steps")
@click.option("--rank", default=16, type=int, help="LoRA rank dimension")
@click.option("--lr", default=1e-4, type=float, help="Learning rate")
@click.option("--dry-run", is_flag=True, help="Simulate training pipeline without GPU weight allocation")
def train_image(dataset, base_model, output, steps, rank, lr, dry_run):
    """Fine-tune an image model (FLUX.1) with custom images/characters."""
    _run_trainer("image", base_model, dataset, output, steps, rank, lr, dry_run)

@train_cmd.command(name="voice")
@click.option("--dataset", required=True, help="Path to voice audio samples folder")
@click.option("--base-model", default="chatterbox", help="Base voice model ID")
@click.option("--output", default="./models/custom_voice_profile", help="Output directory for voice profile")
@click.option("--steps", default=300, type=int, help="Training steps")
@click.option("--rank", default=16, type=int, help="LoRA rank dimension")
@click.option("--lr", default=1e-4, type=float, help="Learning rate")
@click.option("--dry-run", is_flag=True, help="Simulate training pipeline without GPU weight allocation")
def train_voice(dataset, base_model, output, steps, rank, lr, dry_run):
    """Fine-tune a voice model to create persistent character voice profiles."""
    _run_trainer("voice", base_model, dataset, output, steps, rank, lr, dry_run)
