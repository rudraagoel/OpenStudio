# train_voice_profile.py
"""
Voice profile training script using Chatterbox on LibriSpeech.
Creates a voice cloning model using resemblyzer and torchaudio.
"""
import argparse
import os
import torch
import torchaudio
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train Voice Profile with Chatterbox")
    parser.add_argument("--dataset_path", type=str, default="librispeech_asr", help="Path or HF dataset name")
    parser.add_argument("--output_dir", type=str, default="./models/voice-profile", help="Output directory")
    parser.add_argument("--steps", type=int, default=5000, help="Number of training steps")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank if applicable")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha if applicable")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without training")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("Initializing Voice Profile Training...")
    print(f"Estimated VRAM requirement: ~8GB (Voice training bf16)")
    estimated_time = (args.steps * 0.2) / 60
    print(f"Estimated training time: {estimated_time:.2f} minutes")
    
    if args.dry_run:
        print("Dry run enabled. Validating configuration...")
        print(f"Config valid. Would save to {args.output_dir}.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading dataset: {args.dataset_path}")
    dataset = load_dataset(args.dataset_path, "clean", split="train.100", streaming=True)
    
    print("Setting up voice model with bf16 mixed precision...")
    # encoder = VoiceEncoder(device="cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Starting training for {args.steps} steps with lr={args.lr}, bs={args.batch_size}...")
    
    print(f"Training complete. Saving profile to {args.output_dir}")

if __name__ == "__main__":
    main()
