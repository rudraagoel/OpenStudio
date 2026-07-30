import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from diffusers.optimization import get_scheduler
from diffusers import UNet3DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from torchvision import transforms
from tqdm.auto import tqdm
import math
from einops import rearrange

logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="Train Video LoRA on a small video model")
    parser.add_argument("--dataset_path", type=str, default="lambdalabs/pokemon-blip-captions", help="Path or HF dataset name (using image dataset and mocking time for toy run)")
    parser.add_argument("--output_dir", type=str, default="./models/video-lora", help="Output directory")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=8, help="LoRA alpha")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="damo-vilab/text-to-video-ms-1.7b")
    parser.add_argument("--num_frames", type=int, default=8, help="Number of frames per video")
    return parser.parse_args()

def main():
    args = parse_args()
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    set_seed(args.seed)
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info("Initializing Video LoRA Training...")
    
    # Load Models
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    
    try:
        unet = UNet3DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    except Exception:
        logger.warning("Could not load UNet3D from pretrained path, falling back to dummy UNet3D for compilation testing.")
        unet = UNet3DConditionModel(
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("CrossAttnDownBlock3D", "DownBlock3D"),
            up_block_types=("UpBlock3D", "CrossAttnUpBlock3D"),
            block_out_channels=(32, 64),
            layers_per_block=1,
            cross_attention_dim=text_encoder.config.hidden_size
        )

    # Freeze base models
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    
    # Setup LoRA
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        init_lora_weights="gaussian"
    )
    unet = get_peft_model(unet, lora_config)
    unet.train()
    
    if hasattr(unet, "enable_gradient_checkpointing"):
        try:
            unet.enable_gradient_checkpointing()
        except Exception as e:
            print(f"Warning: Could not enable gradient checkpointing: {e}")

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=1e-4)

    # Dataset: Stream WebVid for large scale DDP training
    dataset = load_dataset(args.dataset_path, split="train", streaming=True)
    # dataset = dataset.shuffle(seed=args.seed) # Optional if buffer size specified
    
    transform = transforms.Compose([
        transforms.Resize(64, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    def collate_fn(examples):
        # Create fake video by duplicating the image args.num_frames times
        pixel_values = []
        for example in examples:
            img = transform(example["image"].convert("RGB"))
            video = img.unsqueeze(1).repeat(1, args.num_frames, 1, 1) # [C, F, H, W]
            pixel_values.append(video)
        pixel_values = torch.stack(pixel_values) # [B, C, F, H, W]

        input_ids = tokenizer(
            [example["text"] for example in examples],
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids
        return {"pixel_values": pixel_values, "input_ids": input_ids}
        
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    lr_scheduler = get_scheduler(
        "constant", 
        optimizer=optimizer, 
        num_warmup_steps=0, 
        num_training_steps=args.steps * accelerator.num_processes
    )
    
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    vae.to(accelerator.device, dtype=accelerator.unwrap_model(unet).dtype)
    text_encoder.to(accelerator.device, dtype=accelerator.unwrap_model(unet).dtype)
    
    global_step = 0
    progress_bar = tqdm(total=args.steps, disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    from diffusers import DDPMScheduler
    try:
        noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    except:
        noise_scheduler = DDPMScheduler()

    while global_step < args.steps:
        for step, batch in enumerate(train_dataloader):
            if global_step >= args.steps:
                break
            with accelerator.accumulate(unet):
                videos = batch["pixel_values"].to(dtype=vae.dtype, device=accelerator.device) # [B, C, F, H, W]
                b, c, f, h, w = videos.shape
                
                videos_rearranged = rearrange(videos, "b c f h w -> (b f) c h w")
                
                # Encode frames
                latents = vae.encode(videos_rearranged).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Reshape back to video latents [B, C, F, H, W]
                latents = rearrange(latents, "(b f) c h w -> b c f h w", b=b)

                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get text embeddings
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict noise
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Compute loss
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": loss.item()})
                
            if global_step >= args.steps:
                break
        if global_step >= args.steps:
            break
            
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info(f"Training complete. Saving weights to {args.output_dir}")
        unwrapped_model = accelerator.unwrap_model(unet)
        unwrapped_model.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
