# Open Canon CLI — Standalone AI Video & Production Engine

A standalone, local-first CLI suite for AI Video (up to 5m clip stitching), Image (FLUX.1), Text-to-Speech (Kokoro & CosyVoice), Voice Cloning (Chatterbox), Lip-Sync (Wav2Lip), AI Presenters, 3D Motion FX, Subtitle Captioning (Whisper), Reddit Reels, Custom Model Training (LoRA), Local API Server, and Community Compute Workers.

---

## Installation

```bash
cd cli

# 1. Install PyTorch with CUDA:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Install editable CLI package:
pip install -e .
```

---

## Quick Usage Commands

```bash
# 1. System Check
python -m cli.main setup --check

# 2. Model Management
python -m cli.main models --list
python -m cli.main models --installable
python -m cli.main models --switch wan-t2v-1.3b

# 3. Video Generation (Text-to-Video with TTS & Auto-Tuning)
python -m cli.main generate video --prompt "A futuristic cybernetic tiger in neon rain" --tts --quality fast

# 4. Long Video Stitching (Up to 5 minutes)
python -m cli.main generate video --prompt "Journey across galaxies" --duration 5m

# 5. Image Generation (FLUX.1 Schnell)
python -m cli.main generate image --prompt "A sleek futuristic camera lens"

# 6. Text-to-Speech & Voice Cloning
python -m cli.main generate tts --text "Hello from Open Canon"
python -m cli.main generate voice-clone --text "Sample text" --reference voice_sample.wav

# 7. AI Presenter & 3D Motion Graphics
python -m cli.main generate presenter --character host_1 --text "Welcome to today's broadcast."
python -m cli.main generate motion-fx --prompt "Geometric particle flow" --style 3d_render

# 8. Subtitle Captioning (Alex Hormozi / TikTok Styles)
python -m cli.main caption generate --video clip.mp4 --style hormozi --burn-in

# 9. Reddit Story Shorts / Reels Generator
python -m cli.main social reddit-reel --url "https://reddit.com/r/AskReddit/comments/..."

# 10. Custom Model Training & Publishing
python -m cli.main train video --dataset ./my_dataset --output ./models/custom_video_lora
python -m cli.main publish --model ./models/custom_video_lora --hf-repo open-canon/custom-video-lora

# 11. Local Server & Compute Workers
python -m cli.main serve --port 8000
python -m cli.main compute status
python -m cli.main compute worker --node-name "my_gpu_worker"
```
