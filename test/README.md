# Open Canon AI Studio

**Open Canon** is a comprehensive, open-source AI video production suite designed to democratize high-fidelity video generation. It removes expensive hardware and subscription barriers while preserving creator ownership through a local-first philosophy.

Whether you are an independent creator, YouTuber, studio, or enterprise, Open Canon gives you the tools to generate, edit, and publish state-of-the-art AI videos locally—or effortlessly offload to community compute if your hardware falls short.

---

## 🌟 Key Features

* **✨ State-of-the-Art (SOTA) Models by Default:**
  * **Video**: Defaults to `veo-2-4k` (World-Class 4K Video Generation).
  * **Image**: Defaults to `flux-dev` (Top-tier High-Fidelity Image Generation).
  * **Audio**: High-quality TTS using `whisper-large-v3-turbo`.
  * Also supports `wan-t2v-1.3b`, `wan-t2v-14b`, `hunyuan-video`, and `ltx-video`.
* **💻 Hardware Agnostic Telemetry:** 
  Actively profiles your system's VRAM. If your specs are too low for a requested model, Open Canon automatically falls back to CPU Sequential Layer Offloading and Tiled VAEs to ensure the generation *always completes*, no matter how long it takes.
* **☁️ Community Compute Offloading:**
  Provides precise ETAs comparing local rendering time vs. offloading to the Community Compute Cloud. Easily spin up the `opencanon compute worker` to accelerate your workflow.
* **🗣️ Claude-Style Interactive Shell:**
  Enter the `opencanon interactive` shell for a conversational AI workflow that supports `@image` and `@video` context attachments for seamless prompting.
* **📁 Automated Output Organization:**
  Automatically categorizes generated content into `videos/`, `images/`, `audio/`, and `subtitles/` with comprehensive JSON metadata sidecars for tracking prompt, model, duration, and hardware info.
* **🔧 Advanced Inference Control:**
  Fine-tune Classifier-Free Guidance (CFG), negative prompts, precision (`bfloat16`, `fp8`, `fp16`), and hardware-accelerated 2x frame upscaling.

---

## 🚀 Getting Started

### Installation
Ensure you have Python 3.10+ installed, along with PyTorch for CUDA.

```bash
# Clone the repository
git clone https://github.com/opencanon/opencanon-ai-studio.git
cd opencanon-ai-studio

# Install the CLI package
pip install -e ./cli
```

---

## 💻 Usage & Commands

Open Canon operates via the `opencanon` CLI.

### 1. Interactive Creator Shell
For the best experience, drop into the Claude-style interactive shell:
```bash
opencanon interactive
```
*Tip: Use `@path/to/image.png` inside your prompt to automatically attach an image as context for image-to-video generation!*

### 2. Video Generation
Generate the highest quality videos possible. Open Canon defaults to `veo-2-4k`.
```bash
opencanon generate video --prompt "A cinematic wide shot of a futuristic cyberpunk city, neon lights, highly detailed, 8k resolution"
```
**Advanced Arguments:**
* `--duration`: Video duration (e.g., `5s`, `10s`, `1m`).
* `--quality`: Presets like `fast`, `balanced`, `ultra`. Defaults to `ultra`.
* `--upscale`: Apply 2x super-resolution frame upscaling.
* `--guidance-scale`: Classifier-Free Guidance (CFG) scale (1.0 to 15.0).
* `--model`: Specify an alternative model (e.g., `wan-t2v-14b`, `ltx-video`).
* `--dry-run`: Test generation without downloading massive 100GB+ weights.

### 3. Image Generation
Generate incredibly detailed images. Defaults to `flux-dev` with 25 inference steps.
```bash
opencanon generate image --prompt "A stunning 8k photo of a nebula"
```

### 4. Community Compute Worker
If your local hardware is insufficient, start a worker to join the compute grid or offload your tasks.
```bash
opencanon compute worker --gpus 1 --max-vram 24
opencanon compute status
```

### 5. Model Fine-Tuning (LoRA)
Train your own custom styles or character identities.
```bash
opencanon train lora --dataset /path/to/images --output ./my-lora-model
```

---

## 📂 Project Architecture

Open Canon uses a clean, maintainable structure:

* `cli/cli/main.py`: The entry point for the `opencanon` command.
* `cli/cli/commands/`: Individual CLI command implementations (`generate.py`, `compute.py`, `train.py`, `interactive.py`).
* `cli/cli/inference/`: Handlers and runners for orchestrating the AI models (PyTorch loops, diffusers integration).
* `cli/cli/models/`: Scripts for fine-tuning (`fine_tune.py`) and LoRA injections.
* `cli/cli/utils/`: Telemetry, hardware profiling, ETA logic, and metadata sidecar tracking.
* `cli/cli/registry.py`: The master index of all supported state-of-the-art models.

---

## 🔮 Future Roadmap (Books 1-10)

This CLI is the foundation of Open Canon. According to our Open Canon "Books" specification, the future of the platform will introduce:
* **Creator-Focused Desktop Application**: A visual studio for characters, models, and timeline editing.
* **Plugin Ecosystem**: Extensible custom models, export targets, and automation workflows.
* **Model Hub**: A community-driven library for sharing models, character packs, and voice profiles.
* **Enterprise AI**: Private model libraries, audit logs, and shared organizational assets.

---

**Built with ❤️ for the Creator Community.**
