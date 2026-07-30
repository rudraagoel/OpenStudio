import requests
import json
from rich.console import Console

console = Console()

class LLMDirectorRunner:
    def __init__(self, model_id="llama3.2"):
        # Default to a standard ollama model name, e.g. llama3.2, llama3.1, or llama3
        self.model_id = model_id
        
    def load(self):
        # Ollama manages its own VRAM, no loading required here
        pass
            
    def unload(self):
        # Handled by Ollama automatically after timeout
        pass

    def enhance_prompt(self, user_prompt: str, mode: str = "video", image_context: str = None) -> str:
        """Uses local Ollama API to enhance the prompt for VFX, motion, and 3D graphics, supporting Image-to-Prompt."""
        import base64
        
        fallback = f"{user_prompt}, cinematic 3D graphics, dynamic motion, high-end VFX, highly detailed, Unreal Engine 5 render, raytraced"
        
        system_prompt = (
            "You are an AI Video Director. Your job is to take the user's prompt (and optional reference image) and rewrite it to be a highly detailed "
            "instruction for a text-to-video AI model. "
            "CRITICAL RULES: \n"
            "1. You MUST include descriptors for stunning VFX, dynamic motion, and 3D graphics.\n"
            "2. Keep the core subject of the user's prompt intact. If an image is provided, accurately describe its subjects and integrate them into the prompt.\n"
            "3. Describe lighting, camera movement, and aesthetic.\n"
            "4. Output ONLY the raw enhanced prompt string. Do NOT include introductory text like 'Here is the enhanced prompt'. Just the prompt itself."
        )
        
        payload = {
            "model": self.model_id if not image_context else "llama3.2-vision",
            "system": system_prompt,
            "prompt": f"User Prompt: {user_prompt}",
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 150
            }
        }
        
        if image_context:
            try:
                with open(image_context, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode('utf-8')
                    payload["images"] = [img_b64]
            except Exception as e:
                console.print(f"[yellow]Failed to read image context: {e}[/yellow]")
        
        try:
            model_name = payload["model"]
            console.print(f"[dim]Calling Ollama ({model_name}) for AI Director judgement...[/dim]")
            response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=15)
            
            if response.status_code == 200:
                enhanced = response.json().get("response", "").strip()
                if enhanced.startswith('"') and enhanced.endswith('"'):
                    enhanced = enhanced[1:-1]
                return enhanced
            else:
                console.print(f"[yellow]Ollama API error ({response.status_code}). Using fallback...[/yellow]")
                return fallback
                
        except requests.exceptions.ConnectionError:
            console.print("[dim yellow]Ollama is not running on localhost:11434. Start Ollama to use AI Director. Using fallback...[/dim yellow]")
            return fallback
        except Exception as e:
            console.print(f"[dim yellow]AI Director failed to generate, using fallback... ({e})[/dim yellow]")
            return fallback

    def generate(self, params, progress_callback=None):
        prompt = params.get("prompt", "")
        return self.enhance_prompt(prompt)
