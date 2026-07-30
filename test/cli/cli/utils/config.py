import os
import json
import time
from pathlib import Path

def get_config_dir() -> Path:
    base = Path.home() / ".opencanon"
    base.mkdir(parents=True, exist_ok=True)
    return base

def get_models_dir() -> Path:
    base = get_config_dir() / "models"
    base.mkdir(parents=True, exist_ok=True)
    return base

def get_outputs_dir() -> Path:
    base = get_config_dir() / "outputs"
    base.mkdir(parents=True, exist_ok=True)
    return base

def get_category_outputs_dir(category: str = "videos") -> Path:
    base = get_outputs_dir() / category
    base.mkdir(parents=True, exist_ok=True)
    return base

def save_output_metadata(media_file_path: str, metadata: dict) -> str:
    """Save JSON metadata sidecar file alongside generated media output."""
    path_obj = Path(media_file_path)
    meta_path = path_obj.with_suffix(path_obj.suffix + ".json")
    
    full_meta = {
        "file_name": path_obj.name,
        "file_path": str(path_obj.resolve()),
        "file_size_bytes": path_obj.stat().st_size if path_obj.exists() else 0,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": time.time(),
        **metadata
    }
    meta_path.write_text(json.dumps(full_meta, indent=2), encoding="utf-8")
    return str(meta_path)

def load_config() -> dict:
    config_file = get_config_dir() / "config.json"
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_config(config: dict):
    config_file = get_config_dir() / "config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

def is_model_installed(model_id: str) -> bool:
    target_dir = get_models_dir() / model_id
    if not target_dir.exists():
        return False
    # Check for valid huggingface model files
    return (target_dir / "model_index.json").exists() or (target_dir / "config.json").exists()
