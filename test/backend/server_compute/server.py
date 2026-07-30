import os
import sys
import json
import uuid
import time
from typing import Dict, Any, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Open Canon Local AI Server",
    description="Local hosting HTTP REST & WebSocket API engine for Open Canon",
    version="0.1.0"
)

JOBS_DB: Dict[str, Dict[str, Any]] = {}

class VideoGenRequest(BaseModel):
    prompt: str
    duration: str = "5s"
    model: str = "wan-t2v-1.3b"
    quality: str = "auto"
    tts: bool = False
    dry_run: bool = True

class TTSGenRequest(BaseModel):
    text: str
    voice: str = "af_heart"
    speed: float = 1.0
    dry_run: bool = True

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Open Canon Local AI Hosting Server",
        "docs_url": "/docs"
    }

@app.get("/api/v1/health")
def health():
    import psutil
    import torch
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "ram_gb_used": round(psutil.virtual_memory().used / (1024 ** 3), 2),
        "ram_gb_total": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }

@app.get("/api/v1/models")
def get_models():
    from cli.registry import MODEL_REGISTRY
    from cli.utils.config import is_model_installed
    installed = {mid: is_model_installed(mid) for mid in MODEL_REGISTRY}
    return {
        "models": MODEL_REGISTRY,
        "installed": installed
    }

@app.post("/api/v1/generate/video")
def generate_video(req: VideoGenRequest, background_tasks: BackgroundTasks):
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    JOBS_DB[job_id] = {
        "job_id": job_id,
        "type": "video",
        "prompt": req.prompt,
        "status": "queued",
        "progress": 0,
        "output_path": None,
        "created_at": time.time()
    }
    
    def process_job():
        JOBS_DB[job_id]["status"] = "processing"
        for i in range(10):
            time.sleep(0.1)
            JOBS_DB[job_id]["progress"] = (i + 1) * 10
        JOBS_DB[job_id]["status"] = "completed"
        JOBS_DB[job_id]["output_path"] = f"outputs/video_{job_id}.mp4"

    background_tasks.add_task(process_job)
    return {"job_id": job_id, "status": "queued"}

@app.get("/api/v1/jobs/{job_id}")
def get_job_status(job_id: str):
    if job_id not in JOBS_DB:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS_DB[job_id]
