import os
import subprocess
import math
from typing import Any, Dict, Optional, Callable
import tempfile
import cv2

class VideoStitcher:
    def __init__(self, runner, output_dir: str):
        self.runner = runner
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def extract_last_frame(self, video_path: str, output_image_path: str) -> str:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError(f"Could not read last frame from {video_path}")
            
        cv2.imwrite(output_image_path, frame)
        cap.release()
        return output_image_path

    def concatenate_videos(self, video_paths, output_path, crossfade_duration=0.5):
        if not video_paths:
            raise ValueError("No video paths provided for concatenation.")
        if len(video_paths) == 1:
            import shutil
            shutil.copy(video_paths[0], output_path)
            return output_path

        filter_complex = ""
        inputs = []
        for i, path in enumerate(video_paths):
            inputs.extend(["-i", path])

        for i in range(len(video_paths) - 1):
            if i == 0:
                filter_complex += f"[0:v][1:v]xfade=transition=fade:duration={crossfade_duration}:offset={5.0625-crossfade_duration}[v01];"
            else:
                filter_complex += f"[v{i-1:02d}{i:02d}][{i+1}:v]xfade=transition=fade:duration={crossfade_duration}:offset={5.0625 * (i+1) - crossfade_duration * (i+1)}[v{i:02d}{i+1:02d}];"
        
        last_out = f"[v{len(video_paths)-2:02d}{len(video_paths)-1:02d}]"
        filter_complex = filter_complex.rstrip(';')

        cmd = ["ffmpeg", "-y"] + inputs + ["-filter_complex", filter_complex, "-map", last_out, output_path]
        
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return output_path

    def generate_long_video(self, prompt: str, total_duration_seconds: float, progress_callback: Optional[Callable[[float], None]] = None, **kwargs) -> str:
        clip_duration = 5.0625
        num_clips = math.ceil(total_duration_seconds / clip_duration)
        
        if num_clips == 0:
            raise ValueError("Total duration must be greater than 0")

        temp_dir = tempfile.mkdtemp(dir=self.output_dir)
        clip_paths = []
        
        from .wan import WanI2VRunner, WanT2VRunner
        
        # Determine base runner or fallback to Wan
        base_model_id = getattr(self.runner, "model_id", "wan-t2v-1.3b")
        t2v_runner = self.runner if hasattr(self.runner, "generate") else WanT2VRunner(model_id=base_model_id, device="cuda")
        i2v_runner = WanI2VRunner(model_id="wan-i2v-1.3b", device=getattr(t2v_runner, "device", "cuda"))

        try:
            for i in range(num_clips):
                clip_path = os.path.join(temp_dir, f"clip_{i}.mp4")
                
                def clip_progress(p):
                    if progress_callback:
                        overall = (i + p) / num_clips
                        progress_callback(overall)
                
                params = kwargs.copy()
                params["prompt"] = prompt
                params["output_path"] = clip_path
                
                if i == 0:
                    if hasattr(t2v_runner, "load"): t2v_runner.load()
                    t2v_runner.generate(params, clip_progress)
                    if hasattr(t2v_runner, "unload"): t2v_runner.unload()
                else:
                    prev_clip = clip_paths[-1]
                    last_frame_path = os.path.join(temp_dir, f"frame_{i}.png")
                    self.extract_last_frame(prev_clip, last_frame_path)
                    
                    params["image"] = last_frame_path
                    if hasattr(i2v_runner, "load"): i2v_runner.load()
                    i2v_runner.generate(params, clip_progress)
                    
                    # Unload only at the very end to save memory swapping overhead
                    if i == num_clips - 1 and hasattr(i2v_runner, "unload"):
                        i2v_runner.unload()
                
                clip_paths.append(clip_path)

            final_output = os.path.join(self.output_dir, "stitched_output.mp4")
            self.concatenate_videos(clip_paths, final_output)
            
            return final_output
        finally:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            if t2v_runner and t2v_runner.is_loaded:
                t2v_runner.unload()
            if i2v_runner and i2v_runner.is_loaded:
                i2v_runner.unload()
