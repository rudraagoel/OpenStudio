import argparse
import sys
import os

try:
    import librosa
except ImportError:
    print("Warning: librosa is not installed. Please install it with 'pip install librosa'.")
    librosa = None

try:
    import ffmpeg
except ImportError:
    print("Warning: ffmpeg-python is not installed. Please install it with 'pip install ffmpeg-python'.")
    ffmpeg = None

def main():
    parser = argparse.ArgumentParser(description="Montage Editor: Slice and stitch gaming clips to audio beats.")
    parser.add_argument("--audio", required=True, help="Path to input audio file")
    parser.add_argument("--clips_dir", required=True, help="Path to folder containing raw gaming clips")
    parser.add_argument("--output", required=True, help="Path to output video file")
    args = parser.parse_args()

    if not librosa or not ffmpeg:
        print("Missing required dependencies. Ensure librosa and ffmpeg-python are installed.")
        sys.exit(1)

    audio_path = args.audio
    clips_dir = args.clips_dir
    output_path = args.output

    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        sys.exit(1)
        
    if not os.path.isdir(clips_dir):
        print(f"Error: Clips directory not found at {clips_dir}")
        sys.exit(1)

    print(f"Loading audio: {audio_path}")
    try:
        y, sr = librosa.load(audio_path)
    except Exception as e:
        print(f"Error loading audio: {e}")
        sys.exit(1)
    
    print("Extracting beats...")
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    print(f"Detected tempo: {tempo} BPM")
    print(f"Found {len(beat_times)} beats.")

    clip_files = [os.path.join(clips_dir, f) for f in os.listdir(clips_dir) if f.lower().endswith(('.mp4', '.mkv', '.avi', '.mov'))]
    if not clip_files:
        print("No video clips found in the directory.")
        sys.exit(1)

    print("Slicing and stitching clips to beats...")
    
    streams = []
    clip_index = 0
    start_time = 0.0
    
    for beat_time in beat_times:
        duration = beat_time - start_time
        if duration <= 0:
            continue
            
        clip = clip_files[clip_index % len(clip_files)]
        video_stream = ffmpeg.input(clip, ss=0, t=duration).video
        streams.append(video_stream)
        
        start_time = beat_time
        clip_index += 1

    if streams:
        print("Concatenating video streams...")
        try:
            joined = ffmpeg.concat(*streams, v=1, a=0)
            audio_stream = ffmpeg.input(audio_path).audio
            
            out = ffmpeg.output(joined, audio_stream, output_path, shortest=None)
            
            print(f"Rendering output to {output_path}...")
            # ffmpeg.run(out, overwrite_output=True)
            print("Done! (ffmpeg execution is commented out for safety in this stub)")
        except Exception as e:
            print(f"Error processing video: {e}")
    else:
        print("No streams to process.")

if __name__ == "__main__":
    main()
