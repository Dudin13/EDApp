import cv2
import os
from pathlib import Path

def extract_frames(video_path, output_dir, num_frames=30):
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: Video not found at {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Avoid first and last 5 minutes if video is long enough
    skip_frames = int(5 * 60 * fps)
    if total_frames > 3 * skip_frames:
        start_frame = skip_frames
        end_frame = total_frames - skip_frames
    else:
        start_frame = 0
        end_frame = total_frames

    interval = (end_frame - start_frame) // num_frames
    
    video_name = video_path.stem
    print(f"Extracting {num_frames} frames from {video_name}...")

    count = 0
    for i in range(num_frames):
        target_frame = start_frame + (i * interval)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret:
            break
        
        output_path = output_dir / f"{video_name}_f{target_frame}.jpg"
        cv2.imwrite(str(output_path), frame)
        count += 1

    cap.release()
    print(f"Successfully extracted {count} frames to {output_dir}")

if __name__ == "__main__":
    videos = [
        r"C:\Users\Usuario\Desktop\VideosDePrueba\j16clucena-vs-mirandilla-2026-01-04.mp4",
        r"C:\Users\Usuario\Desktop\VideosDePrueba\j22-chiclana-cf-mirandilla-2026-02-14.mp4",
        r"C:\Users\Usuario\Desktop\VideosDePrueba\j24-ceuta-b-vs-cadiz-cf-mirandilla-2026-02-28.mp4"
    ]
    
    out_dir = Path(r"c:\apped\data\datasets\nuevas_muestras_marzo")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for v in videos:
        extract_frames(v, out_dir, 30)
