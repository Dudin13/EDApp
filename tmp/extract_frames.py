import cv2
import os
from pathlib import Path

videos = [
    ("C:/Users/Usuario/Desktop/VideosDePrueba/j25-cadiz-cf-mirandilla-vs-coria-cf-2026-03-08.mp4", "j25_frame_"),
    ("C:/Users/Usuario/Desktop/VideosDePrueba/j7-mirandilla-vs-ad-ceuta-fc-b-2025-10-19.mp4", "j7_frame_"),
    ("C:/Users/Usuario/Desktop/VideosDePrueba/J14_DH_2526_CadeteA_MalagaCF.mp4", "j14_frame_"),
    ("C:/Users/Usuario/Desktop/VideosDePrueba/J6_DH_2525_CadeteA_UDAlmeria.mp4", "j6_frame_")
]

output_dir = Path("C:/apped/data/datasets/veo_frames_raw/images")
output_dir.mkdir(parents=True, exist_ok=True)

total_new_frames = 0

for video_path, prefix in videos:
    print(f"Processing: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening: {video_path}")
        continue
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"   Total frames: {total_frames}")
    
    # Extraer ~120 frames uniformemente
    num_to_extract = 120
    # Evitar extraer el último frame por si acaso
    step = max(1, total_frames // num_to_extract)
    
    extracted_in_video = 0
    for i in range(num_to_extract):
        frame_idx = i * step
        if frame_idx >= total_frames:
            break
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"   Error reading frame {frame_idx}")
            continue
            
        filename = f"{prefix}f{frame_idx:06d}.jpg"
        save_path = output_dir / filename
        cv2.imwrite(str(save_path), frame)
        extracted_in_video += 1
        
    cap.release()
    print(f"   Extracted {extracted_in_video} frames.")
    total_new_frames += extracted_in_video

print(f"\nEXTRACTION COMPLETED. Total new frames: {total_new_frames}")
