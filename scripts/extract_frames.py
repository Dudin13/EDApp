import cv2
import glob
import os
import sys

src_dir = r"C:\Users\Usuario\Desktop\VideosDePrueba"
dst_dir = r"C:\apped\data\datasets\veo_frames_raw\images"

os.makedirs(dst_dir, exist_ok=True)
videos = glob.glob(os.path.join(src_dir, "*.mp4"))

total_frames_extracted = 0
videos_processed = 0

print(f"Buscando videos en {src_dir}...")
for vid_path in videos:
    vid_name = os.path.basename(vid_path).replace(".mp4", "")
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print(f"Error opening {vid_path}")
        continue
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        continue
        
    num_frames = 250
    indices = [int(i * (total_frames - 1) / num_frames) for i in range(num_frames)]
    
    extracted = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            out_file = os.path.join(dst_dir, f"{vid_name}_f{idx:06d}.jpg")
            cv2.imwrite(out_file, frame)
            extracted += 1
            total_frames_extracted += 1
    
    cap.release()
    print(f"Procesado '{vid_name}': {extracted} frames.")
    videos_processed += 1

print(f"==== RESULTADOS ====")
print(f"TOTAL_VIDEOS={videos_processed}")
print(f"TOTAL_FRAMES={total_frames_extracted}")
