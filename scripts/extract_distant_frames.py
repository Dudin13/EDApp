import cv2
import os
from ultralytics import YOLO
import random

def main():
    model_path = r'c:\apped\models\players_v6.pt'
    samples_dir = r'c:\apped\data\samples'
    output_dir = r'c:\apped\data\para_etiquetar\extremos_campo'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model = YOLO(model_path)
    
    videos = [f for f in os.listdir(samples_dir) if f.lower().endswith('.mp4')]
    if not videos:
        print("No videos found in samples.")
        return

    frames_needed = 300
    frames_extracted = 0
    
    # Border threshold (10% of width/height)
    # Small player threshold (e.g., area < 0.1% of image)
    
    print(f"Processing videos to extract {frames_needed} frames...")
    
    for video_name in videos:
        if frames_extracted >= frames_needed: break
        
        video_path = os.path.join(samples_dir, video_name)
        cap = cv2.VideoCapture(video_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // 1000) # Sample 1000 frames per video to check
        
        frame_idx = 0
        while cap.isOpened() and frames_extracted < frames_needed:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: break
            
            h, w, _ = frame.shape
            
            # Run detection
            results = model(frame, verbose=False)[0]
            boxes = results.boxes.xyxy.cpu().numpy()
            
            is_interesting = False
            
            # Condition 1: Few detections
            if len(boxes) < 5 and len(boxes) > 0:
                is_interesting = True
            
            # Condition 2: Players at extremes/small
            if not is_interesting:
                for box in boxes:
                    bx1, by1, bx2, by2 = box
                    bw = bx2 - bx1
                    bh = by2 - by1
                    # Small if width < 3% of screen width
                    is_small = bw < (w * 0.03)
                    # Near border if center is in outer 15%
                    cx = (bx1 + bx2) / 2
                    is_near_border = cx < (w * 0.15) or cx > (w * 0.85)
                    
                    if is_small or is_near_border:
                        is_interesting = True
                        break
            
            if is_interesting:
                frame_name = f"distant_{video_name}_{frame_idx}.jpg"
                cv2.imwrite(os.path.join(output_dir, frame_name), frame)
                frames_extracted += 1
                if frames_extracted % 50 == 0:
                    print(f"Extracted {frames_extracted}/{frames_needed} frames...")
            
            frame_idx += step
            if frame_idx >= total_frames: break
            
        cap.release()

    print(f"Extraction complete. Total frames: {frames_extracted}")

if __name__ == "__main__":
    main()
