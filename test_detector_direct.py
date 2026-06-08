import sys
import os
import cv2

sys.path.append(os.path.abspath("app"))
from modules.detector import detect_frame, _load_player_model

def main():
    video_path = "data/samples/test_5min.mp4"
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        return
        
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_frame = int(fps * 2 * 140) # Approximately frame 140 of the MOT (70s)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    
    if not ret:
        print("Could not read frame")
        return
        
    print(f"Frame {target_frame} loaded. Shape: {frame.shape}")
    
    # 1. Which model is loaded?
    model = _load_player_model()
    print("Model currently loaded:", getattr(model, 'pt_path', getattr(model, 'ckpt_path', str(model))))
    
    # 2. Test with default config
    dets_default = detect_frame(frame, mode="yolo", confidence=40)
    print(f"Detections at conf=0.40: {len(dets_default)}")
    for d in dets_default:
        print(f" - {d.get('clase')}: conf={d.get('conf', d.get('confianza', 0)):.2f}, bbox=({d['x']:.1f}, {d['y']:.1f})")
        
    # 3. Test with low confidence
    dets_low = detect_frame(frame, mode="yolo", confidence=25)
    print(f"\nDetections at conf=0.25: {len(dets_low)}")
    for d in dets_low:
        print(f" - {d.get('clase')}: conf={d.get('conf', d.get('confianza', 0)):.2f}, bbox=({d['x']:.1f}, {d['y']:.1f})")
        
    cap.release()

if __name__ == "__main__":
    main()
