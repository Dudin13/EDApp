
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time

ROOT = Path("C:/apped")
VIDEO_PATH = ROOT / "data/samples/test_5min.mp4"
MODELS = {
    "V4 (Current)": ROOT / "models/players.pt",
    "V3 (Previous)": ROOT / "models/players_v3.pt"
}
MAX_FRAMES = 300
CONF = 0.20
IMGSZ = 1280

def evaluate_model(name, path):
    print(f"Evaluating {name}...")
    model = YOLO(str(path))
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    
    total_players = 0
    frame_count = 0
    
    start_time = time.time()
    
    while frame_count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Procesar coincidiendo con la lógica de detector.py (imgsz=1280 para VEO)
        results = model.predict(frame, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
        
        # Contar clases: 1 (player) y 0 (goalkeeper) según el modelo kaggle/players.pt
        # Nota: La numeración puede variar, pero filtramos por nombre si es posible
        detections = results.boxes
        players_in_frame = 0
        for box in detections:
            cls = int(box.cls[0])
            name_cls = results.names[cls]
            if name_cls in ("player", "goalkeeper"):
                players_in_frame += 1
        
        total_players += players_in_frame
        frame_count += 1
        
        if frame_count % 50 == 0:
            print(f"  Frame {frame_count}/{MAX_FRAMES}...")
            
    cap.release()
    avg = total_players / frame_count if frame_count > 0 else 0
    duration = time.time() - start_time
    return avg, frame_count, duration

results_summary = {}
for name, path in MODELS.items():
    if path.exists():
        avg, count, dur = evaluate_model(name, path)
        results_summary[name] = {"avg": avg, "frames": count, "time": dur}
    else:
        print(f"Model {path} not found!")

print("\n" + "="*40)
print("COMPARISON RESULTS")
print("="*40)
for name, data in results_summary.items():
    print(f"{name}: {data['avg']:.2f} players/frame (avg) | {data['time']:.1f}s for {data['frames']} frames")
print("="*40)
