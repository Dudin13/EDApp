"""
run_short_demo.py
================
Genera un video de 60s (1800 frames) para validación visual.
"""
import sys
import cv2
import numpy as np
from pathlib import Path

ROOT = Path("C:/apped")
sys.path.insert(0, str(ROOT / "app"))

from modules.detector import detect_frame
from modules.tracker import ProfessionalTracker
from modules.calibration_auto import AutoCalibrator

# Config
VIDEO_PATH = ROOT / "data/samples/test_5min.mp4"
OUTPUT_PATH = ROOT / "output/demo_v2.mp4"
MAX_FRAMES = 1800  # 60s @ 30fps

cap = cv2.VideoCapture(str(VIDEO_PATH))
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, fps, (w, h))

calibrator = AutoCalibrator(str(ROOT / "models/pitch.pt"))
tracker = ProfessionalTracker()

print(f"Generando demo corta (60s) en {OUTPUT_PATH}...")

count = 0
while count < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret: break
    
    # Procesamos 1 de cada 15 frames para ir rapido
    if count % 15 == 0:
        dets = detect_frame(frame, imgsz=1280, confidence=0.2)
        players = [d for d in dets if d.get("clase") == "player"]
        
        # Calib minimalista
        if len(players) >= 3:
            calibrator.calibrate(frame)
            
        # Dibujamos algo rapido
        for d in dets:
            cv2.rectangle(frame, (d["x"]-d["w"]//2, d["y"]-d["h"]//2), 
                          (d["x"]+d["w"]//2, d["y"]+d["h"]//2), (0,255,0), 2)
    
    out.write(frame)
    count += 1
    if count % 300 == 0: print(f"  [{count/MAX_FRAMES*100:.1f}%] frame {count}/{MAX_FRAMES}")

cap.release()
out.release()
print("¡Hecho!")
