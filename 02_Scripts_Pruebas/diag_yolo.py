"""
diag_yolo.py - Diagnostica que detecta el modelo YOLO en frames del clip de prueba.
"""
import sys
sys.path.insert(0, "football_analyzer")

from ultralytics import YOLO
from pathlib import Path
import cv2

SEG_PATH = Path("train_yolo/runs/segment/train/weights/best.pt")
DET_PATH = Path("train_yolo/runs/detect/train/weights/best.pt")

model_path = SEG_PATH if SEG_PATH.exists() else DET_PATH
print(f"[INFO] Usando modelo: {model_path}")

model = YOLO(str(model_path))
print(f"[INFO] Clases del modelo: {model.names}")

cap = cv2.VideoCapture(r"C:\apped\test_clip_2m30s.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"[INFO] Clip: {total_frames} frames @ {fps:.1f} fps = {total_frames/fps:.1f}s")

# Analizar 5 frames distribuidos a lo largo del clip
ball_found = 0
sample_frames = [int(total_frames * i / 5) for i in range(5)]

for fi in sample_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    if not ret:
        print(f"[WARN] No se pudo leer frame {fi}")
        continue

    results = model(frame, conf=0.01, verbose=False)
    dets_by_class = {}
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            name = model.names.get(cls, str(cls))
            conf = float(box.conf[0])
            dets_by_class.setdefault(name, []).append(round(conf, 3))

    t = fi / fps
    print(f"\n[Frame {fi} | t={t:.1f}s]")
    if dets_by_class:
        for cls_name, confs in sorted(dets_by_class.items()):
            print(f"   {cls_name}: {len(confs)} detecciones | max_conf={max(confs):.3f}")
            if cls_name == "ball":
                ball_found += len(confs)
    else:
        print("   (sin detecciones)")

cap.release()
print(f"\n[RESUMEN] Total detecciones de balon con conf>0.01: {ball_found}")
