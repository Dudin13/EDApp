from ultralytics import YOLO
import os

models = [
    'football-player-detection_akram.pt', 
    'football-ball-detection_akram.pt', 
    'football-pitch-detection_akram.pt',
    'players.pt'
]

for m in models:
    path = os.path.join("models", m)
    if os.path.exists(path):
        model = YOLO(path)
        print(f"Model: {m}")
        print(f"  Names: {model.names}")
        print(f"  Task: {model.task}")
    else:
        print(f"Model not found: {m}")
