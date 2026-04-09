
from ultralytics import YOLO
from pathlib import Path

ROOT = Path("C:/apped")
v3_path = ROOT / "models/players_v3.pt"
v4_path = ROOT / "models/players.pt"

if v3_path.exists():
    v3 = YOLO(str(v3_path))
    print(f"V3 Task: {v3.task}")
    print(f"V3 Names: {v3.names}")
if v4_path.exists():
    v4 = YOLO(str(v4_path))
    print(f"V4 Task: {v4.task}")
    print(f"V4 Names: {v4.names}")
