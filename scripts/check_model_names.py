
from ultralytics import YOLO
from pathlib import Path

ROOT = Path("C:/apped")
v4_path = ROOT / "models/players.pt"
v3_path = ROOT / "models/players_v3.pt"

print("V4 (Current) names:", YOLO(str(v4_path)).names)
print("V3 (Previous) names:", YOLO(str(v3_path)).names)
