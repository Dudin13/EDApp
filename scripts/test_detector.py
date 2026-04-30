"""
test_detector.py — Test rápido del motor de detección.
"""
import cv2
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "app"))

from modules.detector import detect_frame, PLAYER_MODEL_PATH, BALL_MODEL_PATH, PITCH_MODEL_PATH

def test():
    print(f"DEBUG: PLAYER_MODEL_PATH = {PLAYER_MODEL_PATH} (Exists: {PLAYER_MODEL_PATH.exists()})")
    print(f"DEBUG: BALL_MODEL_PATH = {BALL_MODEL_PATH} (Exists: {BALL_MODEL_PATH.exists()})")
    print(f"DEBUG: PITCH_MODEL_PATH = {PITCH_MODEL_PATH} (Exists: {PITCH_MODEL_PATH.exists()})")
    
    video_path = "C:/D/EDApp/data/samples/test_match.mp4"
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100) # Saltar al frame 100
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Error: No se pudo leer el video.")
        return

    print("⏳ Ejecutando detección en el primer frame...")
    detections = detect_frame(frame, mode="auto", confidence=25)
    print(f"✅ Detecciones encontradas: {len(detections)}")
    for d in detections[:5]:
        print(f"  - {d['clase']} (conf: {d['conf']}) en {d['x']},{d['y']}")

if __name__ == "__main__":
    test()
