import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
import numpy as np

# Rutas
ROOT = Path("c:/apped")
MODEL_PLAYERS = ROOT / "models/players.pt"
MODEL_BALL = ROOT / "models/ball_specialist.pt"
VIDEO_PATH = ROOT / "data/samples/test_5min.mp4"
OUTPUT_DIR = ROOT / "output/benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_benchmark():
    print("Cargando modelos...")
    players_model = YOLO(str(MODEL_PLAYERS))
    ball_model = YOLO(str(MODEL_BALL))
    
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print("Error abriendo vídeo")
        return

    frames_processed = 0
    ball_detected_frames_new = 0
    ball_detected_frames_base = 0
    max_frames = 1000
    
    captures = []
    capture_interval = 100 # Guardar uno cada 100 frames si tiene balón
    
    print(f"Procesando {max_frames} frames...")
    
    while frames_processed < max_frames:
        ret, frame = cap.read()
        if not ret: break
        
        frames_processed += 1
        
        # 1. Inferencia Jugadores (Res estándar 640)
        results_p = players_model.predict(frame, imgsz=640, conf=0.45, verbose=False)[0]
        
        # 2. Inferencia Balón (Especialista 1280)
        results_b = ball_model.predict(frame, imgsz=1280, conf=0.10, verbose=False)[0]
        
        # Contar balón BASELINE (de players_model)
        ball_found_base = False
        for det in results_p.boxes:
            if int(det.cls[0]) == 3: # En players.pt el balon es clase 3
                ball_found_base = True
                break
        if ball_found_base:
            ball_detected_frames_base += 1

        # Contar balón NUEVO (de ball_specialist)
        ball_found_new = False
        ball_dets = []
        for det in results_b.boxes:
            if int(det.cls[0]) == 0: 
                ball_found_new = True
                conf = float(det.conf[0])
                xyxy = det.xyxy[0].cpu().numpy()
                ball_dets.append((xyxy, conf))
        
        if ball_found_new:
            ball_detected_frames_new += 1
            
        # Generar capturas si es necesario
        if ball_found_new and len(captures) < 3:
            if frames_processed % 30 == 0 or len(captures) == 0:
                vis = frame.copy()
                # Dibujar jugadores
                for det in results_p.boxes:
                    cls = int(det.cls[0])
                    if cls in [0, 1, 2]: # player, goalkeeper, referee
                        xyxy = det.xyxy[0].cpu().numpy()
                        cv2.rectangle(vis, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                
                # Dibujar balón
                for xyxy, conf in ball_dets:
                    cv2.circle(vis, (int((xyxy[0]+xyxy[2])/2), int((xyxy[1]+xyxy[3])/2)), 12, (0, 255, 255), -1)
                    cv2.putText(vis, f"Ball {conf:.2f}", (int(xyxy[0]), int(xyxy[1]-10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                cap_path = OUTPUT_DIR / f"benchmark_ball_{len(captures)+1}.jpg"
                cv2.imwrite(str(cap_path), vis)
                captures.append(str(cap_path))
                print(f"  [CAP] Captura {len(captures)} guardada en frame {frames_processed}")

        if frames_processed % 50 == 0:
            print(f"  Progreso: {frames_processed}/{max_frames}...")

    cap.release()
    
    coverage_new = (ball_detected_frames_new / frames_processed) * 100
    coverage_base = (ball_detected_frames_base / frames_processed) * 100
    print("\n" + "="*30)
    print("RESULTADOS BENCHMARK (1000 frames)")
    print("="*30)
    print(f"Frames totales:    {frames_processed}")
    print(f"Balón BASELINE:    {ball_detected_frames_base} ({coverage_base:.1f}%)")
    print(f"Balón ESPECIALISTA: {ball_detected_frames_new} ({coverage_new:.1f}%)")
    print(f"Diferencia:        {coverage_new - coverage_base:+.1f}%")
    print("="*30)

if __name__ == "__main__":
    run_benchmark()
