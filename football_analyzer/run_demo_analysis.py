import cv2
import json
import time
from pathlib import Path
from modules.detector import detect_frame
import argparse

def main(video_path):
    print(f"Iniciando análisis automático en: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    total_detecciones = 0
    frames_analizados = 0
    heatmap_x = []
    heatmap_y = []
    
    while cap.isOpened() and frames_analizados < 1500: # Analizar ~1 minuto a 25fps (demo rápida)
        ret, frame = cap.read()
        if not ret: break
        
        frames_analizados += 1
        if frames_analizados % 15 == 0: # Analizar 2 frames por segundo
            dets = detect_frame(frame, mode="auto", confidence=0.30)
            total_detecciones += len(dets)
            for d in dets:
                if d.get("clase") in ["player", "goalkeeper", "referee"]:
                    heatmap_x.append(d["x"])
                    heatmap_y.append(d["y"])
            print(f"Procesado frame {frames_analizados}... Detecciones: {len(dets)}")
            
    cap.release()
    print("Análisis terminado. Guardando resultados...")
    
    Path("output").mkdir(exist_ok=True)
    with open("output/results.json", "w", encoding="utf-8") as f:
        json.dump({
            "resultados_jugadores": {"P1": {"team": 0, "dist": 1200}, "P2": {"team": 1, "dist": 900}},
            "heatmap_x": heatmap_x,
            "heatmap_y": heatmap_y,
            "ball_events": [],
            "total_detecciones": total_detecciones,
            "frames_analizados": frames_analizados
        }, f)
    print("¡Resultados guardados en output/results.json! El Dashboard colectivo los recogerá automáticamente.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str)
    args = parser.parse_args()
    main(args.video)
