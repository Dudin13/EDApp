import cv2
from pathlib import Path
import os
import sys

def main():
    root_path = Path(__file__).parent.parent.absolute()
    video_path = root_path / "data" / "samples" / "tactical_10min.mp4"
    output_dir = root_path / "data" / "para_etiquetar" / "tactical_frames"
    
    if not video_path.exists():
        print(f"Error: No se encontró el video {video_path}")
        sys.exit(1)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Error: No se pudo abrir el video.")
        sys.exit(1)
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if fps > 0 else 0
    
    print(f"Video cargado: {video_path.name}")
    print(f"Duración: {duration_s:.1f} segundos ({total_frames} frames a {fps} fps)")
    
    # Queremos 1 frame cada 2 segundos
    frame_interval = int(fps * 2)
    expected_frames = int(duration_s / 2)
    
    print(f"Extrayendo 1 frame cada 2 segundos...")
    print(f"Se generarán aproximadamente {expected_frames} frames.")
    
    saved_count = 0
    current_frame = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_frame % frame_interval == 0:
            second = current_frame / fps
            out_filename = output_dir / f"tactical_10min_sec_{int(second):04d}.jpg"
            cv2.imwrite(str(out_filename), frame)
            saved_count += 1
            if saved_count % 50 == 0:
                print(f"  Guardados {saved_count}/{expected_frames} frames...")
                
        current_frame += 1
        
    cap.release()
    print(f"\nFinalizado. Total de frames guardados: {saved_count}")
    print(f"Ruta de salida: {output_dir}")

if __name__ == "__main__":
    main()
