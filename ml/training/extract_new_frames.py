import cv2
import os
from pathlib import Path
import time

def extract_frames(video_dir, output_dir, interval_seconds=5):
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.ts')
    video_files = [f for f in video_dir.iterdir() if f.suffix.lower() in video_extensions]
    
    if not video_files:
        print(f"[ERROR] No se encontraron videos en {video_dir}")
        return

    print(f"[INFO] Encontrados {len(video_files)} videos. Iniciando extraccion...")
    
    total_extracted = 0
    for video_file in video_files:
        print(f"[PROCESS] Procesando: {video_file.name}")
        cap = cv2.VideoCapture(str(video_file))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 25 # Fallback
        
        frame_interval = int(fps * interval_seconds)
        count = 0
        extracted_from_video = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calcular tiempo actual en segundos
            current_sec = count / fps
            current_min = current_sec / 60
            
            # SECCIÓN DE SALTO: Evitar descanso (típicamente entre min 47 y 62)
            if 47 <= current_min <= 62:
                count += 1
                continue

            if count % frame_interval == 0:
                frame_name = f"{video_file.stem}_min{int(current_min):03d}_f{count:06d}.jpg"
                output_path = output_dir / frame_name
                cv2.imwrite(str(output_path), frame)
                extracted_from_video += 1
                total_extracted += 1
            
            count += 1
            
        cap.release()
        print(f"[OK] Extraidos {extracted_from_video} frames de {video_file.name}")

    print(f"[DONE] Proceso completado. Total frames extraidos: {total_extracted}")
    print(f"[PATH] Localizacion: {output_dir.absolute()}")

if __name__ == "__main__":
    SOURCE_VIDEOS = r"C:\Users\Usuario\Desktop\VideosDePrueba"
    OUTPUT_IMAGES = r"c:\apped\04_Datasets_Entrenamiento\new_frames_raw"
    
    extract_frames(SOURCE_VIDEOS, OUTPUT_IMAGES, interval_seconds=10)
