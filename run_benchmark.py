import os
import sys
import json
import time
from pathlib import Path

# Añadir el directorio 'app' al path para importar módulos
sys.path.append(os.path.abspath("app"))

from modules.video_processor import VideoProcessor

def run_benchmark():
    video_path = "app/videos/test_5min.mp4"
    if not os.path.exists(video_path):
        print(f"Error: No se encuentra el video en {video_path}")
        return

    # Configuración para UCMCTrack
    config = {
        "sample_rate": 2,
        "detection_mode": "yolo",
        "tracker_mode": "ucmctrack",
        "confidence": 40,
        "pitch_width": 105,
        "pitch_height": 68
    }

    print(f"Iniciando benchmark con UCMCTrack sobre: {video_path}")
    print(f"Exportando tracks a: output/tracks_raw.txt")
    
    processor = VideoProcessor(video_path, config)
    
    start_time = time.time()
    
    # Ejecutar el generador del pipeline
    last_progreso = -1
    for progreso, estado, resultados in processor.process():
        if progreso != last_progreso:
            # Limpiar emojis para compatibilidad con consola windows si es necesario
            clean_estado = estado.encode('ascii', 'ignore').decode('ascii')
            print(f"Progreso: {progreso}% - {clean_estado}")
            last_progreso = progreso

    end_time = time.time()
    duration = end_time - start_time
    
    print("-" * 50)
    print(f"Benchmark completado en {duration:.1f} segundos")
    
    mot_file = "output/tracks_raw.txt"
    if os.path.exists(mot_file):
        file_size = os.path.getsize(mot_file)
        print(f"OK: Archivo MOT generado: {mot_file} ({file_size} bytes)")
        
        # Contar líneas para verificar datos
        with open(mot_file, "r") as f:
            lines = f.readlines()
            print(f"Total de líneas en el archivo MOT: {len(lines)}")
            if len(lines) > 0:
                print(f"Ejemplo de línea: {lines[0].strip()}")
    else:
        print(f"ERROR: No se generó el archivo {mot_file}")

if __name__ == "__main__":
    run_benchmark()
