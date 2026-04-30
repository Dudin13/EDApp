"""
audit_analysis.py — Script de auditoría para validar la calidad del análisis v2.
Ejecuta el pipeline sobre un clip de muestra y verifica métricas.
"""

import sys
import os
from pathlib import Path
import json

# Asegurar que el root del proyecto esté en el path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "app"))

from core.pipeline.video_pipeline import VideoAnalysisPipeline

def run_audit(video_path: str):
    print(f"🚀 Iniciando auditoría sobre: {video_path}")
    
    config = {
        "sample_rate": 0.5,
        "confidence": 0.25,
        "team": "Local Audit",
        "rival": "Visit Audit",
        "manual_seeds": []
    }
    
    pipeline = VideoAnalysisPipeline(config)
    final_results = {}
    
    print("⏳ Procesando frames (esto puede tardar unos minutos)...")
    for progreso, estado, resultados in pipeline.process(video_path):
        print(f"  [{progreso}%] {estado}")
        if resultados:
            final_results = resultados

    if not final_results:
        print("❌ Error: No se obtuvieron resultados del pipeline.")
        return

    # Validaciones
    print("\n--- RESULTADOS DE AUDITORÍA ---")
    tracks = final_results.get("tracks", {})
    events = final_results.get("ball_events", [])
    
    print(f"✅ Jugadores detectados: {len(tracks)}")
    print(f"✅ Eventos detectados: {len(events)}")
    
    # Verificar métricas físicas
    for tid, data in list(tracks.items())[:5]:
        print(f"  Jugador {tid}: {data['distance_km']} km | {data['top_speed']} km/h | {data['sprints']} sprints")
        
    if len(events) > 0:
        print(f"✅ Ejemplo de evento: {events[0]['action']} en minuto {events[0]['minute']:.2f}")
    else:
        print("⚠️ Advertencia: No se detectaron eventos (¿balón visible?)")

    print("\n✅ Auditoría completada.")

if __name__ == "__main__":
    # Buscar un video de muestra
    sample_video = project_root / "data" / "samples" / "test_match.mp4"
    if not sample_video.exists():
        # Intentar buscar cualquier mp4 en data/ o uploads/
        videos = list(project_root.glob("**/*.mp4"))
        if videos:
            sample_video = videos[0]
        else:
            print("❌ No se encontró ningún vídeo .mp4 para la auditoría.")
            sys.exit(1)
            
    run_audit(str(sample_video))
