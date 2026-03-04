import os
import sys
from pathlib import Path
from modules.video_processor import VideoProcessor

# Configuración de prueba
VIDEO_PATH = "videos/test_match.mp4"
CONFIG = {
    "sample_rate": 5,      # cada 5 segundos para que sea rápido
    "detection_mode": "yolo",
    "confidence": 40,
    "team": "Mirandilla",
    "rival": "CCCFB",
    "jugadores_local": [{"dorsal": 1, "nombre": "Jugador 1", "equipo": "Mirandilla"}],
    "jugadores_visit": [{"dorsal": 1, "nombre": "Rival 1", "equipo": "CCCFB"}]
}

def verify():
    print("=" * 60)
    print("  VERIFICACIÓN HEADLESS DE ANÁLISIS")
    print("=" * 60)
    
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Vídeo no encontrado: {VIDEO_PATH}")
        return

    print("Iniciando analisis de " + VIDEO_PATH + "...")
    processor = VideoProcessor(VIDEO_PATH, CONFIG)
    
    final_results = {}
    try:
        for progreso, estado, resultados in processor.process():
            if progreso % 20 == 0:
                print(f"[{progreso}%] {estado}")
            if resultados:
                final_results = resultados
    except Exception as e:
        print(f"Error durante el proceso: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("  RESULTADOS DE LA VERIFICACION")
    print("=" * 60)
    
    if not final_results:
        print("No se obtuvieron resultados.")
        return

    n_jugadores = len(final_results.get("resultados_jugadores", {}))
    n_detecciones = final_results.get("total_detecciones", 0)
    n_eventos = len(final_results.get("ball_events", []))
    
    print(f"Jugadores identificados: {n_jugadores}")
    print(f"Total detecciones: {n_detecciones}")
    print(f"Eventos con balon (Clips potenciales): {n_eventos}")
    
    if n_eventos > 0:
        print("\nEjemplos de eventos detectados:")
        for ev in final_results["ball_events"][:3]:
            print(f"  - Min {ev['minute']:.2f}: {ev.get('nombre_jugador', '???')} ({ev.get('nombre_equipo', '???')})")
    else:
        print("\nNo se detectaron eventos con balon. Esto puede ser normal en un clip corto o si el balon no es visible.")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    verify()
