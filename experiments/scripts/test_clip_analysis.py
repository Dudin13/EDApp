"""
test_clip_analysis.py - Prueba el pipeline de analisis en un clip del video de prueba.

Analiza 2 minutos y 30 segundos del video de prueba, comenzando en el minuto 2:38
(el partido empieza ahi). Usa el VideoProcessor real de la app.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "football_analyzer"))

import cv2
from pathlib import Path
import time

# ---------------------------------------------------------------------------
# Parametros del test
# ---------------------------------------------------------------------------
VIDEO_PATH = r"C:\apped\football_analyzer\videos\test_match.mp4"
START_SECOND = 2 * 60 + 38   # 2:38
CLIP_DURATION = 2 * 60 + 30  # 2:30
OUTPUT_CLIP = r"C:\apped\test_clip_2m30s.mp4"

# ---------------------------------------------------------------------------
# 1. Extraer el clip
# ---------------------------------------------------------------------------
def extract_clip(video_path, start_s, duration_s, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir: " + video_path)
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_dur = total_frames / fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Video original: {total_dur/60:.1f} min | {w}x{h} @ {fps:.1f} fps")

    start_frame = int(start_s * fps)
    end_frame = min(int((start_s + duration_s) * fps), total_frames)
    num_frames = end_frame - start_frame
    print(f"[INFO] Extrayendo frames {start_frame}->{end_frame} ({num_frames} frames, {duration_s}s)")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0
    while written < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        written += 1
        if fps > 0 and written % int(fps * 10) == 0:
            print(f"   ... {written}/{num_frames} frames extraidos")

    cap.release()
    out.release()
    print(f"[OK] Clip guardado: {output_path} ({written} frames)\n")
    return written > 0


# ---------------------------------------------------------------------------
# 2. Analizar el clip con VideoProcessor
# ---------------------------------------------------------------------------
def run_analysis(clip_path):
    from modules.video_processor import VideoProcessor

    config = {
        "sample_rate": 2,
        "detection_mode": "yolo",
        "confidence": 40,
        "jugadores_local": [],
        "jugadores_visit": [],
        "team": "Local",
        "rival": "Visitante",
    }

    print("[INFO] Iniciando analisis con VideoProcessor...")
    print("       (1 frame cada 2 segundos)\n")

    processor = VideoProcessor(clip_path, config)
    resultados = {}
    t0 = time.time()

    for progress, status, partial in processor.process():
        # Strip non-ASCII to avoid cp1252 errors
        safe_status = status.encode("ascii", "replace").decode("ascii")
        print(f"  [{progress:3d}%] {safe_status}")
        if progress == 100:
            resultados = partial

    elapsed = time.time() - t0
    print(f"\n[INFO] Tiempo de analisis: {elapsed:.1f}s")

    rj = resultados.get("resultados_jugadores", {})
    ball_events = resultados.get("ball_events", [])
    total_dets = resultados.get("total_detecciones", 0)

    print("=" * 55)
    print("RESUMEN DEL ANALISIS")
    print("=" * 55)
    print(f"  Jugadores/tracks unicos detectados: {len(rj)}")
    print(f"  Total detecciones                 : {total_dets}")
    print(f"  Eventos con balon registrados     : {len(ball_events)}")

    if ball_events:
        print("\nEVENTOS DE BALON:")
        for ev in ball_events:
            m = ev["minute"]
            nombre = ev.get("nombre_jugador", f"Track {ev['track_id']}")
            eq = ev.get("nombre_equipo", "?")
            print(f"   min {m:5.2f}  {nombre} ({eq})")

    print("=" * 55)
    print("[OK] Test completado correctamente.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not Path(OUTPUT_CLIP).exists():
        ok = extract_clip(VIDEO_PATH, START_SECOND, CLIP_DURATION, OUTPUT_CLIP)
        if not ok:
            sys.exit(1)
    else:
        print(f"[INFO] Usando clip ya existente: {OUTPUT_CLIP}\n")

    run_analysis(OUTPUT_CLIP)
