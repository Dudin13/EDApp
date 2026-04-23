import json
import subprocess
import os
from pathlib import Path

# ── Configuración ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "output/events.json"
OUTPUT_DIR = ROOT / "output/clips"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def make_clips():
    if not JSON_PATH.exists():
        print(f"[ERROR] No se encuentra {JSON_PATH}. Ejecuta run_demo_analysis.py primero.")
        return

    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    source_video = data.get("source_video", "")
    events       = data.get("events", [])

    if not source_video or not os.path.exists(source_video):
        print(f"[ERROR] Video fuente no encontrado: {source_video}")
        return

    print(f"[INFO] Iniciando generacion de {len(events)} clips...")

    print(f"[PATH] Destino: {OUTPUT_DIR}\n")


    count = 0
    for i, ev in enumerate(events):
        ts     = ev["timestamp"]
        minute = ev["minute"]
        action = ev["action"].replace(" ", "_").lower()
        team_id = ev["team"]
        team_str = "equipoA" if team_id == 0 else "equipoB" if team_id == 1 else "otro"
        
        # 6 segundos total (±3s del evento)
        start = max(0, ts - 3)
        duration = 6

        
        output_name = f"clip_{minute:04.1f}_{action}_{team_str}.mp4"
        output_path = OUTPUT_DIR / output_name
        
        # Comando ffmpeg optimizado
        # -ss ANTES de -i para fast seeking
        # Usamos libx264 ultrafast para asegurar que el corte sea preciso en el frame
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start:.2f}",
            "-t", str(duration),
            "-i", source_video,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-c:a", "aac",
            str(output_path)
        ]
        
        print(f"  [{i+1:02d}/{len(events):02d}] Generando: {output_name} ...", end="\r")
        
        try:
            subprocess.run(cmd, check=True)
            count += 1
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Fallo al generar clip {i}: {e}")

    print(f"\n\n[OK] Proceso completado.")
    print(f"[STATS] Clips generados: {count}")
    print(f"[PATH] Carpeta: {OUTPUT_DIR}")


if __name__ == "__main__":
    make_clips()
