import cv2
import os
from pathlib import Path

def extract_frames(video_dir, output_dir, interval_seconds=8, target_count=None, prefix=None, auto_label=False, conf=0.4):
    """
    Extrae frames de todos los videos de un directorio.

    Args:
        video_dir:        Carpeta con los videos
        output_dir:       Carpeta de salida para los frames
        interval_seconds: Intervalo entre frames extraidos (si target_count es None)
        target_count:     Numero total de frames deseados (distribuidos uniformemente)
        prefix:           Prefijo para el nombre de los archivos (default: video stem)
        auto_label:       Si True, auto-etiqueta con best.pt a imgsz=1280
        conf:             Confianza minima para el auto-etiquetado
    """
    video_dir  = Path(video_dir)
    output_dir = Path(output_dir) / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.ts')
    video_files = sorted([
        f for f in video_dir.iterdir()
        if f.suffix.lower() in video_extensions
    ])

    if not video_files:
        print(f"[ERROR] No se encontraron videos en {video_dir}")
        return

    print(f"[INFO] Encontrados {len(video_files)} videos.")
    if target_count:
        print(f"[INFO] Objetivo: {target_count} frames totales | Prefijo: {prefix}")
    else:
        print(f"[INFO] Intervalo: {interval_seconds}s | Auto-label: {auto_label}")
    print()

    # Cargar modelo para auto-etiquetado si se solicita
    model = None
    if auto_label:
        try:
            from ultralytics import YOLO
            BASE       = Path(os.environ.get("APPED_ROOT", "C:/apped"))
            model_path = BASE / "ml" / "training" / "runs" / "segment" / "players_v1" / "weights" / "best.pt"
            if not model_path.exists():
                # Buscar cualquier best.pt disponible
                for pt in (BASE / "ml" / "training" / "runs").rglob("best.pt"):
                    model_path = pt
                    break
            if model_path.exists():
                model = YOLO(str(model_path))
                print(f"[INFO] Modelo cargado para auto-etiquetado: {model_path.name}")
            else:
                print(f"[WARN] No se encontro best.pt - extrayendo frames sin etiquetar")
                auto_label = False
        except ImportError:
            print(f"[WARN] ultralytics no disponible - extrayendo sin etiquetar")
            auto_label = False

    total_extracted = 0
    total_labeled   = 0

    for video_file in video_files:
        print(f"[PROC] {video_file.name}")
        cap = cv2.VideoCapture(str(video_file))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_v_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calcular intervalo dinámico si se pide target_count
        if target_count:
            # Estimación de frames "útiles" (restando halftime y primer minuto)
            # Aproximadamente 90 min de juego = 5400s
            useful_seconds = (total_v_frames / fps) - 900 - 60 
            if useful_seconds < 0: useful_seconds = total_v_frames / fps
            
            frame_interval = int((useful_seconds * fps) / target_count)
            frame_interval = max(1, frame_interval)
        else:
            frame_interval = int(fps * interval_seconds)

        count              = 0
        extracted_from_video = 0
        current_prefix = prefix if prefix else video_file.stem

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_sec = count / fps
            current_min = current_sec / 60

            # Saltar descanso (min 47-62)
            if 47 <= current_min <= 62:
                count += 1
                continue

            # Saltar primeros 60 segundos (logos, presentacion)
            if current_sec < 60:
                count += 1
                continue

            if count % frame_interval == 0:
                frame_name  = f"{current_prefix}_min{int(current_min):03d}_f{count:06d}.jpg"
                output_path = output_dir / frame_name
                cv2.imwrite(str(output_path), frame)
                extracted_from_video += 1
                total_extracted += 1

                # Auto-etiquetar si se solicita
                if auto_label and model is not None:
                    _auto_label_frame(model, frame, output_path, conf)
                    total_labeled += 1
                
                if target_count and extracted_from_video >= target_count:
                    break

            count += 1

        cap.release()
        print(f"  [OK] {extracted_from_video} frames extraidos de {video_file.name}")

    print()
    print(f"[DONE] Total frames extraidos: {total_extracted}")
    if auto_label:
        print(f"[DONE] Total frames etiquetados: {total_labeled}")
    print(f"[PATH] {output_dir.absolute()}")


def _auto_label_frame(model, frame, img_path: Path, conf: float):
    """
    Auto-etiqueta un frame con el modelo YOLO y guarda el .txt en formato YOLO.
    Usa imgsz=1280 para detectar bien en camaras VEO panoramicas.
    """
    results = model(frame, imgsz=1280, conf=conf, verbose=False)
    # Corrección: labels_autolabel debe estar al nivel de images/
    labels_dir = img_path.parent.parent / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    label_path = labels_dir / (img_path.stem + ".txt")

    lines = []
    for r in results:
        for box in r.boxes:
            cls  = int(box.cls[0])
            cx   = float(box.xywhn[0][0])
            cy   = float(box.xywhn[0][1])
            bw   = float(box.xywhn[0][2])
            bh   = float(box.xywhn[0][3])
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    label_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    BASE = Path(os.environ.get("APPED_ROOT", "C:/apped"))

    # ── CONFIG PARA MEDICOACH J35 ──────────────────────────────────────────
    VIDEO_FILE    = r"C:\Users\Usuario\Desktop\VideosDePrueba\2025-2026_LALIGAHYPERMOTION_J35_CAD-AND_TAC_v1.mp4"
    OUTPUT_BASE   = BASE / "data" / "datasets" / "veo_frames_raw"
    
    # Extraer 200 frames distribuidos
    extract_frames(
        video_dir=Path(VIDEO_FILE).parent,
        output_dir=OUTPUT_BASE,
        target_count=200,
        prefix="mediacoach_j35",
        auto_label=False # El usuario dijo "No entrenes aun", evito overhead
    )

