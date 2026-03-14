import cv2
import os
from pathlib import Path

def extract_frames(video_dir, output_dir, interval_seconds=8, auto_label=False, conf=0.4):
    """
    Extrae frames de todos los videos de un directorio.

    Args:
        video_dir:        Carpeta con los videos
        output_dir:       Carpeta de salida para los frames
        interval_seconds: Intervalo entre frames extraidos (default 8s)
        auto_label:       Si True, auto-etiqueta con best.pt a imgsz=1280
        conf:             Confianza minima para el auto-etiquetado
    """
    video_dir  = Path(video_dir)
    output_dir = Path(output_dir)
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

        frame_interval     = int(fps * interval_seconds)
        count              = 0
        extracted_from_video = 0

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
                frame_name  = f"{video_file.stem}_min{int(current_min):03d}_f{count:06d}.jpg"
                output_path = output_dir / frame_name
                cv2.imwrite(str(output_path), frame)
                extracted_from_video += 1
                total_extracted += 1

                # Auto-etiquetar si se solicita
                if auto_label and model is not None:
                    _auto_label_frame(model, frame, output_path, conf)
                    total_labeled += 1

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
    labels_dir = img_path.parent.parent / "labels_autolabel"
    labels_dir.mkdir(parents=True, exist_ok=True)

    label_path = labels_dir / (img_path.stem + ".txt")
    h, w = frame.shape[:2]

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

    # ── CONFIGURA AQUI ─────────────────────────────────────────────────────
    SOURCE_VIDEOS = r"C:\Users\Usuario\Desktop\VideosDePrueba"
    OUTPUT_IMAGES = str(BASE / "data" / "datasets" / "veo_frames_raw")

    # interval_seconds=8  → ~150 frames por partido de 90 min (recomendado)
    # auto_label=True     → etiqueta automaticamente con best.pt a imgsz=1280
    # conf=0.4            → confianza minima para aceptar una deteccion
    # ───────────────────────────────────────────────────────────────────────

    extract_frames(
        video_dir=SOURCE_VIDEOS,
        output_dir=OUTPUT_IMAGES,
        interval_seconds=8,
        auto_label=True,
        conf=0.4,
    )

    print()
    print("Siguiente paso:")
    print("  1. Revisa las etiquetas en el labeller:")
    print("     .\\venv_cuda\\Scripts\\python.exe ml/labeller/labeller_app.py")
    print("  2. Cuando esten revisadas, añade los frames al dataset:")
    print("     .\\venv_cuda\\Scripts\\python.exe ml/training/build_hybrid_dataset.py")
    print("  3. Reentrena con imgsz=1280:")
    print("     .\\venv_cuda\\Scripts\\python.exe ml/training/train_unified.py --target players")
