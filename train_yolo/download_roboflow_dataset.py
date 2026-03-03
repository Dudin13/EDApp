"""
download_roboflow_dataset.py — Descarga el dataset publico de futbol de Roboflow Universe.

Dataset: football-players-detection (roboflow-jvuqo)
  - +1000 imagenes etiquetadas de partidos de futbol
  - Clases: player, goalkeeper, referee, ball
  - Formato: YOLOv8

Tras la descarga lanza el entrenamiento con YOLOv8s (small, mejor precision que nano).
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def main():
    # Cargar API key
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / "football_analyzer" / ".env")
    except Exception:
        pass

    api_key = os.getenv("ROBOFLOW_API_KEY", "")
    if not api_key:
        print("ERROR: ROBOFLOW_API_KEY no encontrada en football_analyzer/.env")
        sys.exit(1)

    print("=" * 60)
    print("  Descargando dataset Roboflow football-players-detection")
    print("=" * 60)

    # ── Descarga del dataset ──────────────────────────────────────
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)

    # Dataset publico con +1000 imagenes
    project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
    dataset_path = ROOT / "roboflow_dataset"

    print(f"\nDescargando en: {dataset_path}")
    print("(Puede tardar 1-2 minutos segun la conexion)\n")

    dataset = project.version(1).download(
        model_format="yolov8",
        location=str(dataset_path),
        overwrite=True
    )

    print(f"\nDataset descargado en: {dataset.location}")

    # Contar imagenes descargadas
    yaml_path = Path(dataset.location) / "data.yaml"
    imgs_train = list((Path(dataset.location) / "train" / "images").glob("*.jpg"))
    imgs_valid = list((Path(dataset.location) / "valid" / "images").glob("*.jpg"))

    print(f"  Imagenes train: {len(imgs_train)}")
    print(f"  Imagenes valid: {len(imgs_valid)}")
    print(f"  data.yaml: {yaml_path}")

    if not yaml_path.exists():
        # Buscar yaml en subcarpetas
        yamls = list(Path(dataset.location).glob("**/*.yaml"))
        yaml_path = yamls[0] if yamls else None
        if not yaml_path:
            print("ERROR: No se encontro data.yaml")
            sys.exit(1)

    # ── Entrenamiento con dataset grande ─────────────────────────
    print("\n" + "=" * 60)
    print("  Iniciando entrenamiento YOLOv8s con dataset Roboflow")
    print("=" * 60)

    from ultralytics import YOLO

    # Limpiar runs anteriores
    output_dir = ROOT / "train_yolo" / "runs" / "detect" / "train"
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
        print("Cache anterior limpiado")

    model = YOLO("yolov8s.pt")  # small: mejor balance precision/velocidad

    print(f"\nDataset: {yaml_path}")
    print(f"Imagenes train: {len(imgs_train)} | valid: {len(imgs_valid)}")
    print(f"Modelo base: YOLOv8s (transfer learning desde COCO)\n")

    results = model.train(
        data=str(yaml_path),
        epochs=50,              # Con 1000+ imgs, 50 epochs suele ser suficiente
        imgsz=640,
        batch=8,
        workers=2,
        project=str(ROOT / "train_yolo" / "runs" / "detect"),
        name="train",
        exist_ok=True,
        patience=10,            # Early stopping agresivo
        save=True,
        save_period=10,         # Guardar cada 10 epochs
        plots=True,
        verbose=True,
        # Augmentacion moderada (el dataset ya es variado)
        augment=True,
        fliplr=0.5,
        mosaic=0.8,
        hsv_s=0.4,
        hsv_v=0.3,
        degrees=5,
    )

    best = ROOT / "train_yolo" / "runs" / "detect" / "train" / "weights" / "best.pt"
    if best.exists():
        size_mb = best.stat().st_size / 1024 / 1024
        print(f"\n{'='*60}")
        print(f"  ENTRENAMIENTO COMPLETADO")
        print(f"  Modelo: {best} ({size_mb:.1f} MB)")
        print(f"  En la app: Settings -> Motor: 'yolo (local)'")
        print(f"{'='*60}")
    else:
        print("\nERROR: No se genero best.pt")


if __name__ == "__main__":
    main()
