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

    # Dataset publico con anotaciones de SEGMENTACION (poligonos)
    # Buscamos un buen dataset de futbol con segmentacion
    # Proyecto de ejemplo encontrado con segmentacion de instancias de futbol
    # usaremos uno de la busqueda: workspace='wisd' project='instance-segmentation-football'
    project = rf.workspace("eduardos-workspace-nfw58").project("instance-segmentation-football-pqine")
    # Idealmente deberiamos usar un projecto especifico de "Instance Segmentation"
    dataset_path = ROOT / "roboflow_dataset"

    # Obtener la ultima version disponible
    versions = project.versions()
    latest_version = versions[-1].version if versions else 1

    print(f"\nDescargando en: {dataset_path} (Version: {latest_version})")
    print("(Puede tardar 1-2 minutos segun la conexion)\n")

    # COMENTADO: No descargamos para evitar sobreescribir los datos ya limpios
    # dataset = project.version(latest_version).download(
    #     model_format="yolov8", # Roboflow usa 'yolov8' incluso para segmentation
    #     location=str(dataset_path),
    #     overwrite=True
    # )

    print(f"\nUsando dataset existente en: {dataset_path}")

    # Contar imagenes descargadas
    yaml_path = dataset_path / "data.yaml"
    imgs_train = list((dataset_path / "train" / "images").glob("*.jpg"))
    imgs_valid = list((dataset_path / "valid" / "images").glob("*.jpg"))

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

    # ── Entrenamiento con dataset grande (SEGMENTACION) ──────────
    print("\n" + "=" * 60)
    print("  Iniciando entrenamiento YOLOv8s-seg con dataset Roboflow")
    print("=" * 60)

    from ultralytics import YOLO

    # Limpiar runs anteriores
    output_dir = ROOT / "train_yolo" / "runs" / "segment" / "train"
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
        print("Cache anterior limpiado")

    # Cambio CRITICO: usarmos el modelo de segmentacion (-seg)
    model = YOLO("yolov8s-seg.pt")  

    print(f"\nDataset: {yaml_path}")
    print(f"Imagenes train: {len(imgs_train)} | valid: {len(imgs_valid)}")
    print(f"Modelo base: YOLOv8s-seg (Segmentacion)\n")

    results = model.train(
        data=str(yaml_path),
        epochs=50,              
        imgsz=640,
        batch=16,               # Optimizado para RTX 2070
        workers=4,              # Optimizado para carga
        project=str(ROOT / "train_yolo" / "runs" / "segment"),
        name="train",
        exist_ok=True,
        patience=10,            
        save=True,
        save_period=10,         
        plots=True,
        verbose=True,
        # Augmentacion moderada 
        augment=True,
        fliplr=0.5,
        mosaic=0.8,
        hsv_s=0.4,
        hsv_v=0.3,
        degrees=5,
    )

    best = ROOT / "train_yolo" / "runs" / "segment" / "train" / "weights" / "best.pt"
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
