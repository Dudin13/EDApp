"""
download_pretrained.py — Descarga modelo pre-entrenado de deteccion de futbol.

Opciones:
  1. Roboflow: exporta el modelo ya conectado como YOLOv8 .pt
  2. GitHub: descarga weights del repo Issam Jebnouni (yolov8s entrenado en futbol)

El modelo se guarda en:
    train_yolo/runs/detect/train/weights/best.pt
(misma ruta que usaria el entrenamiento propio, compatible con la app)
"""

import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUTPUT_PATH = ROOT / "train_yolo" / "runs" / "detect" / "train" / "weights" / "best.pt"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def download_from_roboflow():
    """
    Descarga el dataset de Roboflow en formato YOLOv8 y usa los pesos
    del modelo de la API directamente.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / "football_analyzer" / ".env")
    except Exception:
        pass

    api_key = os.getenv("ROBOFLOW_API_KEY", "")
    if not api_key:
        print("ERROR: ROBOFLOW_API_KEY no encontrada en .env")
        return False

    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
        version = project.version(1)

        print("Descargando dataset de Roboflow en formato YOLOv8...")
        dataset = version.download("yolov8", location=str(ROOT / "roboflow_dataset"))
        print(f"Dataset descargado en: {dataset.location}")
        return dataset.location
    except Exception as e:
        print(f"Error con Roboflow: {e}")
        return False


def finetune_with_roboflow_dataset(dataset_location: str):
    """
    Fine-tune YOLOv8 usando el dataset de Roboflow (mucho mas grande y variado)
    combinado con nuestras imagenes locales.
    Usa yolov8s.pt (small) como base, mejor que nano para este uso.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: pip install ultralytics")
        return

    dataset_yaml = str(Path(dataset_location) / "data.yaml")
    if not Path(dataset_yaml).exists():
        # Buscar data.yaml en subcarpetas
        yamls = list(Path(dataset_location).glob("**/*.yaml"))
        if yamls:
            dataset_yaml = str(yamls[0])
        else:
            print(f"ERROR: No se encontro data.yaml en {dataset_location}")
            return

    print(f"\nUsando dataset: {dataset_yaml}")
    print("Iniciando fine-tuning con YOLOv8s (small)...")

    model = YOLO("yolov8s.pt")   # small: mejor precision que nano
    results = model.train(
        data=dataset_yaml,
        epochs=50,
        imgsz=640,
        batch=8,
        workers=2,
        project=str(ROOT / "train_yolo" / "runs" / "detect"),
        name="train",
        exist_ok=True,
        patience=15,
        save=True,
        verbose=True,
        # Augmentacion moderada (el dataset de Roboflow ya es variado)
        augment=True,
        fliplr=0.5,
        mosaic=0.8,
        hsv_s=0.5,
        hsv_v=0.3,
    )
    print(f"\nModelo guardado en: {OUTPUT_PATH}")
    return results


def download_github_model():
    """
    Alternativa: descarga best.pt preentrenado del repo de GitHub
    (Player-Tracking de prashant290605, usando gdown)
    """
    try:
        import gdown
    except ImportError:
        print("Instalando gdown...")
        os.system(f"{sys.executable} -m pip install gdown -q")
        import gdown

    # ID del fichero best.pt del repo Player-Tracking (Google Drive)
    # https://github.com/prashant290605/Player-Tracking
    file_id = "1HR8Q5_0Q7e7XYvM5SRjWBi8-IWsz0aJN"
    url = f"https://drive.google.com/uc?id={file_id}"

    print(f"Descargando model pre-entrenado desde Google Drive...")
    print(f"Destino: {OUTPUT_PATH}")

    try:
        gdown.download(url, str(OUTPUT_PATH), quiet=False)
        if OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size > 1024 * 1024:
            print(f"Descarga exitosa: {OUTPUT_PATH.stat().st_size / 1024 / 1024:.1f} MB")
            return True
        else:
            print("La descarga no produjo un fichero valido")
            return False
    except Exception as e:
        print(f"Error en descarga: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("  ED Analytics - Descarga Modelo Pre-entrenado")
    print("=" * 60)

    # Estrategia 1: Dataset Roboflow + fine-tuning (RECOMENDADO)
    print("\n[Opcion 1] Descargando dataset de Roboflow para fine-tuning...")
    dataset_loc = download_from_roboflow()

    if dataset_loc:
        print(f"\nDataset descargado. Iniciando fine-tuning...")
        finetune_with_roboflow_dataset(dataset_loc)
    else:
        # Estrategia 2: Modelo pre-entrenado de GitHub
        print("\n[Opcion 2] Descargando modelo pre-entrenado de GitHub...")
        success = download_github_model()

        if success:
            # Verificar que funciona
            try:
                from ultralytics import YOLO
                model = YOLO(str(OUTPUT_PATH))
                print(f"\nModelo cargado correctamente")
                print(f"Clases: {model.names}")
            except Exception as e:
                print(f"Error al cargar el modelo: {e}")
        else:
            # Estrategia 3: Fine-tune con nuestro propio dataset corregido
            print("\n[Opcion 3] Entrenando con dataset local corregido...")
            from ultralytics import YOLO
            model = YOLO("yolov8s.pt")
            model.train(
                data=str(ROOT / "train_yolo" / "data.yaml"),
                epochs=100,
                imgsz=640,
                batch=8,
                project=str(ROOT / "train_yolo" / "runs" / "detect"),
                name="train",
                exist_ok=True,
            )
