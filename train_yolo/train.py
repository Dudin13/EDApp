"""
train.py — Entrenamiento YOLOv8 con las imágenes anotadas de partidos de fútbol.

Dataset generado con Grounding DINO (auto_label.py):
  - 50 frames extraídos de partidos (frames_etiquetado/)
  - 50 etiquetas en formato YOLO (labels_yolo/)
  - Clases: 0=player, 1=goalkeeper, 2=referee, 3=ball

Uso:
    python train_yolo/train.py

El mejor modelo se guarda en:
    train_yolo/runs/detect/train/weights/best.pt
"""

import os
import sys
import shutil
from pathlib import Path

# ── Rutas ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent          # C:/apped
TRAIN_DIR = Path(__file__).parent           # C:/apped/train_yolo
IMAGES_DIR = ROOT / "frames_etiquetado"
LABELS_DIR = ROOT / "labels_yolo"
DATA_YAML = TRAIN_DIR / "data.yaml"
OUTPUT_DIR = TRAIN_DIR / "runs"

# ── Verificación ───────────────────────────────────────────────────────────
def check_dataset():
    images = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png"))
    labels = list(LABELS_DIR.glob("*.txt"))
    print(f"✓ Imágenes encontradas: {len(images)}")
    print(f"✓ Etiquetas encontradas: {len(labels)}")

    if len(images) == 0:
        print("❌ No hay imágenes. Ejecuta primero extraer_frames.py")
        sys.exit(1)
    if len(labels) == 0:
        print("❌ No hay etiquetas. Ejecuta primero auto_label.py")
        sys.exit(1)

    # Verificar que cada imagen tiene su etiqueta
    img_stems = {p.stem for p in images}
    lab_stems = {p.stem for p in labels}
    sin_etiqueta = img_stems - lab_stems
    if sin_etiqueta:
        print(f"⚠️  {len(sin_etiqueta)} imágenes sin etiqueta: {list(sin_etiqueta)[:5]}")

    return len(images), len(labels)


def prepare_data_yaml():
    """Crea/actualiza el data.yaml con las rutas absolutas correctas."""
    content = f"""# YOLOv8 Dataset Config — Fútbol ED Analytics
path: {ROOT.as_posix()}
train: frames_etiquetado
val: frames_etiquetado

nc: 4
names:
  0: player
  1: goalkeeper
  2: referee
  3: ball
"""
    DATA_YAML.write_text(content, encoding="utf-8")
    print(f"✓ data.yaml escrito en {DATA_YAML}")


def train(
    model_size: str = "n",          # n=nano (rápido), s=small, m=medium
    epochs: int = 80,
    imgsz: int = 640,
    batch: int = 8,
    workers: int = 2,
):
    """
    Entrena YOLOv8 con augmentación agresiva (idóneo para dataset pequeño).
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics no instalado. Ejecuta: pip install ultralytics")
        sys.exit(1)

    print(f"\n🚀 Iniciando entrenamiento YOLOv8{model_size}")
    print(f"   epochs={epochs} | imgsz={imgsz} | batch={batch}")
    print(f"   data={DATA_YAML}\n")

    model = YOLO(f"yolov8{model_size}.pt")  # descarga el modelo base automáticamente

    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        project=str(OUTPUT_DIR / "detect"),
        name="train",
        exist_ok=True,

        # Augmentación agresiva (compensa dataset pequeño)
        augment=True,
        hsv_h=0.015,     # variación tono
        hsv_s=0.7,       # variación saturación
        hsv_v=0.4,       # variación brillo
        degrees=5,       # rotación ligera
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,
        flipud=0.05,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,

        # Otras opciones
        patience=20,     # early stopping
        save=True,
        plots=True,
        verbose=True,
    )

    best_model = OUTPUT_DIR / "detect" / "train" / "weights" / "best.pt"
    if best_model.exists():
        size_mb = best_model.stat().st_size / 1024 / 1024
        print(f"\n✅ Entrenamiento completado!")
        print(f"   Mejor modelo: {best_model} ({size_mb:.1f} MB)")
        print(f"   Para usarlo en la app: Settings → Motor: 'yolo (local)'")
    else:
        print("\n⚠️ No se encontró best.pt. Revisa los logs de entrenamiento.")

    return results


def validate(model_path: str = None):
    """Valida el modelo entrenado."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics no instalado")
        return

    if model_path is None:
        model_path = str(OUTPUT_DIR / "detect" / "train" / "weights" / "best.pt")

    if not Path(model_path).exists():
        print(f"❌ Modelo no encontrado: {model_path}")
        return

    print(f"\n🔍 Validando modelo: {model_path}")
    model = YOLO(model_path)
    metrics = model.val(data=str(DATA_YAML), verbose=True)
    print(f"\n📊 mAP50: {metrics.box.map50:.3f}")
    print(f"   mAP50-95: {metrics.box.map:.3f}")
    return metrics


if __name__ == "__main__":
    print("=" * 60)
    print("  ED Analytics — Entrenamiento YOLOv8")
    print("=" * 60)

    n_imgs, n_labs = check_dataset()
    prepare_data_yaml()

    if n_imgs < 20:
        print(f"\n⚠️ Solo {n_imgs} imágenes. Resultado puede ser pobre.")
        print("   Ejecuta augment_dataset.py para ampliar el dataset.")
        print("   Continuando de todas formas...\n")

    # Ajustar epochs según cantidad de datos
    epochs = min(150, max(50, n_imgs * 2))
    batch = 4 if n_imgs < 30 else 8

    print(f"\n📐 Configuración automática: epochs={epochs}, batch={batch}")

    train(
        model_size="n",   # nano: rápido, suficiente para este dataset
        epochs=epochs,
        imgsz=640,
        batch=batch,
    )

    print("\n🎯 Ejecutando validación final...")
    validate()
