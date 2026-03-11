from ultralytics import YOLO
from pathlib import Path
import os
import torch

# ✅ Paths via variable de entorno para portabilidad
BASE = Path(os.environ.get("APPED_ROOT", "C:/apped"))
DATA_YAML = BASE / "04_Datasets_Entrenamiento/super_focused_50/data.yaml"
OUTPUT_DIR = BASE / "train_yolo/runs/detect/SuperFocused_v1"


def train():
    # ✅ Validar que el yaml existe antes de empezar
    assert DATA_YAML.exists(), f"❌ data.yaml no encontrado: {DATA_YAML}"

    # ✅ Info de GPU antes de entrenar
    if torch.cuda.is_available():
        print(f"🖥️  CUDA disponible - GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  CUDA no disponible, entrenando en CPU (será lento)")

    print(f"📁 Dataset: {DATA_YAML}")
    print("🚀 Iniciando entrenamiento SUPER FOCUSED (50 imágenes)...")

    model = YOLO("yolo11m-seg.pt")

    results = model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=-1,           # ✅ Auto-detect batch size según VRAM disponible
        workers=0,
        project=str(OUTPUT_DIR.parent),
        name=OUTPUT_DIR.name,
        exist_ok=True,
        optimizer='AdamW',
        augment=True,
        patience=20,
        save=True,
        plots=True,
        verbose=True,
    )

    # ✅ Mostrar métricas finales
    print("\n✅ Entrenamiento completado.")
    map50 = results.results_dict.get('metrics/mAP50(B)', 'N/A')
    map50_95 = results.results_dict.get('metrics/mAP50-95(B)', 'N/A')
    print(f"📊 mAP50:    {map50}")
    print(f"📊 mAP50-95: {map50_95}")
    print(f"💾 Modelo guardado en: {OUTPUT_DIR / 'weights/best.pt'}")


if __name__ == "__main__":
    train()
