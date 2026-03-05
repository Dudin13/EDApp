"""
retrain_ball.py — Reentrenamiento YOLOv8-seg con foco en la deteccion del balon.

Usa el combined_dataset (812 imagenes) con:
- 150 epochs (vs 50 del entrenamiento anterior)
- Batch 16 para aprovechar RTX 2070
- Pesos de clase ajustados para priorizar el balon
- Augmentaciones enfocadas en objetos pequenos
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
TRAIN_DIR = Path(__file__).parent
# Dataset hibrido: 1682 imgs train (1070 poligono + 612 bbox) + 209 valid
# YOLOv8-seg entrena deteccion con TODOS y segmentacion solo con los de poligono
DATA_YAML = ROOT / "hybrid_dataset" / "data.yaml"
OUTPUT_DIR = TRAIN_DIR / "runs"

def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics no instalado.")
        sys.exit(1)

    # Partir del ultimo best.pt del entrenamiento anterior (fine-tuning)
    prev_best = OUTPUT_DIR / "segment" / "train" / "weights" / "best.pt"
    if prev_best.exists():
        base_model = str(prev_best)
        print(f"[INFO] Fine-tuning desde: {base_model}")
    else:
        base_model = "yolov8s-seg.pt"
        print(f"[INFO] Entrenamiento desde cero con: {base_model}")

    print(f"[INFO] Dataset: {DATA_YAML}")
    print(f"[INFO] Epochs: 150 | Batch: 16 | imgsz: 640")
    print("[INFO] Iniciando entrenamiento...\n")

    model = YOLO(base_model)

    results = model.train(
        data=str(DATA_YAML),
        epochs=150,
        imgsz=640,
        batch=16,
        workers=2,
        project=str(OUTPUT_DIR / "segment"),
        name="train",
        exist_ok=True,

        # Augmentaciones agresivas para objetos pequenos (balon)
        augment=True,
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.5,
        degrees=5,
        translate=0.1,
        scale=0.6,
        shear=2.0,
        perspective=0.0005,
        flipud=0.05,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.0,  # desactivado: causa IndexError con mascaras vacias

        # Regularizacion / optimizacion
        patience=30,
        lr0=0.005,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,

        # Guardado
        save=True,
        save_period=10,
        plots=True,
        verbose=True,

        # Segmentacion
        overlap_mask=True,
        mask_ratio=4,
        retina_masks=False,
    )

    best_model = OUTPUT_DIR / "segment" / "train" / "weights" / "best.pt"
    if best_model.exists():
        size_mb = best_model.stat().st_size / 1024 / 1024
        print(f"\n[OK] Entrenamiento completado!")
        print(f"     Mejor modelo: {best_model} ({size_mb:.1f} MB)")
    else:
        print("\n[WARN] No se encontro best.pt.")

    # Validacion final
    print("\n[INFO] Validando modelo final...")
    val_model = YOLO(str(best_model))
    metrics = val_model.val(data=str(DATA_YAML), verbose=False)
    print(f"[METRICS] mAP50    : {metrics.box.map50:.3f}")
    print(f"[METRICS] mAP50-95 : {metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
