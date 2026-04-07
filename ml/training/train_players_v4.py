"""
train_players_v4.py — Entrenamiento limpio de detect_players_v4_clean.pt

Mejoras respecto a V3:
  [OK] Dataset solo con bbox labels correctos (no polígonos)
  [OK] Clases correctamente remapeadas
  [OK] Imgsz=960 para detectar jugadores pequeños en campo abierto
  [OK] Batch optimizado para RTX 2070 Super (8GB VRAM)
  [OK] Modelo final copiado automáticamente a assets/weights/
  [OK] Nombre claro: detect_players_v4_clean.pt

Uso:
    python ml/training/train_players_v4.py
    python ml/training/train_players_v4.py --epochs 100 --batch 12
"""

import sys
import shutil
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASET_YAML  = PROJECT_ROOT / "data" / "datasets" / "players_v4_dataset" / "data.yaml"
RUNS_DIR      = PROJECT_ROOT / "ml" / "training" / "runs"
WEIGHTS_DIR   = PROJECT_ROOT / "assets" / "weights"
OUTPUT_NAME   = "detect_players_v4_clean"   # Nombre final del modelo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int,   default=60)
    parser.add_argument("--batch",  type=int,   default=8)
    parser.add_argument("--imgsz",  type=int,   default=960)
    parser.add_argument("--model",  type=str,   default="yolo11m.pt",
                        help="Modelo base: yolo11m.pt, yolov8m.pt, etc.")
    args = parser.parse_args()

    # Verificar que el dataset existe
    if not DATASET_YAML.exists():
        print("[ERROR] Dataset no preparado. Ejecuta primero:")
        print("        python ml/training/prepare_dataset.py")
        sys.exit(1)

    from ultralytics import YOLO

    print("=" * 60)
    print(f"  ENTRENAMIENTO: {OUTPUT_NAME}")
    print("=" * 60)
    print(f"  Dataset:  {DATASET_YAML}")
    print(f"  Modelo:   {args.model}")
    print(f"  Épocas:   {args.epochs}")
    print(f"  Batch:    {args.batch}")
    print(f"  Imgsz:    {args.imgsz}")
    print(f"  Salida:   {WEIGHTS_DIR / (OUTPUT_NAME + '.pt')}")
    print("=" * 60)

    # Verificar GPU
    import torch
    device = "0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {gpu} ({vram:.1f} GB VRAM)")
    else:
        print("  [WARN]  Sin GPU — entrenamiento muy lento en CPU")
    print()

    model = YOLO(args.model)

    results = model.train(
        data        = str(DATASET_YAML),
        epochs      = args.epochs,
        imgsz       = args.imgsz,
        batch       = args.batch,
        device      = device,
        project     = str(RUNS_DIR / "detect"),
        name        = OUTPUT_NAME,
        exist_ok    = True,
        # Augmentación controlada para fútbol (cámara fija, perspectiva consistente)
        degrees     = 5.0,      # Rotación mínima (cámara casi fija)
        fliplr      = 0.5,      # Flip horizontal (simétrico)
        flipud      = 0.0,      # Sin flip vertical (el campo siempre está abajo)
        mosaic      = 1.0,
        mixup       = 0.1,
        hsv_h       = 0.015,
        hsv_s       = 0.7,
        hsv_v       = 0.4,
        # Evitar box loss mal optimizado
        box         = 7.5,
        cls         = 0.5,
        dfl         = 1.5,
        patience    = 20,       # Early stopping si no mejora en 20 épocas
        save_period = 10,       # Guardar checkpoint cada 10 épocas
        plots       = True,
        verbose     = True,
    )

    # ─── Copiar el mejor modelo a assets/weights/ ────────────────────────────
    run_dir  = RUNS_DIR / "detect" / OUTPUT_NAME
    best_pt  = run_dir / "weights" / "best.pt"
    dest_pt  = WEIGHTS_DIR / f"{OUTPUT_NAME}.pt"

    if best_pt.exists():
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, dest_pt)
        print()
        print("=" * 60)
        print(f"  [OK] ENTRENAMIENTO COMPLETADO")
        print(f"  Modelo guardado en: {dest_pt}")
        print(f"  mAP50 final: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
        print("=" * 60)
        print()
        print("  Actualiza settings.py con:")
        print(f"    PLAYER_MODEL_NAME = '{OUTPUT_NAME}.pt'")
        print()
        _update_settings(dest_pt.name)
    else:
        print(f"[ERROR] No se encontró best.pt en {run_dir}")


def _update_settings(new_model_name: str):
    """Actualiza PLAYER_MODEL_NAME en settings.py automáticamente."""
    settings_path = PROJECT_ROOT / "core" / "config" / "settings.py"
    if not settings_path.exists():
        return
    content = settings_path.read_text()
    import re
    new_content = re.sub(
        r'PLAYER_MODEL_NAME\s*:\s*str\s*=\s*"[^"]+"',
        f'PLAYER_MODEL_NAME: str = "{new_model_name}"',
        content
    )
    if new_content != content:
        settings_path.write_text(new_content)
        print(f"  [OK] settings.py actualizado → PLAYER_MODEL_NAME = '{new_model_name}'")
    else:
        print(f"  [WARN]  No se pudo actualizar settings.py automáticamente")


if __name__ == "__main__":
    main()
