"""
train_unified.py
================
Script único de entrenamiento que reemplaza train.py, retrain_ball.py
y train_super_focused.py.

Modos disponibles:
  players   → Detectar jugadores, porteros y árbitros (yolo11m-seg)
  ball      → Detectar balón (yolo11m, conf baja, imgsz 1280)
  focused   → Fine-tuning rápido con las 50 imágenes manuales
  all       → Entrenar los tres en secuencia

Uso:
    python ml/training/train_unified.py --target players
    python ml/training/train_unified.py --target ball
    python ml/training/train_unified.py --target focused
    python ml/training/train_unified.py --target all
    python ml/training/train_unified.py --target players --epochs 100 --batch 4
    python ml/training/train_unified.py --target players --resume   # continuar entrenamiento

IMPORTANTE: Ejecuta create_val_split.py antes de entrenar por primera vez.
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# ── Rutas base ─────────────────────────────────────────────────────────────
BASE         = Path(os.environ.get("APPED_ROOT", "C:/apped"))
DATASET_ROOT = BASE / "04_Datasets_Entrenamiento" / "hybrid_dataset"
SUPER_ROOT   = BASE / "04_Datasets_Entrenamiento" / "super_focused_50"
OUTPUT_DIR   = BASE / "train_yolo" / "runs"
WEIGHTS_DIR  = BASE / "EDApp" / "assets" / "weights"

# ── Configuración por modo ─────────────────────────────────────────────────
CONFIGS = {
    "players": {
        "description": "Detección de jugadores, porteros y árbitros",
        "model_base":  "yolo11m-seg.pt",          # modelo seg para siluetas
        "data_yaml":   DATASET_ROOT / "data.yaml",
        "output_name": "players_v1",
        "epochs":      150,
        "imgsz":       640,
        "batch":       -1,                         # auto según VRAM
        "conf_thresh": 0.35,
        "augment": {
            "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
            "degrees": 5, "translate": 0.1, "scale": 0.5,
            "shear": 2.0, "perspective": 0.0005,
            "flipud": 0.05, "fliplr": 0.5,
            "mosaic": 1.0, "mixup": 0.0,           # mixup OFF (crashea con máscaras vacías)
            "copy_paste": 0.0,                      # copy_paste OFF (ídem)
            "auto_augment": "",                     # OFF (activa SemanticSeg que crashea)
        },
    },
    "ball": {
        "description": "Detección especializada del balón",
        "model_base":  "yolo11m.pt",               # detect puro (más rápido que seg para el balón)
        "data_yaml":   DATASET_ROOT / "data.yaml",
        "output_name": "ball_v1",
        "epochs":      150,
        "imgsz":       1280,                        # resolución alta — el balón es tiny
        "batch":       4,                           # batch pequeño por el imgsz 1280
        "conf_thresh": 0.10,                        # conf baja — el balón es difícil
        "augment": {
            "hsv_h": 0.02, "hsv_s": 0.7, "hsv_v": 0.5,
            "degrees": 5, "translate": 0.1, "scale": 0.6,
            "shear": 2.0, "perspective": 0.0005,
            "flipud": 0.05, "fliplr": 0.5,
            "mosaic": 1.0, "mixup": 0.0,
            "copy_paste": 0.0,
            "auto_augment": "",
        },
    },
    "focused": {
        "description": "Fine-tuning rápido con las 50 imágenes manuales",
        "model_base":  None,                        # usa el best.pt de players si existe
        "data_yaml":   SUPER_ROOT / "data.yaml",
        "output_name": "focused_v1",
        "epochs":      100,
        "imgsz":       640,
        "batch":       -1,
        "conf_thresh": 0.25,
        "augment": {
            "hsv_h": 0.015, "hsv_s": 0.6, "hsv_v": 0.4,
            "degrees": 3, "translate": 0.1, "scale": 0.4,
            "shear": 1.0, "perspective": 0.0003,
            "flipud": 0.0, "fliplr": 0.5,
            "mosaic": 0.8, "mixup": 0.0,
            "copy_paste": 0.0,
            "auto_augment": "",
        },
    },
}


def print_gpu_info():
    """Muestra info de GPU antes de entrenar."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU:  {name}")
        print(f"  VRAM: {vram:.1f} GB")
    else:
        print("  ⚠️  CUDA no disponible — entrenando en CPU (será muy lento)")


def find_previous_best(target: str) -> Path | None:
    """Busca el mejor modelo de un entrenamiento anterior para hacer fine-tuning."""
    run_dir = OUTPUT_DIR / "detect" / CONFIGS[target]["output_name"] / "weights" / "best.pt"
    if run_dir.exists():
        return run_dir
    # Buscar en segment también
    run_dir2 = OUTPUT_DIR / "segment" / CONFIGS[target]["output_name"] / "weights" / "best.pt"
    if run_dir2.exists():
        return run_dir2
    return None


def find_players_best() -> Path | None:
    """Busca el best.pt de players para usarlo como base en focused."""
    p = OUTPUT_DIR / "segment" / CONFIGS["players"]["output_name"] / "weights" / "best.pt"
    return p if p.exists() else None


def validate_data_yaml(yaml_path: Path) -> bool:
    """Comprueba que el data.yaml tiene val separado de train."""
    if not yaml_path.exists():
        print(f"  ❌ data.yaml no encontrado: {yaml_path}")
        print(f"     Ejecuta build_hybrid_dataset.py y create_val_split.py primero.")
        return False

    content = yaml_path.read_text(encoding="utf-8")

    # Detectar el bug clásico: val apuntando a train
    if "val: train" in content or "val: images/train" in content:
        print(f"  ❌ data.yaml tiene val == train → las métricas serán falsas")
        print(f"     Ejecuta: python ml/training/create_val_split.py")
        return False

    print(f"  ✅ data.yaml OK: {yaml_path.name}")
    return True


def train_target(target: str, epochs: int = None, batch: int = None, resume: bool = False):
    """
    Entrena un modelo para el target indicado.

    Args:
        target:  "players", "ball" o "focused"
        epochs:  Sobreescribe el valor por defecto del modo
        batch:   Sobreescribe el batch por defecto
        resume:  Si True, continúa desde el último checkpoint
    """
    cfg = CONFIGS[target]

    print(f"\n{'='*60}")
    print(f"  MODO: {target.upper()} — {cfg['description']}")
    print(f"{'='*60}")
    print_gpu_info()

    # Validar data.yaml
    if not validate_data_yaml(cfg["data_yaml"]):
        return False

    # Determinar modelo base
    model_base = cfg["model_base"]

    if target == "focused":
        # Para focused: preferir el best.pt de players como punto de partida
        players_best = find_players_best()
        if players_best:
            model_base = str(players_best)
            print(f"  Fine-tuning desde: {players_best.name}")
        else:
            model_base = "yolo11m-seg.pt"
            print(f"  ⚠️  No se encontró best.pt de players — usando modelo base")
    elif resume:
        # Resume: buscar último checkpoint
        last_ckpt = OUTPUT_DIR / "segment" / cfg["output_name"] / "weights" / "last.pt"
        if last_ckpt.exists():
            model_base = str(last_ckpt)
            print(f"  Resumiendo desde: {last_ckpt}")
        else:
            print(f"  ⚠️  No se encontró last.pt para resume — empezando desde cero")

    # Parámetros finales
    _epochs = epochs if epochs is not None else cfg["epochs"]
    _batch  = batch  if batch  is not None else cfg["batch"]
    _imgsz  = cfg["imgsz"]
    _name   = cfg["output_name"]

    # Tipo de output: seg o detect
    is_seg = "seg" in str(model_base)
    project_type = "segment" if is_seg else "detect"

    print(f"\n  Configuración:")
    print(f"    Modelo base: {model_base}")
    print(f"    Dataset:     {cfg['data_yaml']}")
    print(f"    Epochs:      {_epochs}")
    print(f"    Batch:       {'auto' if _batch == -1 else _batch}")
    print(f"    imgsz:       {_imgsz}")
    print(f"    Output:      {OUTPUT_DIR / project_type / _name}")
    print()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ❌ ultralytics no instalado: pip install ultralytics")
        return False

    model = YOLO(model_base)

    results = model.train(
        data=str(cfg["data_yaml"]),
        epochs=_epochs,
        imgsz=_imgsz,
        batch=_batch,
        workers=0,                          # 0 en Windows evita bloqueos
        optimizer="AdamW",
        project=str(OUTPUT_DIR / project_type),
        name=_name,
        exist_ok=True,
        patience=30,                        # early stopping
        save=True,
        save_period=10,                     # guardar checkpoint cada 10 epochs
        plots=True,
        verbose=True,

        # Parámetros de segmentación (solo aplican en modo seg)
        overlap_mask=True,
        mask_ratio=4,
        retina_masks=False,

        # Augmentación específica del modo
        **cfg["augment"],
    )

    # ── Resultado ──────────────────────────────────────────────────────────
    best = OUTPUT_DIR / project_type / _name / "weights" / "best.pt"
    if best.exists():
        size_mb = best.stat().st_size / 1024 / 1024
        map50    = results.results_dict.get("metrics/mAP50(B)",    "N/A")
        map5095  = results.results_dict.get("metrics/mAP50-95(B)", "N/A")

        print(f"\n{'='*60}")
        print(f"  ✅ ENTRENAMIENTO COMPLETADO — {target.upper()}")
        print(f"  Modelo: {best} ({size_mb:.1f} MB)")
        if isinstance(map50, float):
            print(f"  mAP50:    {map50:.3f}")
            print(f"  mAP50-95: {map5095:.3f}")
            if map50 < 0.3:
                print(f"  ⚠️  mAP50 bajo — considera añadir más imágenes al dataset")
            elif map50 >= 0.7:
                print(f"  🎉 Excelente rendimiento")
        print(f"{'='*60}")

        # Copiar best.pt a assets/weights/ para que la app lo use
        dest_name = f"detect_{target}.pt"
        dest = WEIGHTS_DIR / dest_name
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(str(best), str(dest))
        print(f"\n  📦 Copiado a: {dest}")
        print(f"     La app usará este modelo automáticamente.")
        return True
    else:
        print(f"\n  ❌ No se encontró best.pt en {best.parent}")
        return False


def validate_model(target: str):
    """Valida el modelo entrenado sobre el split de val."""
    cfg = CONFIGS[target]
    is_seg = "seg" in cfg.get("model_base", "")
    project_type = "segment" if is_seg else "detect"
    best = OUTPUT_DIR / project_type / cfg["output_name"] / "weights" / "best.pt"

    if not best.exists():
        print(f"  ⚠️  No hay modelo para validar: {best}")
        return

    print(f"\n  Validando {target}...")
    from ultralytics import YOLO
    model = YOLO(str(best))
    metrics = model.val(data=str(cfg["data_yaml"]), verbose=False)

    print(f"  mAP50:    {metrics.box.map50:.3f}")
    print(f"  mAP50-95: {metrics.box.map:.3f}")

    # Métricas por clase
    if hasattr(metrics.box, "ap_class_index"):
        class_names = {0: "Goalkeeper", 1: "Player", 2: "ball", 3: "referee"}
        print(f"\n  Por clase:")
        for i, cls_idx in enumerate(metrics.box.ap_class_index):
            ap = metrics.box.ap[i] if i < len(metrics.box.ap) else 0
            name = class_names.get(int(cls_idx), f"clase_{cls_idx}")
            bar = "█" * int(ap * 20)
            print(f"    {name:12s}: {ap:.3f}  {bar}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EDApp — Script unificado de entrenamiento YOLO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python ml/training/train_unified.py --target players
  python ml/training/train_unified.py --target ball --epochs 200
  python ml/training/train_unified.py --target focused --batch 2
  python ml/training/train_unified.py --target all
  python ml/training/train_unified.py --target players --resume
        """
    )
    parser.add_argument(
        "--target", required=True,
        choices=["players", "ball", "focused", "all"],
        help="Qué modelo entrenar"
    )
    parser.add_argument("--epochs", type=int,  default=None, help="Sobreescribir nº de epochs")
    parser.add_argument("--batch",  type=int,  default=None, help="Sobreescribir batch size")
    parser.add_argument("--resume", action="store_true",     help="Continuar entrenamiento anterior")
    parser.add_argument("--val-only", action="store_true",   help="Solo validar, no entrenar")
    args = parser.parse_args()

    targets = ["players", "ball", "focused"] if args.target == "all" else [args.target]

    for t in targets:
        if args.val_only:
            validate_model(t)
        else:
            ok = train_target(t, epochs=args.epochs, batch=args.batch, resume=args.resume)
            if not ok and args.target == "all":
                print(f"\n  ❌ Fallo en {t} — abortando el resto")
                sys.exit(1)
