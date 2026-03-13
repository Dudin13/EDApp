"""
create_val_split.py
===================
Crea un split de validación real a partir del hybrid_dataset.

Problema actual: data.yaml tiene val: images/train (mismas imágenes que train).
Esto hace que las métricas de mAP sean falsas — el modelo "valida" sobre datos
que ya ha visto durante el entrenamiento.

Solución: mover aleatoriamente un 15% de las imágenes de train/ a val/,
manteniendo siempre el par imagen+etiqueta juntos.

Uso:
    python ml/training/create_val_split.py
    python ml/training/create_val_split.py --ratio 0.2   # 20% para val
    python ml/training/create_val_split.py --dry-run     # previsualizar sin mover nada

Qué hace:
    1. Lee todas las imágenes de hybrid_dataset/train/images/
    2. Selecciona aleatoriamente el 15% (estratificado por clase si es posible)
    3. Mueve imagen + etiqueta a hybrid_dataset/valid/images/ y valid/labels/
    4. Actualiza data.yaml para que val apunte a valid/images
    5. Genera un resumen con estadísticas del split resultante
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict

# ── Rutas ──────────────────────────────────────────────────────────────────
BASE         = Path(os.environ.get("APPED_ROOT", "C:/apped"))
DATASET_ROOT = BASE / "04_Datasets_Entrenamiento" / "hybrid_dataset"
TRAIN_IMGS   = DATASET_ROOT / "train" / "images"
TRAIN_LABS   = DATASET_ROOT / "train" / "labels"
VAL_IMGS     = DATASET_ROOT / "valid" / "images"
VAL_LABS     = DATASET_ROOT / "valid" / "labels"
DATA_YAML    = DATASET_ROOT / "data.yaml"

# Clases del modelo
CLASS_NAMES = {0: "Goalkeeper", 1: "Player", 2: "ball", 3: "referee"}


def count_classes_in_label(label_path: Path) -> list[int]:
    """Devuelve las clases presentes en un archivo de etiqueta YOLO."""
    classes = []
    if not label_path.exists():
        return classes
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if parts:
            try:
                classes.append(int(float(parts[0])))
            except ValueError:
                pass
    return classes


def get_primary_class(label_path: Path) -> int:
    """Devuelve la clase más frecuente en una etiqueta (para estratificación)."""
    classes = count_classes_in_label(label_path)
    if not classes:
        return -1
    return max(set(classes), key=classes.count)


def create_val_split(val_ratio: float = 0.15, dry_run: bool = False, seed: int = 42):
    """
    Divide train/ en train/ y valid/ con el ratio indicado.
    
    Args:
        val_ratio: Fracción de imágenes a mover a valid/ (0.15 = 15%)
        dry_run:   Si True, solo muestra qué haría sin mover nada
        seed:      Semilla aleatoria para reproducibilidad
    """
    random.seed(seed)

    # ── Validar que existe el dataset ─────────────────────────────────────
    if not TRAIN_IMGS.exists():
        print(f"❌ No se encuentra: {TRAIN_IMGS}")
        print(f"   Asegúrate de haber ejecutado build_hybrid_dataset.py primero.")
        return False

    # ── Cargar todas las imágenes ──────────────────────────────────────────
    exts = {".jpg", ".jpeg", ".png"}
    all_imgs = sorted([f for f in TRAIN_IMGS.iterdir() if f.suffix.lower() in exts])
    total = len(all_imgs)

    if total == 0:
        print(f"❌ No hay imágenes en {TRAIN_IMGS}")
        return False

    print(f"\n{'='*60}")
    print(f"  EDApp — Creación de Split de Validación")
    print(f"{'='*60}")
    print(f"  Dataset:    {DATASET_ROOT}")
    print(f"  Train imgs: {total}")
    print(f"  Val ratio:  {val_ratio:.0%}")
    print(f"  Val target: ~{int(total * val_ratio)} imágenes")
    print(f"  Semilla:    {seed}")
    if dry_run:
        print(f"  MODO:       DRY RUN (no se mueve nada)")
    print(f"{'='*60}\n")

    # ── Estratificar por clase predominante ────────────────────────────────
    # Agrupar imágenes por su clase principal para que el val tenga
    # una distribución similar al train
    by_class = defaultdict(list)
    no_label = []

    for img in all_imgs:
        lab = TRAIN_LABS / (img.stem + ".txt")
        cls = get_primary_class(lab)
        if cls == -1:
            no_label.append(img)
        else:
            by_class[cls].append(img)

    print(f"  Distribución actual en train/:")
    for cls_id, imgs in sorted(by_class.items()):
        name = CLASS_NAMES.get(cls_id, f"clase_{cls_id}")
        print(f"    {name:12s}: {len(imgs):4d} imágenes")
    if no_label:
        print(f"    sin etiqueta: {len(no_label):4d} imágenes (background)")
    print()

    # ── Seleccionar imágenes para val (estratificado) ──────────────────────
    val_imgs = []

    for cls_id, imgs in by_class.items():
        n_val = max(1, int(len(imgs) * val_ratio))
        selected = random.sample(imgs, n_val)
        val_imgs.extend(selected)

    # También mover algunos backgrounds a val si los hay
    if no_label:
        n_bg_val = max(1, int(len(no_label) * val_ratio))
        val_imgs.extend(random.sample(no_label, n_bg_val))

    val_imgs = list(set(val_imgs))  # eliminar duplicados
    train_remaining = total - len(val_imgs)

    print(f"  Split resultante:")
    print(f"    Train: {train_remaining} imágenes ({train_remaining/total:.0%})")
    print(f"    Val:   {len(val_imgs)} imágenes ({len(val_imgs)/total:.0%})")
    print()

    if dry_run:
        print("  [DRY RUN] Imágenes que se moverían a val/:")
        for img in sorted(val_imgs)[:10]:
            print(f"    {img.name}")
        if len(val_imgs) > 10:
            print(f"    ... y {len(val_imgs)-10} más")
        print("\n  Ejecuta sin --dry-run para aplicar los cambios.")
        return True

    # ── Crear carpetas val/ ────────────────────────────────────────────────
    VAL_IMGS.mkdir(parents=True, exist_ok=True)
    VAL_LABS.mkdir(parents=True, exist_ok=True)

    # ── Mover imagen + etiqueta a val/ ────────────────────────────────────
    moved = 0
    missing_labels = 0

    for img_path in val_imgs:
        dst_img = VAL_IMGS / img_path.name
        shutil.move(str(img_path), str(dst_img))

        lab_src = TRAIN_LABS / (img_path.stem + ".txt")
        if lab_src.exists():
            dst_lab = VAL_LABS / lab_src.name
            shutil.move(str(lab_src), str(dst_lab))
        else:
            missing_labels += 1
            # Crear etiqueta vacía (background) en val
            (VAL_LABS / (img_path.stem + ".txt")).write_text("", encoding="utf-8")

        moved += 1
        if moved % 50 == 0:
            print(f"  Movidas: {moved}/{len(val_imgs)}...")

    print(f"\n  ✅ {moved} imágenes movidas a valid/")
    if missing_labels:
        print(f"  ⚠️  {missing_labels} imágenes sin etiqueta → creadas como background")

    # ── Actualizar data.yaml ───────────────────────────────────────────────
    update_data_yaml()

    # ── Resumen final ──────────────────────────────────────────────────────
    final_train = len(list(TRAIN_IMGS.glob("*.jpg"))) + len(list(TRAIN_IMGS.glob("*.png")))
    final_val   = len(list(VAL_IMGS.glob("*.jpg")))  + len(list(VAL_IMGS.glob("*.png")))

    print(f"\n{'='*60}")
    print(f"  SPLIT COMPLETADO")
    print(f"  Train: {final_train} imágenes → {TRAIN_IMGS}")
    print(f"  Val:   {final_val}   imágenes → {VAL_IMGS}")
    print(f"  data.yaml actualizado → {DATA_YAML}")
    print(f"{'='*60}")
    print(f"\n✅ Ahora las métricas de mAP serán reales durante el entrenamiento.")
    print(f"   Lanza el entrenamiento con: python ml/training/train_unified.py")
    return True


def update_data_yaml():
    """Reescribe data.yaml con val: valid/images (en lugar de images/train)."""
    if not DATA_YAML.exists():
        print(f"⚠️  data.yaml no encontrado en {DATA_YAML}, creando uno nuevo...")

    yaml_content = f"""# EDApp — Dataset híbrido fútbol
# Generado por create_val_split.py
# IMPORTANTE: val apunta a valid/ (split real, NO el mismo que train)
path: {DATASET_ROOT.as_posix()}
train: train/images
val:   valid/images

nc: 4
names:
  0: Goalkeeper
  1: Player
  2: ball
  3: referee
"""
    DATA_YAML.write_text(yaml_content, encoding="utf-8")
    print(f"\n  ✅ data.yaml actualizado: val → valid/images")


def check_existing_val():
    """Comprueba si ya existe un split de validación válido."""
    if not VAL_IMGS.exists():
        return 0
    imgs = list(VAL_IMGS.glob("*.jpg")) + list(VAL_IMGS.glob("*.png"))
    return len(imgs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crear split de validación para el hybrid_dataset")
    parser.add_argument("--ratio",   type=float, default=0.15,  help="Fracción de imágenes para val (default: 0.15)")
    parser.add_argument("--seed",    type=int,   default=42,    help="Semilla aleatoria (default: 42)")
    parser.add_argument("--dry-run", action="store_true",       help="Previsualizar sin mover nada")
    args = parser.parse_args()

    # Comprobar si ya existe val/
    existing = check_existing_val()
    if existing > 0 and not args.dry_run:
        print(f"\n⚠️  Ya existe valid/ con {existing} imágenes.")
        resp = input("   ¿Quieres rehacerlo desde cero? (s/N): ").strip().lower()
        if resp != "s":
            print("   Cancelado. El split existente se mantiene.")
            # Actualizar data.yaml igualmente por si apuntaba mal
            update_data_yaml()
            exit(0)
        # Limpiar val/ existente
        shutil.rmtree(VAL_IMGS.parent)
        print("   Val/ anterior eliminado. Regenerando...")

    create_val_split(val_ratio=args.ratio, dry_run=args.dry_run, seed=args.seed)
