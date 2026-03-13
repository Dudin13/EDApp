"""
remap_classes.py
================
Remapea las etiquetas del hybrid_dataset al nuevo esquema de 4 clases:

  ANTES                        DESPUES
  clase 0: player (tipo A)  -> 0: player
  clase 1: player (tipo B)  -> 0: player      (fusionar con 0)
  clase 2: (no existe)
  clase 3: referee          -> 2: referee
  clase 4: goalkeeper       -> 1: goalkeeper
  clase 5: (ignorar, 2 inst)

  clase 3: ball             -> se anade despues con dataset externo

Uso:
    python ml/training/remap_classes.py
    python ml/training/remap_classes.py --dry-run   # previsualizar sin cambiar nada
"""

import os
import shutil
import argparse
from pathlib import Path
from collections import Counter

BASE         = Path(os.environ.get("APPED_ROOT", "C:/apped"))
DATASET_ROOT = BASE / "data" / "datasets" / "hybrid_dataset"

# Mapeo: clase_antigua -> clase_nueva  (None = eliminar esa instancia)
REMAP = {
    0: 0,     # player tipo A    -> player
    1: 0,     # player tipo B    -> player (fusionar)
    2: None,  # no existe
    3: 2,     # referee          -> referee
    4: 1,     # goalkeeper       -> goalkeeper
    5: None,  # ignorar (2 inst)
}

CLASS_NAMES_NEW = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}


def remap_label_file(label_path: Path, dry_run: bool) -> dict:
    """Remapea un archivo de etiquetas. Devuelve estadisticas."""
    stats = Counter()
    lines_out = []

    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if not parts or len(parts) < 5:
            continue
        old_cls = int(float(parts[0]))
        new_cls = REMAP.get(old_cls)
        if new_cls is None:
            stats["dropped"] += 1
            continue
        parts[0] = str(new_cls)
        lines_out.append(" ".join(parts))
        stats[f"cls_{new_cls}"] += 1

    if not dry_run:
        label_path.write_text("\n".join(lines_out) + ("\n" if lines_out else ""),
                               encoding="utf-8")
    return stats


def remap_split(split: str, dry_run: bool) -> Counter:
    labels_dir = DATASET_ROOT / split / "labels"
    if not labels_dir.exists():
        print(f"  [{split}] No existe: {labels_dir}")
        return Counter()

    files   = list(labels_dir.glob("*.txt"))
    total   = Counter()
    for i, lf in enumerate(files):
        s = remap_label_file(lf, dry_run)
        total.update(s)
        if (i + 1) % 100 == 0:
            print(f"  [{split}] Procesados {i+1}/{len(files)}...")

    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remapea clases del hybrid_dataset")
    parser.add_argument("--dry-run", action="store_true", help="Previsualizar sin modificar")
    args = parser.parse_args()

    print("\n" + "="*55)
    print("  EDApp - Remapeo de clases del dataset")
    print("="*55)
    print(f"\n  Dataset: {DATASET_ROOT}")
    print(f"  Modo:    {'DRY RUN (sin cambios)' if args.dry_run else 'APLICAR CAMBIOS'}")
    print("\n  Mapeo:")
    for old, new in REMAP.items():
        arrow = f"-> {CLASS_NAMES_NEW[new]} ({new})" if new is not None else "-> ELIMINAR"
        print(f"    clase {old} {arrow}")

    if not args.dry_run:
        # Backup antes de modificar
        backup = DATASET_ROOT.parent / (DATASET_ROOT.name + "_backup_labels")
        if not backup.exists():
            print(f"\n  Creando backup en {backup.name}...")
            for split in ("train", "valid"):
                src = DATASET_ROOT / split / "labels"
                dst = backup / split / "labels"
                if src.exists():
                    shutil.copytree(src, dst)
            print("  Backup OK")

    total_train = remap_split("train", args.dry_run)
    total_val   = remap_split("valid", args.dry_run)

    print(f"\n  Resultado train:")
    for cls_id, name in CLASS_NAMES_NEW.items():
        n = total_train.get(f"cls_{cls_id}", 0)
        print(f"    {name:12s}: {n:,}")
    print(f"    eliminadas:   {total_train.get('dropped', 0):,}")

    print(f"\n  Resultado val:")
    for cls_id, name in CLASS_NAMES_NEW.items():
        n = total_val.get(f"cls_{cls_id}", 0)
        print(f"    {name:12s}: {n:,}")
    print(f"    eliminadas:   {total_val.get('dropped', 0):,}")

    if args.dry_run:
        print("\n  Ejecuta sin --dry-run para aplicar los cambios.")
    else:
        # Actualizar data.yaml
        data_yaml = DATASET_ROOT / "data.yaml"
        yaml_content = f"""# EDApp - Dataset deteccion de futbol
# Capa 1: deteccion de objetos (jugador, portero, arbitro, balon)
# La separacion de equipos la hace SigLIP en capa 2
path: {DATASET_ROOT.as_posix()}
train: train/images
val:   valid/images

nc: 4
names:
  0: player
  1: goalkeeper
  2: referee
  3: ball
"""
        data_yaml.write_text(yaml_content, encoding="utf-8")
        print(f"\n  data.yaml actualizado: {data_yaml}")
        print("\n  Siguiente paso:")
        print("    python ml/training/add_ball_dataset.py   (añadir dataset de balon)")
        print("    python ml/training/train_unified.py --target players")

    print("="*55 + "\n")
