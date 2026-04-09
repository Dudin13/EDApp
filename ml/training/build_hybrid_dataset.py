"""
build_hybrid_dataset.py
=======================
Fusiona todas las fuentes de datos en hybrid_dataset listo para entrenar.

Fuentes soportadas:
  1. hybrid_dataset actual      -> base existente (ya en orden correcto)
  2. veo_frames_raw/             -> frames VEO auto-etiquetados con best.pt a imgsz=1280
  3. local_dataset/ (opcional)  -> frames manuales adicionales

Clases TARGET (data.yaml actual — NO cambiar):
  0 = player
  1 = goalkeeper
  2 = referee
  3 = ball

Uso:
  # Solo anadir frames VEO nuevos al dataset existente
  python ml/training/build_hybrid_dataset.py

  # Reconstruir dataset desde cero (elimina todo y empieza limpio)
  python ml/training/build_hybrid_dataset.py --clean

  # Solo ver cuantos archivos habria sin tocar nada
  python ml/training/build_hybrid_dataset.py --dry-run
"""

import os
import sys
import shutil
import argparse
import random
from pathlib import Path

# ── Rutas ──────────────────────────────────────────────────────────────────
BASE        = Path(os.environ.get("APPED_ROOT", "C:/apped"))
DATASET_DIR = BASE / "data" / "datasets" / "hybrid_dataset"
VEO_DIR     = BASE / "data" / "datasets" / "veo_frames_raw"
VEO_LABELS  = BASE / "data" / "datasets" / "labels_autolabel"
LOCAL_DIR   = BASE / "data" / "datasets" / "local_dataset"
VEO_FINAL_DIR = BASE / "data" / "datasets" / "veo_frames_raw_final"

# Clases TARGET — igual que data.yaml
TARGET_CLASSES = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}

# Split ratio para datos nuevos
VAL_RATIO = 0.15


def count_images(split_dir: Path) -> int:
    if not split_dir.exists():
        return 0
    return len([f for f in split_dir.glob("*")
                if f.suffix.lower() in (".jpg", ".jpeg", ".png")])


def copy_with_label(img_path: Path, label_path: Path,
                    dst_images: Path, dst_labels: Path,
                    prefix: str = "", remap: dict = None, dry_run: bool = False) -> bool:
    """
    Copia imagen + etiqueta al destino.
    Si remap es None, copia la etiqueta tal cual.
    Devuelve True si se copió correctamente.
    """
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    # Nombre destino con prefijo para evitar colisiones
    img_name   = f"{prefix}_{img_path.name}" if prefix else img_path.name
    label_name = Path(img_name).stem + ".txt"

    dst_img   = dst_images / img_name
    dst_label = dst_labels / label_name

    # Evitar sobreescribir si ya existe
    if dst_img.exists():
        return False

    if dry_run:
        return True

    # Copiar imagen
    shutil.copy2(img_path, dst_img)

    # Copiar / remapear etiqueta
    if label_path.exists():
        if remap:
            lines_out = []
            for line in label_path.read_text(encoding="utf-8", errors="replace").strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                old_cls = int(float(parts[0]))
                new_cls = remap.get(old_cls, old_cls)
                lines_out.append(f"{new_cls} " + " ".join(parts[1:]))
            dst_label.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
        else:
            shutil.copy2(label_path, dst_label)
    else:
        # Sin etiqueta -> archivo vacio (imagen de fondo)
        dst_label.write_text("", encoding="utf-8")

    return True


def validate_label_file(label_path: Path, valid_classes: set) -> tuple[bool, int]:
    """
    Valida que un archivo de etiquetas tiene formato correcto y clases validas.
    Devuelve (es_valido, num_detecciones).
    """
    if not label_path.exists():
        return False, 0

    content = label_path.read_text(encoding="utf-8", errors="replace").strip()
    if not content:
        return True, 0  # Archivo vacio = imagen de fondo, valida

    detections = 0
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            return False, 0
        try:
            cls = int(float(parts[0]))
            if valid_classes and cls not in valid_classes:
                return False, 0
            # Verificar coordenadas en rango [0,1]
            coords = [float(x) for x in parts[1:5]]
            if not all(0.0 <= c <= 1.0 for c in coords):
                return False, 0
            detections += 1
        except ValueError:
            return False, 0

    return True, detections


def add_veo_frames(dry_run: bool = False) -> tuple[int, int]:
    """
    Anade los frames VEO auto-etiquetados al hybrid_dataset.
    Los divide en train (85%) y valid (15%) automaticamente.

    Returns:
        (n_train, n_valid) frames anadidos
    """
    if not VEO_DIR.exists():
        print(f"[WARN] No se encontro directorio VEO: {VEO_DIR}")
        return 0, 0

    # Recopilar todos los frames VEO con etiquetas validas
    valid_classes = set(TARGET_CLASSES.keys())
    valid_pairs   = []
    skipped       = 0

    img_files = sorted([
        f for f in VEO_DIR.glob("*")
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])

    for img in img_files:
        label = VEO_LABELS / (img.stem + ".txt")
        is_valid, n_dets = validate_label_file(label, valid_classes)
        if is_valid and n_dets > 0:
            valid_pairs.append((img, label))
        else:
            skipped += 1

    if not valid_pairs:
        print(f"[WARN] No se encontraron pares imagen/etiqueta validos en {VEO_DIR}")
        return 0, 0

    print(f"  Pares validos: {len(valid_pairs)} | Saltados: {skipped}")

    # Mezclar aleatoriamente y dividir
    random.seed(42)
    random.shuffle(valid_pairs)
    n_val   = max(1, int(len(valid_pairs) * VAL_RATIO))
    n_train = len(valid_pairs) - n_val

    train_pairs = valid_pairs[:n_train]
    val_pairs   = valid_pairs[n_train:]

    # Copiar al dataset
    train_imgs = DATASET_DIR / "train" / "images"
    train_labs = DATASET_DIR / "train" / "labels"
    val_imgs   = DATASET_DIR / "valid" / "images"
    val_labs   = DATASET_DIR / "valid" / "labels"

    added_train = 0
    added_val   = 0

    for img, label in train_pairs:
        if copy_with_label(img, label, train_imgs, train_labs, prefix="veo", dry_run=dry_run):
            added_train += 1

    for img, label in val_pairs:
        if copy_with_label(img, label, val_imgs, val_labs, prefix="veo", dry_run=dry_run):
            added_val += 1

    return added_train, added_val


def add_local_frames(dry_run: bool = False) -> tuple[int, int]:
    """
    Anade frames del dataset local si existe.
    Asume que ya estan en el orden correcto de clases (0=player, 1=goalkeeper...).
    """
    if not LOCAL_DIR.exists():
        return 0, 0

    added_train = added_val = 0
    for split, dst_split in [("train", "train"), ("valid", "valid")]:
        src_imgs = LOCAL_DIR / split / "images"
        src_labs = LOCAL_DIR / split / "labels"
        if not src_imgs.exists():
            continue

        dst_imgs = DATASET_DIR / dst_split / "images"
        dst_labs = DATASET_DIR / dst_split / "labels"

        for img in src_imgs.glob("*"):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            label = src_labs / (img.stem + ".txt")
            if copy_with_label(img, label, dst_imgs, dst_labs, prefix="local", dry_run=dry_run):
                if split == "train":
                    added_train += 1
                else:
                    added_val += 1

    return added_train, added_val


def update_data_yaml():
    """Actualiza data.yaml con las rutas y clases correctas."""
    yaml_content = f"""# EDApp - Dataset deteccion de futbol
# Capa 1: deteccion de objetos (jugador, portero, arbitro, balon)
# La separacion de equipos la hace TeamClassifier en capa 2
path: {DATASET_DIR.as_posix()}
train: train/images
val:   valid/images

nc: 4
names:
  0: player
  1: goalkeeper
  2: referee
  3: ball
"""
    yaml_path = DATASET_DIR / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"[OK] data.yaml actualizado: {yaml_path}")


def print_summary(label: str = ""):
    """Muestra el estado actual del dataset."""
    train_n = count_images(DATASET_DIR / "train" / "images")
    val_n   = count_images(DATASET_DIR / "valid" / "images")
    total   = train_n + val_n
    print(f"\n  {'Dataset ' + label if label else 'Dataset'} — estado actual:")
    print(f"    Train: {train_n:>5} imagenes")
    print(f"    Valid: {val_n:>5} imagenes")
    print(f"    Total: {total:>5} imagenes")


def main():
    parser = argparse.ArgumentParser(description="Fusionar fuentes de datos en hybrid_dataset")
    parser.add_argument("--clean",   action="store_true", help="Limpiar y reconstruir desde cero")
    parser.add_argument("--dry-run", action="store_true", help="Simular sin copiar nada")
    args = parser.parse_args()

    print("=" * 60)
    print("  Build Hybrid Dataset v2")
    print("=" * 60)

    if args.dry_run:
        print("  [DRY RUN] Simulacion — no se copiara ningun archivo")

    # Verificar que el dataset base existe
    if not DATASET_DIR.exists():
        print(f"[ERROR] No se encontro el dataset base: {DATASET_DIR}")
        print("  Ejecuta primero: python ml/training/create_val_split.py")
        sys.exit(1)

    # Estado inicial
    print_summary("ANTES")

    if args.clean:
        print(f"\n[CLEAN] Limpiando dataset...")
        for split in ["train", "valid"]:
            for subdir in ["images", "labels"]:
                d = DATASET_DIR / split / subdir
                if d.exists():
                    shutil.rmtree(d)
                    d.mkdir(parents=True)
        print("  Listo.")

    # ── Anadir frames VEO ──────────────────────────────────────────────────
    print(f"\n[1/2] Anadiendo frames VEO auto-etiquetados...")
    print(f"  Origen imagenes: {VEO_DIR}")
    print(f"  Origen etiquetas: {VEO_LABELS}")

    veo_train, veo_val = add_veo_frames(dry_run=args.dry_run)
    print(f"  Anadidos: {veo_train} train + {veo_val} valid = {veo_train + veo_val} total")

    # ── Anadir dataset local (si existe) ───────────────────────────────────
    print(f"\n[2/3] Buscando dataset local adicional...")
    if LOCAL_DIR.exists():
        loc_train, loc_val = add_local_frames(dry_run=args.dry_run)
        print(f"  Anadidos: {loc_train} train + {loc_val} valid")
    else:
        print(f"  No encontrado ({LOCAL_DIR}) — OK, saltando")

    # ── Anadir frames manuales finales (VEO_FINAL_DIR) ───────────────────
    print(f"\n[3/3] Anadiendo frames corregidos manualmente (veo_frames_raw_final)...")
    if VEO_FINAL_DIR.exists():
        # Reutilizamos la lógica de local pero adaptada a la estructura de veo_frames_raw_final
        # En veo_frames_raw_final están en /images y /labels directamente
        added_f = 0
        src_imgs = VEO_FINAL_DIR / "images"
        src_labs = VEO_FINAL_DIR / "labels"
        dst_imgs = DATASET_DIR / "train" / "images"
        dst_labs = DATASET_DIR / "train" / "labels"

        for img in src_imgs.glob("*.jpg"):
            label = src_labs / (img.stem + ".txt")
            if copy_with_label(img, label, dst_imgs, dst_labs, prefix="final", dry_run=args.dry_run):
                added_f += 1
        print(f"  Anadidos: {added_f} frames revisados.")
    else:
        print(f"  No encontrado ({VEO_FINAL_DIR}) — saltando")

    # Actualizar data.yaml
    if not args.dry_run:
        update_data_yaml()

    # Estado final
    print_summary("DESPUES")

    # Instrucciones siguientes
    if not args.dry_run and (veo_train + veo_val) > 0:
        print(f"\n{'='*60}")
        print(f"  DATASET LISTO PARA ENTRENAR")
        print(f"{'='*60}")
        print(f"\n  Siguiente paso — lanza el entrenamiento v2 esta noche:")
        print(f"  .\\venv_cuda\\Scripts\\python.exe ml/training/train_unified.py --target players")
        print()
    elif args.dry_run:
        print(f"\n  Simulacion completada. Lanza sin --dry-run para aplicar los cambios.")


if __name__ == "__main__":
    main()
