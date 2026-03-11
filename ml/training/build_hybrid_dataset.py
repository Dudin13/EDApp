"""
build_hybrid_dataset.py
=======================
Construye un dataset hibrido para entrenar YOLOv8-seg mezclando:

1. roboflow_dataset/   -> 1070 imagenes con POLIGONOS (segmentacion)
   Clases: 0=Goalkeeper, 1=Player, 2=ball, 3=referee

2. football-players-detection (Roboflow) -> ~500 imagenes con BBOX
   Clases: 0=ball, 1=goalkeeper, 2=player, 3=referee  (necesita remapeo)

3. local_dataset/ (si existe) -> Frames propios con BBOX
   Clases: 0=player, 1=goalkeeper, 2=referee, 3=ball  (necesita remapeo)

YOLOv8-seg acepta etiquetas mixtas:
 - Lineas con POLIGONO (clase + pares xy): entrenan detection + segmentation
 - Lineas con BBOX (clase + cx cy w h): entrenan SOLO detection
El resultado es un modelo que detecta bien gracias a mas datos, 
y segmenta bien gracias a las imagenes con poligono.

Clases TARGET (alineadas con el modelo actual):
  0 = Goalkeeper
  1 = Player
  2 = ball
  3 = referee
"""

import os
import sys
import shutil
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
TRAIN_DIR = Path(__file__).parent
OUTPUT_DIR = ROOT / "hybrid_dataset"

# Clases TARGET del modelo (no cambiar)
TARGET_CLASSES = {0: "Goalkeeper", 1: "Player", 2: "ball", 3: "referee"}

load_dotenv(ROOT / "football_analyzer" / ".env")
API_KEY = os.getenv("ROBOFLOW_API_KEY", "")


# ─── REMAPEOS ────────────────────────────────────────────────────────────────
# De las clases del dataset de Roboflow bbox (football-players-detection)
# Esas tienen: 0=ball, 1=goalkeeper, 2=player, 3=referee  (distintas al target)
REMAP_RFBBOX = {0: 2, 1: 0, 2: 1, 3: 3}  # ball->2, goalkeeper->0, player->1, ref->3

# De las clases locales (local_dataset)
# Esas tienen: 0=player, 1=goalkeeper, 2=referee, 3=ball
REMAP_LOCAL = {0: 1, 1: 0, 2: 3, 3: 2}  # player->1, goalkeeper->0, ref->3, ball->2


def remap_label_file(src: Path, dst: Path, remap: dict):
    """Copia un archivo de etiquetas aplicando remapeo de clases."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines_out = []
    for line in src.read_text(encoding="utf-8", errors="replace").strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        old_cls = int(float(parts[0]))
        new_cls = remap.get(old_cls, old_cls)
        lines_out.append(f"{new_cls} " + " ".join(parts[1:]))
    dst.write_text("\n".join(lines_out) + "\n", encoding="utf-8")


def copy_split(src_images: Path, src_labels: Path, split: str, remap: dict = None):
    """Copia imagenes y etiquetas al split correspondiente del hybrid_dataset."""
    dst_images = OUTPUT_DIR / split / "images"
    dst_labels = OUTPUT_DIR / split / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    copied = 0
    for img in src_images.glob("*"):
        if img.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        label = src_labels / (img.stem + ".txt")
        if not label.exists():
            # Imagen sin etiqueta -> copiar de todas formas como background
            shutil.copy2(img, dst_images / img.name)
            (dst_labels / (img.stem + ".txt")).write_text("", encoding="utf-8")
            copied += 1
            continue

        # Copiar imagen
        dst_img = dst_images / img.name
        if dst_img.exists():
            # Evitar colision de nombres
            dst_img = dst_images / f"rf_{img.name}"
        shutil.copy2(img, dst_img)

        # Copiar/remapear etiqueta
        stem = dst_img.stem
        if remap:
            remap_label_file(label, dst_labels / f"{stem}.txt", remap)
        else:
            shutil.copy2(label, dst_labels / f"{stem}.txt")
        copied += 1
    return copied


def download_extra_dataset():
    """Descarga football-players-detection de Roboflow (bbox)."""
    if not API_KEY:
        print("[WARN] Sin API key de Roboflow. Saltando descarga extra.")
        return None

    dest = ROOT / "roboflow_bbox_dataset"
    if dest.exists() and any(dest.rglob("*.jpg")):
        print(f"[INFO] Dataset bbox ya existe en {dest}. Usando el existente.")
        return dest

    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=API_KEY)
        # football-players-detection por roboflow-jvuqo — muy popular, ~500 imgs
        project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
        dataset = project.version(1).download("yolov8", location=str(dest))
        print(f"[OK] Dataset bbox descargado en {dest}")
        return dest
    except Exception as e:
        print(f"[WARN] No se pudo descargar dataset extra: {e}")
        return None


def write_data_yaml():
    yaml_content = f"""# Dataset hibrido: segmentacion (poligono) + detection (bbox)
# YOLOv8-seg entrena ambas cabezas con los datos que tienen mascara,
# y solo la cabeza de deteccion con los que solo tienen bbox.
path: {OUTPUT_DIR.as_posix()}
train: train/images
val: valid/images

nc: 4
names:
  0: Goalkeeper
  1: Player
  2: ball
  3: referee
"""
    (OUTPUT_DIR / "data.yaml").write_text(yaml_content, encoding="utf-8")
    print(f"[OK] data.yaml escrito en {OUTPUT_DIR / 'data.yaml'}")


def main():
    print("=" * 60)
    print("  Build Hybrid Dataset (Seg + BBox)")
    print("=" * 60)

    # Limpiar si existe
    if OUTPUT_DIR.exists():
        print(f"[INFO] Limpiando {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    total = 0

    # ── 1. Dataset Roboflow seg (polígonos) — SIN remapeo (ya esta en orden correcto)
    rf_seg = ROOT / "roboflow_dataset"
    if rf_seg.exists():
        print(f"\n[1/3] Copiando dataset seg (poligonos)...")
        for split in ["train", "valid"]:
            src_imgs = rf_seg / split / "images"
            src_labs = rf_seg / split / "labels"
            if src_imgs.exists():
                n = copy_split(src_imgs, src_labs, split, remap=None)
                total += n
                print(f"   {split}: {n} imagenes")
    else:
        print(f"[WARN] roboflow_dataset no encontrado en {rf_seg}")

    # ── 2. Dataset Roboflow bbox adicional (football-players-detection)
    print(f"\n[2/3] Descargando dataset bbox adicional de Roboflow...")
    rf_bbox = download_extra_dataset()
    if rf_bbox and rf_bbox.exists():
        # Este dataset tiene clases: 0=ball, 1=goalkeeper, 2=player, 3=referee
        for split in ["train", "valid"]:
            src_imgs = rf_bbox / split / "images"
            src_labs = rf_bbox / split / "labels"
            if not src_imgs.exists():
                # probar con "validation" como nombre
                src_imgs = rf_bbox / "valid" / "images"
                src_labs = rf_bbox / "valid" / "labels"
                if not src_imgs.exists():
                    continue
            n = copy_split(src_imgs, src_labs, split, remap=REMAP_RFBBOX)
            total += n
            print(f"   {split}: {n} imagenes (bbox, remapeadas)")

    # ── 3. Dataset local (si existe y tiene etiquetas validas)
    local_ds = ROOT / "local_dataset"
    if local_ds.exists():
        print(f"\n[3/3] Añadiendo dataset local...")
        src_imgs = local_ds / "train" / "images"
        src_labs = local_ds / "train" / "labels"
        if src_imgs.exists():
            n = copy_split(src_imgs, src_labs, "train", remap=REMAP_LOCAL)
            total += n
            print(f"   train: {n} imagenes (local, bbox)")
    else:
        print(f"\n[3/3] Sin dataset local — OK")

    write_data_yaml()

    # Resumen
    train_count = len(list((OUTPUT_DIR / "train" / "images").glob("*")))
    valid_count = len(list((OUTPUT_DIR / "valid" / "images").glob("*")))
    print(f"\n{'='*60}")
    print(f"  DATASET HIBRIDO LISTO")
    print(f"  Train: {train_count} imagenes | Valid: {valid_count} imagenes")
    print(f"  Total: {train_count + valid_count} imagenes")
    print(f"  Ruta:  {OUTPUT_DIR}")
    print(f"{'='*60}")
    print(f"\n[NEXT] Lanza el entrenamiento con:")
    print(f"  python train_yolo/retrain_ball.py")
    print(f"  (Asegurate de cambiar DATA_YAML a: {OUTPUT_DIR / 'data.yaml'})")


if __name__ == "__main__":
    main()
