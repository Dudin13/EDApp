# -*- coding: utf-8 -*-
"""
prepare_dataset.py - Prepara el dataset de entrenamiento para detect_players_v4_clean

Fuentes:
  1. roboflow_bbox_dataset  (663 imgs, labels bbox correctos, clases a remapear)
  2. labeller_flow/2_entrenadas_manual  (imagenes etiquetadas manualmente)
  3. labeller_flow/3_final  (si hay)

Mapeado de clases roboflow -> proyecto:
  roboflow: 0=ball, 1=goalkeeper, 2=player, 3=referee
  proyecto:  0=player, 1=goalkeeper, 2=referee, 3=ball
  remap:     {0->3, 1->1, 2->0, 3->2}

Salida: data/datasets/players_v4_dataset/
"""

import os
import shutil
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR      = PROJECT_ROOT / "data" / "datasets" / "players_v4_dataset"
ROBOFLOW_DIR = PROJECT_ROOT / "data" / "datasets" / "roboflow_bbox_dataset"
MANUAL_DIR   = PROJECT_ROOT / "data" / "datasets" / "labeller_flow"

# Mapeado roboflow -> proyecto
CLASS_REMAP = {0: 3, 1: 1, 2: 0, 3: 2}

# Limpia y crea estructura de salida
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)

for split in ("train", "valid"):
    (OUT_DIR / split / "images").mkdir(parents=True)
    (OUT_DIR / split / "labels").mkdir(parents=True)

print(f"[Prepare] Directorio de salida: {OUT_DIR}")


def remap_label_file(src, dst, remap):
    lines = []
    for line in src.read_text(encoding="utf-8", errors="ignore").strip().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        old_cls = int(parts[0])
        new_cls = remap.get(old_cls, old_cls)
        lines.append(f"{new_cls} " + " ".join(parts[1:]))
    dst.write_text("\n".join(lines), encoding="utf-8")


def copy_pair(img_src, lbl_src, split, suffix=""):
    stem = img_src.stem + suffix
    shutil.copy2(img_src, OUT_DIR / split / "images" / (stem + img_src.suffix))
    shutil.copy2(lbl_src, OUT_DIR / split / "labels" / (stem + ".txt"))


# 1. Roboflow bbox dataset (con remapeo de clases)
print("\n[1/3] Procesando roboflow_bbox_dataset...")
count_rb = 0
for split_name in ("train", "valid"):
    img_dir = ROBOFLOW_DIR / split_name / "images"
    lbl_dir = ROBOFLOW_DIR / split_name / "labels"
    for img in list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")):
        lbl = lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            continue
        shutil.copy2(img, OUT_DIR / split_name / "images" / img.name)
        remap_label_file(lbl, OUT_DIR / split_name / "labels" / (img.stem + ".txt"), CLASS_REMAP)
        count_rb += 1

print(f"  -> {count_rb} pares copiados (clases remapeadas)")


# 2. Imagenes etiquetadas manualmente
print("\n[2/3] Procesando datos manuales del labeller...")
manual_pairs = []
for src_dir in [MANUAL_DIR / "2_entrenadas_manual", MANUAL_DIR / "3_final"]:
    if not src_dir.exists():
        continue
    lbl_dir = src_dir / "labels"
    for img in src_dir.glob("*"):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lbl = lbl_dir / (img.stem + ".txt")
        if lbl.exists() and lbl.stat().st_size > 0:
            manual_pairs.append((img, lbl))

random.seed(42)
random.shuffle(manual_pairs)
split_idx = max(1, int(len(manual_pairs) * 0.85))
count_manual = 0
for split_name, pairs in [("train", manual_pairs[:split_idx]), ("valid", manual_pairs[split_idx:])]:
    for img, lbl in pairs:
        copy_pair(img, lbl, split_name, suffix="_manual")
        count_manual += 1

print(f"  -> {count_manual} pares manuales copiados")


# 3. Resumen y data.yaml
train_count = len(list((OUT_DIR / "train" / "images").iterdir()))
valid_count  = len(list((OUT_DIR / "valid"  / "images").iterdir()))

print(f"\n[3/3] Dataset final:")
print(f"  train: {train_count} imagenes")
print(f"  valid: {valid_count} imagenes")
print(f"  total: {train_count + valid_count}")

yaml_content = (
    "# EDudin - Dataset deteccion de jugadores v4 (limpio)\n"
    "# Generado por: ml/training/prepare_dataset.py\n"
    f"path: {OUT_DIR.as_posix()}\n"
    "train: train/images\n"
    "val:   valid/images\n"
    "\n"
    "nc: 4\n"
    "names:\n"
    "  0: player\n"
    "  1: goalkeeper\n"
    "  2: referee\n"
    "  3: ball\n"
)
(OUT_DIR / "data.yaml").write_text(yaml_content, encoding="utf-8")
print(f"\n[OK] data.yaml escrito en {OUT_DIR / 'data.yaml'}")
print("\nEjecuta ahora: python ml/training/train_players_v4.py")
