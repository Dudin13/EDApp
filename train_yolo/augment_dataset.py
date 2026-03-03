"""
augment_dataset.py — Aumentación del dataset de entrenamiento.

A partir de las 50 imágenes originales genera versiones aumentadas
para llegar a ~200+ imágenes efectivas de entrenamiento.

Aumentaciones:
  - Flip horizontal (con ajuste de etiquetas YOLO)
  - Variaciones de brillo/contraste
  - Cambios de color (ecualizacion HSV)
  - Ruido gaussiano

Uso:
    python train_yolo/augment_dataset.py

Las imágenes y etiquetas aumentadas se guardan en las mismas carpetas
(frames_etiquetado/ y labels_yolo/) con sufijos _augN.
"""

import cv2
import numpy as np
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
IMAGES_DIR = ROOT / "frames_etiquetado"
LABELS_DIR = ROOT / "labels_yolo"


def load_label(label_path: Path) -> list:
    """Carga etiquetas YOLO: [[cls, cx, cy, w, h], ...]"""
    if not label_path.exists():
        return []
    lines = label_path.read_text().strip().split("\n")
    result = []
    for line in lines:
        if line.strip():
            parts = line.strip().split()
            if len(parts) == 5:
                result.append([int(parts[0])] + [float(x) for x in parts[1:]])
    return result


def save_label(label_path: Path, labels: list):
    lines = [f"{int(l[0])} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}" for l in labels]
    label_path.write_text("\n".join(lines))


def flip_horizontal(image: np.ndarray, labels: list) -> tuple:
    """Flip horizontal: cx_new = 1 - cx"""
    flipped = cv2.flip(image, 1)
    new_labels = []
    for l in labels:
        cls, cx, cy, w, h = l
        new_labels.append([cls, 1.0 - cx, cy, w, h])
    return flipped, new_labels


def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """Factor > 1 = más brillo, < 1 = más oscuro"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def add_noise(image: np.ndarray, sigma: float = 10) -> np.ndarray:
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def adjust_saturation(image: np.ndarray, factor: float) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def augment_dataset(target_per_image: int = 3):
    """
    Genera `target_per_image` versiones aumentadas por cada imagen original.
    """
    images = sorted(list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png")))
    print(f"📂 Imágenes originales: {len(images)}")

    augmentations = [
        # (nombre_sufijo, fn_imagen, fn_etiquetas)
        ("_aug1_flip",     lambda img: flip_horizontal(img, [])[0],   "flip"),
        ("_aug2_bright",   lambda img: adjust_brightness(img, 1.3),   None),
        ("_aug3_dark",     lambda img: adjust_brightness(img, 0.7),   None),
        ("_aug4_sat",      lambda img: adjust_saturation(img, 1.4),   None),
        ("_aug5_noise",    lambda img: add_noise(img, 8),             None),
        ("_aug6_flipbr",   lambda img: adjust_brightness(flip_horizontal(img, [])[0], 1.2), "flip"),
    ]

    count_new = 0
    for img_path in images:
        label_path = LABELS_DIR / (img_path.stem + ".txt")
        labels = load_label(label_path)

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ No se pudo leer: {img_path}")
            continue

        aug_list = augmentations[:target_per_image]

        for suffix, aug_fn, label_mode in aug_list:
            new_img_path = IMAGES_DIR / (img_path.stem + suffix + img_path.suffix)
            new_lab_path = LABELS_DIR / (img_path.stem + suffix + ".txt")

            if new_img_path.exists():
                continue  # Ya existe

            try:
                aug_img = aug_fn(img)
                cv2.imwrite(str(new_img_path), aug_img)

                # Ajustar etiquetas según el tipo de aumentación
                if label_mode == "flip":
                    aug_labels = [[l[0], 1.0 - l[1], l[2], l[3], l[4]] for l in labels]
                else:
                    aug_labels = labels  # mismas etiquetas (solo cambio de color/brillo)

                save_label(new_lab_path, aug_labels)
                count_new += 1

            except Exception as e:
                print(f"⚠️ Error augmentando {img_path.name}: {e}")

    total = len(images) + count_new
    print(f"✅ Augmentación completada: {count_new} nuevas imágenes generadas")
    print(f"   Total dataset: {total} imágenes")
    return count_new


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print("=" * 60)
    print("  ED Analytics - Data Augmentation")
    print("=" * 60)
    augment_dataset(target_per_image=3)
    print("\n>> Ahora ejecuta: python train_yolo/train.py")
