
import os
import shutil
import random
from pathlib import Path

def split_dataset(src_img_dir, src_lbl_dir, dest_dir, split_ratio=0.8):
    src_img_dir = Path(src_img_dir)
    src_lbl_dir = Path(src_lbl_dir)
    dest_dir = Path(dest_dir)

    # Crear estructura
    for folder in ['train/images', 'train/labels', 'val/images', 'val/labels']:
        (dest_dir / folder).mkdir(parents=True, exist_ok=True)

    images = list(src_img_dir.glob("*.jpg"))
    random.shuffle(images)

    split_idx = int(len(images) * split_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    def copy_set(image_list, subset):
        for img_path in image_list:
            # Copiar imagen
            shutil.copy(img_path, dest_dir / subset / "images" / img_path.name)
            
            # Copiar label correspondiente
            lbl_name = img_path.stem + ".txt"
            lbl_path = src_lbl_dir / lbl_name
            if lbl_path.exists():
                shutil.copy(lbl_path, dest_dir / subset / "labels" / lbl_name)

    print(f"Dividiendo {len(images)} imágenes: {len(train_images)} train, {len(val_images)} val")
    copy_set(train_images, "train")
    copy_set(val_images, "val")
    print("Reorganización completada.")

if __name__ == "__main__":
    split_dataset(
        "c:/apped/football_analyzer/dataset_prep/images",
        "c:/apped/football_analyzer/dataset_prep/labels",
        "c:/apped/football_analyzer/dataset_pro"
    )
