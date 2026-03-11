import os
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
ROBOFLOW_DIR = ROOT / "roboflow_dataset"
LOCAL_DIR = ROOT / "dataset"
COMBINED_DIR = ROOT / "combined_dataset"

def combine():
    print("=" * 60)
    print("  COMBINANDO DATASETS PARA ENTRENAMIENTO FINAL")
    print("=" * 60)

    if COMBINED_DIR.exists():
        shutil.rmtree(COMBINED_DIR)
    
    # Crear estructura
    (COMBINED_DIR / "train" / "images").mkdir(parents=True, exist_ok=True)
    (COMBINED_DIR / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (COMBINED_DIR / "valid" / "images").mkdir(parents=True, exist_ok=True)
    (COMBINED_DIR / "valid" / "labels").mkdir(parents=True, exist_ok=True)

    copied_images = 0
    copied_labels = 0

    # 1. Copiar Roboflow (Train y Valid)
    print("Copiando dataset Roboflow (+1000 imagenes)...")
    for split in ["train", "valid"]:
        rf_imgs = ROBOFLOW_DIR / split / "images"
        rf_lbls = ROBOFLOW_DIR / split / "labels"
        
        if rf_imgs.exists():
            for img in rf_imgs.glob("*.jpg"):
                shutil.copy(img, COMBINED_DIR / split / "images" / img.name)
                copied_images += 1
        
        if rf_lbls.exists():
            for lbl in rf_lbls.glob("*.txt"):
                shutil.copy(lbl, COMBINED_DIR / split / "labels" / lbl.name)
                copied_labels += 1

    # 2. Copiar Local (Todo a Train para forzar aprendizaje del contexto)
    print("Copiando dataset Local (50 frames locales aumentados)...")
    local_imgs = LOCAL_DIR / "images" / "train"
    local_lbls = LOCAL_DIR / "labels" / "train"
    
    if local_imgs.exists():
        for img in local_imgs.glob("*.jpg"):
            shutil.copy(img, COMBINED_DIR / "train" / "images" / f"local_{img.name}")
            copied_images += 1
            
    if local_lbls.exists():
        for lbl in local_lbls.glob("*.txt"):
            shutil.copy(lbl, COMBINED_DIR / "train" / "labels" / f"local_{lbl.name}")
            copied_labels += 1

    # 3. Crear data.yaml
    yaml_content = f"""# YOLOv8 Dataset combinado
path: {COMBINED_DIR.as_posix()}
train: train/images
val: valid/images

nc: 4
names:
  0: player
  1: goalkeeper
  2: referee
  3: ball
"""
    yaml_path = COMBINED_DIR / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")

    print(f"\nDataset combinado creado con exito en {COMBINED_DIR}")
    print(f"   Total imagenes copiadas (aprox): {copied_images}")
    print(f"   Total etiquetas copiadas (aprox): {copied_labels}")
    print(f"   Archivo de configuracion: {yaml_path}")

if __name__ == "__main__":
    combine()
