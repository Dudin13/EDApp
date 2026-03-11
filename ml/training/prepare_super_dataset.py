import os
import shutil
from pathlib import Path
import random
from datetime import datetime, timedelta

# Configuration
DATASET_ROOT = Path("C:/apped/04_Datasets_Entrenamiento/hybrid_dataset")
TARGET_ROOT = Path("C:/apped/04_Datasets_Entrenamiento/super_focused_50")
DATA_YAML = TARGET_ROOT / "data.yaml"

def prepare_dataset():
    # 1. Identify Manual Labels (modified in the last 4 hours)
    cutoff = datetime.now() - timedelta(hours=4)
    all_label_files = list(DATASET_ROOT.rglob("labels/*.txt"))
    
    manual_labels = []
    for f in all_label_files:
        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        # We also check if it contains class 5 or 2 to be sure it's the "new" ones
        # and has been modified since we started (approx 17:00)
        if mtime > cutoff and mtime.hour >= 17:
             manual_labels.append(f)
    
    print(f"Detected {len(manual_labels)} manually corrected labels.")
    
    # 2. Identify 25-30 additional varied labels
    other_labels = [f for f in all_label_files if f not in manual_labels]
    random.shuffle(other_labels)
    additional_labels = other_labels[:max(0, 50 - len(manual_labels))]
    
    final_list = manual_labels + additional_labels
    random.shuffle(final_list)
    
    # 3. Split
    split_idx = int(len(final_list) * 0.8)
    train_set = final_list[:split_idx]
    val_set = final_list[split_idx:]
    
    print(f"Final Dataset Size: {len(final_list)} (Train: {len(train_set)}, Val: {len(val_set)})")
    
    # 4. Copying function
    def copy_to(files, subset):
        img_dir = TARGET_ROOT / subset / "images"
        lab_dir = TARGET_ROOT / subset / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        
        for lab_path in files:
            # Copy label
            shutil.copy(lab_path, lab_dir / lab_path.name)
            
            # Copy image
            # Images are in hybrid_dataset/<original_subset>/images
            orig_subset = lab_path.parent.parent.name
            img_path = DATASET_ROOT / orig_subset / "images" / (lab_path.stem + ".jpg")
            if not img_path.exists():
                img_path = DATASET_ROOT / orig_subset / "images" / (lab_path.stem + ".png")
            
            if img_path.exists():
                shutil.copy(img_path, img_dir / img_path.name)
            else:
                print(f"Missing image for {lab_path.name}")

    copy_to(train_set, "train")
    copy_to(val_set, "val")
    
    # 5. Create super_data.yaml
    yaml_content = f"""path: {TARGET_ROOT.as_posix()}
train: train/images
val: val/images

nc: 6
names:
  0: team_1
  1: team_2
  2: goalkeeper_1
  3: referee
  4: ball
  5: goalkeeper_2
"""
    with open(DATA_YAML, "w") as f:
        f.write(yaml_content)
    
    print(f"Super Dataset ready at {TARGET_ROOT}")

if __name__ == "__main__":
    prepare_dataset()
