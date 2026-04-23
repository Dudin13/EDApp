import os
import random
import shutil
from pathlib import Path

def prepare_v6_general_split():
    print("Preparing V6 General Split (Manually reviewed images)...")
    
    dataset_root = Path("data/datasets/v6_general_dataset")
    img_dir = dataset_root / "images"
    lbl_dir = dataset_root / "labels"
    
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    
    # Define sources to scan for manual annotations
    sources = [
        {
            "labels": Path("data/datasets/veo_frames_raw/labels"),
            "images": Path("data/datasets/veo_frames_raw/images")
        },
        {
            "labels": Path("data/datasets/hybrid_dataset/train/labels"),
            "images": Path("data/datasets/hybrid_dataset/train/images")
        },
        {
            "labels": Path("data/datasets/hybrid_dataset/val/labels"),
            "images": Path("data/datasets/hybrid_dataset/val/images")
        }
    ]
    
    manual_images_meta = []
    
    for src in sources:
        if not src["labels"].exists():
            print(f"Warning: Label path {src['labels']} not found.")
            continue
            
        for label_file in src["labels"].glob("*.txt"):
            try:
                content = label_file.read_text(encoding='utf-8').splitlines()
                # Check if it has the manual marker
                if any("# revisado_manual" in line for line in content):
                    img_name = label_file.stem + ".jpg"
                    img_path = src["images"] / img_name
                    if not img_path.exists():
                        img_path = src["images"] / (label_file.stem + ".png")
                    
                    if img_path.exists():
                        manual_images_meta.append({
                            "img": img_path,
                            "lbl_original": label_file,
                            "content": content
                        })
            except Exception as e:
                print(f"Error reading {label_file}: {e}")
                
    print(f"Found {len(manual_images_meta)} manually reviewed images.")
    
    # Shuffle for split
    random.seed(42)
    random.shuffle(manual_images_meta)
    
    train_metadata = []
    
    for item in manual_images_meta:
        img_p = item["img"]
        lbl_p = item["lbl_original"]
        lines = item["content"]
        
        # Link or Copy image
        target_img = img_dir / img_p.name
        if not target_img.exists():
            try:
                # Use hardlink if possible to save space
                os.link(img_p, target_img)
            except:
                shutil.copy(img_p, target_img)
        
        # Clean and save label (remove comments/markers)
        clean_lines = [line for line in lines if not line.strip().startswith("#")]
        (lbl_dir / lbl_p.name).write_text("\n".join(clean_lines))
        
        # Use absolute path for TXT files
        train_metadata.append(str(target_img.absolute()))

    # Split 90/10
    split_idx = int(len(train_metadata) * 0.9)
    train_imgs = train_metadata[:split_idx]
    val_imgs = train_metadata[split_idx:]
    
    # Save split lists in data/datasets
    train_txt = Path("data/datasets/v6_general_train.txt")
    val_txt = Path("data/datasets/v6_general_val.txt")
    train_txt.write_text("\n".join(train_imgs))
    val_txt.write_text("\n".join(val_imgs))
    
    # Generate YAML
    yaml_content = f"""
path: {dataset_root.absolute()}
train: {train_txt.absolute()}
val: {val_txt.absolute()}

nc: 4
names:
  0: player
  1: goalkeeper
  2: referee
  3: ball
"""
    Path("data/datasets/v6_general.yaml").write_text(yaml_content.strip())
    print(f"V6 split preparation complete. Total: {len(manual_images_meta)} images.")

if __name__ == "__main__":
    prepare_v6_general_split()
