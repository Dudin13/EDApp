import os
import random
import shutil
from pathlib import Path

def prepare_general_split():
    print("Preparing General Split (1280 manual images)...")
    
    dataset_root = Path("data/datasets/v5_general_dataset")
    img_dir = dataset_root / "images"
    lbl_dir = dataset_root / "labels"
    
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    
    sources = [
        {
            "labels": Path("data/datasets/veo_frames_raw/labels"),
            "images": Path("data/datasets/veo_frames_raw/images")
        },
        {
            "labels": Path("data/datasets/hybrid_dataset/train/labels"),
            "images": Path("data/datasets/hybrid_dataset/train/images")
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
                
    print(f"Found {len(manual_images_meta)} manually reviewed images. Creating clean dataset...")
    
    # Shuffle
    random.seed(42)
    random.shuffle(manual_images_meta)
    
    train_metadata = []
    
    for item in manual_images_meta:
        img_p = item["img"]
        lbl_p = item["lbl_original"]
        lines = item["content"]
        
        # Link image
        target_img = img_dir / img_p.name
        if not target_img.exists():
            try:
                os.link(img_p, target_img)
            except:
                shutil.copy(img_p, target_img)
        
        # Clean and save label
        clean_lines = [line for line in lines if not line.strip().startswith("#")]
        (lbl_dir / lbl_p.name).write_text("\n".join(clean_lines))
        
        train_metadata.append(str(target_img.absolute()))

    split_idx = int(len(train_metadata) * 0.9)
    train_imgs = train_metadata[:split_idx]
    val_imgs = train_metadata[split_idx:]
    
    Path("data/datasets/v5_general_train.txt").write_text("\n".join(train_imgs))
    Path("data/datasets/v5_general_val.txt").write_text("\n".join(val_imgs))
    
    yaml_content = f"""
path: {dataset_root.absolute()}
train: images
val: images

nc: 4
names:
  0: player
  1: goalkeeper
  2: referee
  3: ball
"""
    Path("data/datasets/v5_general.yaml").write_text(yaml_content.strip())
    print("General split preparation complete.")

def prepare_ball_split():
    print("Preparing Ball Specialist Split (1926 images)...")
    
    label_src = Path("data/datasets/forzasys_labels")
    image_src = Path("data/datasets/veo_frames_raw/images")
    temp_label_dir = Path("data/datasets/v5_ball_temp_labels")
    temp_label_dir.mkdir(parents=True, exist_ok=True)
    
    ball_images = []
    
    for label_file in label_src.glob("*.txt"):
        try:
            lines = label_file.read_text().splitlines()
            ball_lines = []
            for line in lines:
                parts = line.split()
                if not parts: continue
                # Match class 3 (ball)
                if parts[0] == "3":
                    # Remap to class 0 for specialist model
                    parts[0] = "0"
                    ball_lines.append(" ".join(parts))
            
            if ball_lines:
                # Save remapped label
                new_label_path = temp_label_dir / label_file.name
                new_label_path.write_text("\n".join(ball_lines))
                
                # Check image existence
                img_name = label_file.stem + ".jpg"
                img_path = image_src / img_name
                if img_path.exists():
                    ball_images.append(str(img_path.absolute()))
        except Exception as e:
            print(f"Error processing {label_file}: {e}")
            
    print(f"Processed {len(ball_images)} ball images.")
    
    # Shuffle and split
    random.shuffle(ball_images)
    split_idx = int(len(ball_images) * 0.9)
    train_imgs = ball_images[:split_idx]
    val_imgs = ball_images[split_idx:]
    
    Path("data/datasets/v5_ball_train.txt").write_text("\n".join(train_imgs))
    Path("data/datasets/v5_ball_val.txt").write_text("\n".join(val_imgs))
    
    # YOLO needs the labels in a sibling directory named 'labels' relative to 'images' OR
    # in a directory where it can find them by replacing 'images' with 'labels' in the path.
    # Since we are using a list of images, it will look for labels in a folder named 'labels' 
    # relative to the image folder.
    # HOWEVER, we want to use the TEMPORARY labels.
    # The best way is to create a symlink or just tell YOLO where the labels are?
    # Actually, if we use a list of images, YOLO typically expects labels in the same 
    # folder structure as images but with 'images' replaced by 'labels'.
    
    # To avoid messing with the source folder, we'll create a symlink forest or 
    # just create a consolidated folder for the ball dataset.
    
    # Let's create a specialized dataset folder for the ball specialist to avoid path issues.
    ball_dataset_root = Path("data/datasets/v5_ball_dataset")
    (ball_dataset_root / "images").mkdir(parents=True, exist_ok=True)
    (ball_dataset_root / "labels").mkdir(parents=True, exist_ok=True)
    
    # We will use the previously mapped images paths but for the labels we'll use the ones in temp.
    # Wait, YOLO is picky. Let's just create a list of images and ensure labels are next to them?
    # No, let's just use a custom YAML that points to the temp labels.
    # Actually, YOLOv8 doesn't support separate label paths in YAML easily.
    
    # Let's do the standard approach:
    # 1. Create a folder with symlinks to images.
    # 2. Put our remapped labels in the sibling 'labels' folder.
    
    print("Creating structured ball dataset (symlinked images + remapped labels)...")
    for img_abs_path in ball_images:
        img_p = Path(img_abs_path)
        # Create symlink in ball_dataset_root/images
        target_img = ball_dataset_root / "images" / img_p.name
        if not target_img.exists():
            try:
                # On Windows, symlink might need admin. Use copy if it fails or just use hardlink.
                os.link(img_p, target_img)
            except:
                shutil.copy(img_p, target_img)
        
        # Copy remapped label
        label_name = img_p.stem + ".txt"
        shutil.copy(temp_label_dir / label_name, ball_dataset_root / "labels" / label_name)

    yaml_content = f"""
path: {ball_dataset_root.absolute()}
train: images
val: images

nc: 1
names:
  0: ball
"""
    Path("data/datasets/v5_ball.yaml").write_text(yaml_content.strip())
    print("Ball split preparation complete.")

if __name__ == "__main__":
    prepare_general_split()
    prepare_ball_split()
