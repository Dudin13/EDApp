import os
import random
import shutil
from pathlib import Path

def prepare_v7_general_split():
    print("=" * 60)
    print("Preparing V7 General Split...")
    print("=" * 60)
    
    dataset_root = Path("data/datasets/v7_general_dataset")
    img_dir = dataset_root / "images"
    lbl_dir = dataset_root / "labels"
    
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    
    all_images_meta = []
    
    # ── 1. MANUALLY REVIEWED IMAGES (Expected: 1,593) ──
    print("\nScanning for manually reviewed images...")
    manual_sources = [
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
    
    manual_count = 0
    for src in manual_sources:
        if not src["labels"].exists():
            print(f"  Warning: Label path {src['labels']} not found.")
            continue
            
        for label_file in src["labels"].glob("*.txt"):
            try:
                content = label_file.read_text(encoding='utf-8', errors='replace').splitlines()
                # Check if it has the manual marker
                if any("# revisado_manual" in line for line in content):
                    img_name = label_file.stem + ".jpg"
                    img_path = src["images"] / img_name
                    if not img_path.exists():
                        img_path = src["images"] / (label_file.stem + ".png")
                    
                    if img_path.exists():
                        # Clean lines (remove comment markers)
                        clean_lines = [line for line in content if not line.strip().startswith("#")]
                        all_images_meta.append({
                            "img": img_path,
                            "lbl_name": label_file.name,
                            "lines": clean_lines,
                            "source": "manual"
                        })
                        manual_count += 1
            except Exception as e:
                print(f"  Error reading manual label {label_file}: {e}")
                
    print(f"  Added {manual_count} manually reviewed images.")
    
    # ── 2. ROBOFLOW REMAPPED IMAGES (Expected: 372) ──
    print("\nScanning for Roboflow remapped images...")
    roboflow_root = Path("data/datasets/roboflow_new")
    roboflow_count = 0
    
    if roboflow_root.exists():
        # Scan train/valid/test directories
        for split in ["train", "valid", "test"]:
            split_labs = roboflow_root / split / "labels"
            split_imgs = roboflow_root / split / "images"
            
            if split_labs.exists():
                for label_file in split_labs.glob("*.txt"):
                    img_name = label_file.stem + ".jpg"
                    img_path = split_imgs / img_name
                    if not img_path.exists():
                        img_path = split_imgs / (label_file.stem + ".png")
                        
                    if img_path.exists():
                        try:
                            lines = label_file.read_text(encoding='utf-8', errors='replace').splitlines()
                            all_images_meta.append({
                                "img": img_path,
                                "lbl_name": f"rf_{label_file.name}", # Prefix to prevent name collisions
                                "lines": lines,
                                "source": "roboflow"
                            })
                            roboflow_count += 1
                        except Exception as e:
                            print(f"  Error reading roboflow label {label_file}: {e}")
                            
    print(f"  Added {roboflow_count} Roboflow images.")
    
    # ── 3. SOCCERSYNTH REMAPPED IMAGES (SOLO JUGADORES) ──
    print("\nScanning for SoccerSynth remapped images (solo jugadores)...")
    soccersynth_root = Path("data/datasets/soccersynth_remapped")
    soccersynth_count = 0
    
    if soccersynth_root.exists() and (soccersynth_root / "labels").exists():
        src_labs = soccersynth_root / "labels"
        src_imgs = soccersynth_root / "images"
        
        for label_file in src_labs.glob("*.txt"):
            img_name = label_file.stem + ".jpg"
            img_path = src_imgs / img_name
            if not img_path.exists():
                img_path = src_imgs / (label_file.stem + ".png")
                
            if img_path.exists():
                try:
                    lines = label_file.read_text(encoding='utf-8', errors='replace').splitlines()
                    # Filter lines: only keep players (class 0)
                    filtered_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if parts:
                            cls_id = int(parts[0])
                            if cls_id == 0: # Only class 0 (player) is kept
                                filtered_lines.append(line)
                                
                    # Only add if there is at least one player detection
                    if filtered_lines:
                        all_images_meta.append({
                            "img": img_path,
                            "lbl_name": f"synth_{label_file.name}", # Prefix to prevent name collisions
                            "lines": filtered_lines,
                            "source": "soccersynth"
                        })
                        soccersynth_count += 1
                except Exception as e:
                    print(f"  Error reading SoccerSynth label {label_file}: {e}")
                    
    print(f"  Added {soccersynth_count} SoccerSynth images.")
    
    # ── 4. PROCESS AND COPY/HARDLINK ──
    print(f"\nProcessing total of {len(all_images_meta)} images...")
    
    train_metadata = []
    
    for item in all_images_meta:
        img_p = item["img"]
        lbl_name = item["lbl_name"]
        lines = item["lines"]
        
        # Determine unique destination image name based on lbl_name
        dest_img_name = Path(lbl_name).stem + img_p.suffix
        target_img = img_dir / dest_img_name
        
        # Link or copy image
        if not target_img.exists():
            try:
                os.link(img_p, target_img)
            except:
                try:
                    shutil.copy2(img_p, target_img)
                except Exception as e:
                    print(f"  Error copying image {img_p} to {target_img}: {e}")
                    continue
                    
        # Write clean labels
        target_lbl = lbl_dir / lbl_name
        try:
            target_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as e:
            print(f"  Error writing label to {target_lbl}: {e}")
            continue
            
        train_metadata.append(str(target_img.absolute()))
        
    print(f"Successfully processed {len(train_metadata)} images in total.")
    
    # ── 5. SPLIT 90/10 ──
    random.seed(42)
    random.shuffle(train_metadata)
    
    split_idx = int(len(train_metadata) * 0.9)
    train_imgs = train_metadata[:split_idx]
    val_imgs = train_metadata[split_idx:]
    
    # Save split lists
    train_txt = Path("data/datasets/v7_general_train.txt")
    val_txt = Path("data/datasets/v7_general_val.txt")
    
    train_txt.write_text("\n".join(train_imgs), encoding="utf-8")
    val_txt.write_text("\n".join(val_imgs), encoding="utf-8")
    
    # Generate YAML
    yaml_content = f"""
path: {dataset_root.absolute().as_posix()}
train: {train_txt.absolute().as_posix()}
val: {val_txt.absolute().as_posix()}

nc: 4
names:
  0: player
  1: goalkeeper
  2: referee
  3: ball
"""
    yaml_path = Path("data/datasets/v7_general.yaml")
    yaml_path.write_text(yaml_content.strip(), encoding="utf-8")
    
    print("\n" + "=" * 60)
    print("V7 split preparation complete.")
    print(f"YAML config saved to: {yaml_path.absolute()}")
    print(f"Total training images: {len(train_imgs)}")
    print(f"Total validation images: {len(val_imgs)}")
    print("=" * 60)

if __name__ == "__main__":
    prepare_v7_general_split()
