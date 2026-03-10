import os
import cv2
import torch
from pathlib import Path
from ultralytics import SAM
from datetime import datetime, timedelta

# Configuration
DATASET_ROOT = Path("C:/apped/04_Datasets_Entrenamiento/hybrid_dataset")
MODEL_PATH = "sam2_b.pt"

def salvage():
    print("🛸 Initializing SAM Salvage Engine...")
    model = SAM(MODEL_PATH)
    
    # 1. Find manual labels (modified in last 4 hours)
    cutoff = datetime.now() - timedelta(hours=4)
    all_labels = list(DATASET_ROOT.rglob("labels/*.txt"))
    
    manual_files = []
    for f in all_labels:
        if datetime.fromtimestamp(f.stat().st_mtime) > cutoff:
            manual_files.append(f)
            
    if not manual_files:
        print("❌ No recently modified files found.")
        return

    print(f"📦 Processing {len(manual_files)} manual files...")

    for lab_file in manual_files:
        # Find matching image
        subset = lab_file.parent.parent.name
        img_file = DATASET_ROOT / subset / "images" / (lab_file.stem + ".jpg")
        if not img_file.exists():
            img_file = DATASET_ROOT / subset / "images" / (lab_file.stem + ".png")
        
        if not img_file.exists():
            print(f"⚠️ Skip: Image not found for {lab_file.name}")
            continue

        print(f"🔍 Processing {img_file.name}...")
        img = cv2.imread(str(img_file))
        h, w = img.shape[:2]

        with open(lab_file, 'r') as f:
            lines = f.readlines()

        new_labels = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5: continue
            
            cls = parts[0]
            cx, cy = float(parts[1]), float(parts[2])
            pw, ph = float(parts[3]), float(parts[4])

            # If it's a 0-size box (a click), use SAM
            if pw < 1e-4 and ph < 1e-4:
                # Click points for SAM (x, y)
                points = [[cx * w, cy * h]]
                
                # Predict
                results = model.predict(img, points=points, labels=[1], verbose=False)
                
                if results and len(results[0].masks.xyn) > 0:
                    # Take the first mask polygon
                    poly = results[0].masks.xyn[0]
                    poly_str = " ".join([f"{pt[0]:.6f} {pt[1]:.6f}" for pt in poly])
                    new_labels.append(f"{cls} {poly_str}")
                else:
                    # Fallback to a tiny box if SAM fails
                    new_labels.append(line.strip())
            else:
                # Keep original polygon or box
                new_labels.append(line.strip())

        # Save corrected labels
        with open(lab_file, 'w') as f:
            f.write("\n".join(new_labels))
            
    print("✅ Salvage complete! Points converted to Polygons.")

if __name__ == "__main__":
    salvage()
