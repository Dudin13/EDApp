import os
from pathlib import Path
from ultralytics import YOLO
import torch

# Configuración
BASE_DIR = Path("C:/apped")
IMAGE_DIR = BASE_DIR / "data/datasets/veo_frames_raw/images"
LABEL_DIR = BASE_DIR / "data/datasets/veo_frames_raw/labels"
MODEL_PATH = BASE_DIR / "models/players.pt"
CONF_THRESHOLD = 0.25
IMGSZ = 1280

def main():
    print("Starting Auto-Labeling...")
    
    # 1. Load model
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    
    # Asegurar que la carpeta de labels exista
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. Listar imágenes
    images = sorted(list(IMAGE_DIR.glob("mediacoach_j35_*.jpg")) + list(IMAGE_DIR.glob("mediacoach_j35_*.png")))
    total_images = len(images)
    print(f"Images found: {total_images}")
    
    processed_count = 0
    labels_generated = 0
    empty_images = 0
    
    # 3. Process
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    for img_path in images:
        # Inferencia
        results = model.predict(source=str(img_path), conf=CONF_THRESHOLD, imgsz=IMGSZ, device=device, verbose=False)[0]
        
        # Preparar archivo de etiqueta
        label_path = LABEL_DIR / (img_path.stem + ".txt")
        
        yolo_lines = []
        for box in results.boxes:
            cls = int(box.cls[0])
            # YOLO format: cls cx cy w h (normalized)
            xywhn = box.xywhn[0].tolist()
            line = f"{cls} {' '.join(f'{x:.6f}' for x in xywhn)}"
            yolo_lines.append(line)
        
        # Guardar (incluso si está vacío, o según prefieras - el usuario pide generar .txt)
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))
        
        processed_count += 1
        labels_generated += len(yolo_lines)
        if len(yolo_lines) == 0:
            empty_images += 1
            
        if processed_count % 100 == 0:
            print(f"Processed {processed_count}/{total_images}...")

    # 4. Summary
    print("\n" + "="*40)
    print("AUTO-LABELING SUMMARY")
    print("="*40)
    print(f"Total images processed: {processed_count}")
    print(f"Total labels (objects) generated: {labels_generated}")
    print(f"Images without detections: {empty_images}")
    print("="*40)

if __name__ == "__main__":
    main()
