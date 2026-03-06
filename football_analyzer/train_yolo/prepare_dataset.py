
import cv2
import os
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def extract_and_auto_label(video_folder, output_base_folder, model_path="best.pt"):
    input_path = Path(video_folder)
    output_path = Path(output_base_folder)
    images_out = output_path / "images"
    labels_out = output_path / "labels"
    
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    
    # Cargar modelo para auto-etiquetado
    model = YOLO(model_path) if os.path.exists(model_path) else YOLO("yolov8n.pt")
    
    video_files = list(input_path.glob("*.mp4"))
    
    for v_file in video_files:
        print(f"Procesando: {v_file.name}")
        cap = cv2.VideoCapture(str(v_file))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Extraer un frame cada 1000 para diversidad (y no llenar el disco)
        for i in range(0, total_frames, 1000):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret: break
            
            frame_name = f"{v_file.stem}_f{i}"
            img_file = images_out / f"{frame_name}.jpg"
            cv2.imwrite(str(img_file), frame)
            
            # Auto-etiquetado
            results = model(frame, conf=0.25, verbose=False)
            
            with open(labels_out / f"{frame_name}.txt", "w") as f:
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        # YOLO format: cls x_center y_center width height (normalized)
                        x, y, w, h = box.xywhn[0]
                        f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        
        cap.release()
    print("Extracción y auto-etiquetado completado.")

if __name__ == "__main__":
    # Ajustar rutas según el entorno del usuario
    extract_and_auto_label(
        r"C:\Users\Usuario\Desktop\VideosDePrueba",
        r"c:\apped\football_analyzer\dataset_prep"
    )
