
import os
import torch
from ultralytics import YOLO

def train_model():
    base_dir = "c:/apped/football_analyzer/train_yolo"
    data_yaml = os.path.join(base_dir, "data.yaml")
    
    # Seleccionar dispositivo automáticamente
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Iniciando entrenamiento en: {device}")
    
    model_path = os.path.join(base_dir, "runs/detect/train/weights/best.pt")
    if not os.path.exists(model_path):
        print("Pre-trained 'best.pt' not found, using base YOLOV8n")
        model_path = "yolov8n.pt" 
    
    model = YOLO(model_path)
    
    model.train(
        data=data_yaml,
        epochs=30 if device == 'cpu' else 100, # Menos si es CPU para que termine
        imgsz=640,
        batch=4 if device == 'cpu' else 16,     # Batch pequeño para CPU
        plots=True,
        device=device,
        project=os.path.join(base_dir, "runs/detect"),
        name="pro_training"
    )

if __name__ == "__main__":
    train_model()
