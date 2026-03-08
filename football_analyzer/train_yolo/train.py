import os
import torch
import shutil
from ultralytics import YOLO

def train_model():
    base_dir = "c:/apped/football_analyzer/train_yolo"
    data_yaml = os.path.join(base_dir, "data.yaml")
    
    # Seleccionar dispositivo automáticamente
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Iniciando entrenamiento en: {device}")
    
    model_path = os.path.join(base_dir, "runs/segment/pro_training/weights/best.pt")
    if not os.path.exists(model_path):
        print("Pre-trained 'best.pt' for segmentation not found, using base YOLOV8n-seg")
        model_path = "yolov8n-seg.pt" 
    
    model = YOLO(model_path)
    
    model.train(
        data=data_yaml,
        epochs=30 if device == 'cpu' else 100,
        imgsz=640,
        batch=4 if device == 'cpu' else 16,
        plots=True,
        device=device,
        project=os.path.join(base_dir, "runs/segment"),
        name="pro_training",
        task="segment"
    )
    
    # Copiar best_football_seg.pt a /modules donde lo espera el detector
    origin_best = os.path.join(base_dir, "runs/segment/pro_training/weights/best.pt")
    target_best = "c:/apped/football_analyzer/modules/best_football_seg.pt"
    if os.path.exists(origin_best):
        print(f"Copiando pesos generados de {origin_best} a {target_best}")
        shutil.copy2(origin_best, target_best)
        print("Entrenamiento completado y pesos aplicados a la IA principal.")

if __name__ == "__main__":
    train_model()
