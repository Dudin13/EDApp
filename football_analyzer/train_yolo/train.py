import os
import torch
import shutil
from ultralytics import YOLO

def train_model():
    base_dir = "c:/apped/football_analyzer/train_yolo"
    data_yaml = os.path.join(base_dir, "data.yaml")
    
    # Seleccionar dispositivo GPU (CUDA) - Crítico para el modelo EDudin
    device = 0 if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("⚠️ ADVERTENCIA: No se detectó GPU. El entrenamiento del modelo EDudin será muy lento.")
    else:
        print(f"🚀 Iniciando entrenamiento del modelo definitivo 'EDudin' en GPU (CUDA)...")
    
    # Priorizar YOLO11 para la arquitectura EDudin
    model_path = "yolo11n-seg.pt" 
    if os.path.exists(os.path.join(base_dir, "runs/segment/EDudin_v1/weights/best.pt")):
        model_path = os.path.join(base_dir, "runs/segment/EDudin_v1/weights/best.pt")
        print(f"Continuando entrenamiento sobre la base EDudin existente...")
    
    model = YOLO(model_path)
    
    model.train(
        data=data_yaml,
        epochs=200, # Entrenamiento profundo para perfección
        imgsz=1024, # Mayor resolución para dorsales y detalles finos
        batch=16 if device != 'cpu' else 2,
        plots=True,
        device=device,
        project=os.path.join(base_dir, "runs/segment"),
        name="EDudin_v1",
        task="segment",
        optimizer='AdamW', 
        lr0=0.001,
        cos_lr=True, 
        patience=50, # No parar hasta que sea perfecto
        save=True,
        cache=True # Usar RAM para mayor velocidad
    )
    
    # Copiar best_EDudin.pt a /modules
    origin_best = os.path.join(base_dir, "runs/segment/EDudin_v1/weights/best.pt")
    target_best = "c:/apped/football_analyzer/modules/EDudin_final.pt"
    if os.path.exists(origin_best):
        print(f"Copiando pesos generados de {origin_best} a {target_best}")
        shutil.copy2(origin_best, target_best)
        print("Entrenamiento completado y pesos aplicados a la IA principal.")

if __name__ == "__main__":
    train_model()
