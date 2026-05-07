# EXPERIMENTAL - No funciona con VEO panorámico
# El modelo necesita crops de jugadores cercanos
# Revisar cuando tengamos módulo de dorsales (Fase 6)

import cv2
import torch
import os
from pathlib import Path

# Intentar importar YOLO de ultralytics
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

MODEL_PATH = Path("C:/apped/models/tacticzone_jersey.pt")
VIDEO_PATH = Path("C:/apped/data/samples/test_5min.mp4")
OUTPUT_DIR = Path("C:/apped/data/samples")

def test_jersey():
    if not MODEL_PATH.exists():
        print(f"[ERROR] No se encuentra el modelo en {MODEL_PATH}")
        return

    print(f"[INFO] Cargando modelo JERSEY: {MODEL_PATH}")
    
    model = None
    if HAS_ULTRALYTICS:
        try:
            model = YOLO(str(MODEL_PATH))
            print("[OK] Modelo cargado con Ultralytics YOLO.")
        except Exception as e:
            print(f"[INFO] No se pudo cargar con Ultralytics: {e}")
    
    if model is None:
        try:
            model = torch.jit.load(str(MODEL_PATH))
            print("[OK] Modelo cargado con torch.jit (TorchScript).")
        except Exception as e:
            print(f"[ERROR] Fallo total al cargar modelo: {e}")
            return

    if not VIDEO_PATH.exists():
        print(f"[ERROR] No se encuentra el video en {VIDEO_PATH}")
        return

    print(f"[INFO] Procesando video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    
    frame_count = 0
    total_detections = 0
    max_frames = 500  # Analizamos 500 frames para una buena estadística
    
    detected_classes_count = {}
    saved_frames = 0
    frames_to_save = 3

    # Clases esperadas: 0-9
    print(f"[INFO] Clases del modelo: {model.names if hasattr(model, 'names') else 'Desconocidas'}")

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        frame_detections = 0
        current_frame_results = []

        # Ejecutar inferencia
        if hasattr(model, 'predict'):
            results = model.predict(frame, verbose=False, conf=0.1, imgsz=1280)
            for r in results:
                frame_detections += len(r.boxes)
                for box in r.boxes:
                    cls_id = int(box.cls)
                    cls_name = model.names[cls_id] if hasattr(model, 'names') else str(cls_id)
                    detected_classes_count[cls_name] = detected_classes_count.get(cls_name, 0) + 1
                    
                    # Guardar info para dibujo posterior si es necesario
                    b = box.xyxy[0].cpu().numpy()
                    current_frame_results.append((b, cls_name))
        
        total_detections += frame_detections
        
        # Guardar frames con detecciones (intentamos capturar frames con más de 1 detección si es posible)
        if frame_detections > 0 and saved_frames < frames_to_save:
            # Dibujar detecciones en el frame para el usuario
            for b, name in current_frame_results:
                cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
                cv2.putText(frame, name, (int(b[0]), int(b[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            out_path = OUTPUT_DIR / f"jersey_test_{saved_frames+1}.jpg"
            cv2.imwrite(str(out_path), frame)
            print(f"[INFO] Frame guardado con {frame_detections} detecciones: {out_path}")
            saved_frames += 1

        if frame_count % 100 == 0:
            print(f"       Procesados {frame_count} frames... (Detecciones acumuladas: {total_detections})")

    cap.release()
    
    if frame_count > 0:
        avg_det = total_detections / frame_count
        print("\n" + "="*40)
        print("RESULTADOS DE LA PRUEBA JERSEY")
        print("="*40)
        print(f"Frames analizados: {frame_count}")
        print(f"Detecciones totales: {total_detections}")
        print(f"Detecciones promedio por frame: {avg_det:.4f}")
        print("-" * 40)
        print("Clases detectadas (frecuencia en boxes):")
        # Asegurarnos de mostrar 0-9 si existen
        for i in range(10):
            name = str(i)
            # A veces los modelos tienen nombres como '0', '1', etc.
            # O pueden estar en model.names
            if hasattr(model, 'names') and i in model.names:
                name = model.names[i]
            
            count = detected_classes_count.get(name, 0)
            if count > 0:
                print(f"  - Clase {name}: {count}")
            elif name in detected_classes_count:
                 print(f"  - Clase {name}: {detected_classes_count[name]}")
                 
        # Mostrar otras clases si las hay
        for cls, count in detected_classes_count.items():
            if cls not in [str(i) for i in range(10)] and cls not in (model.names.values() if hasattr(model, 'names') else []):
                print(f"  - Clase {cls}: {count}")
                
        print("="*40)
    else:
        print("[ERROR] No se analizaron frames.")

if __name__ == "__main__":
    test_jersey()
