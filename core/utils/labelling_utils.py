import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO, SAM
from core.config.settings import settings
from core.logger import logger

class LabellingEngine:
    def __init__(self):
        self.yolo_model = YOLO(settings.PLAYER_MODEL_PATH)
        self.sam_model = None # Lazy load
        self.classes = ["Team 1", "Team 2", "Goalkeeper", "Referee", "Ball"]

    def _get_sam(self):
        if self.sam_model is None:
            logger.info("Cargando SAM para Magic Click...")
            self.sam_model = SAM("sam2_b.pt") 
        return self.sam_model

    def predict_yolo(self, image_path):
        """Predice usando YOLO y devuelve polígonos/boxes en formato normalizado."""
        results = self.yolo_model.predict(image_path, conf=0.25, verbose=False)[0]
        detections = []
        
        if results.masks is not None:
            for mask, cls in zip(results.masks.xyn, results.boxes.cls):
                detections.append({
                    "cls": int(cls),
                    "points": mask.tolist()
                })
        else:
            # Fallback a boxes si no hay máscaras
            for box, cls in zip(results.boxes.xyxyn, results.boxes.cls):
                x1, y1, x2, y2 = box.tolist()
                detections.append({
                    "cls": int(cls),
                    "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                })
        return detections

    def predict_sam(self, image_rgb, points):
        """Aplica SAM basado en puntos (clicks) del usuario."""
        sam = self._get_sam()
        h, w = image_rgb.shape[:2]
        # Puntos deben estar en coordenadas de imagen reales
        input_points = [[p[0] * w, p[1] * h] for p in points]
        
        results = sam.predict(image_rgb, points=input_points, labels=[1]*len(input_points), verbose=False)
        if results and len(results[0].masks.xyn) > 0:
            return results[0].masks.xyn[0].tolist()
        return None

    def save_yolo_labels(self, labels, output_path):
        """Guarda etiquetas en formato YOLO (class x1 y1 x2 y2 ...)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        lines = []
        for label in labels:
            cls = label["cls"]
            pts = " ".join([f"{p[0]:.6f} {p[1]:.6f}" for p in label["points"]])
            lines.append(f"{cls} {pts}")
            
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Etiquetas guardadas en {output_path}")

labelling_engine = LabellingEngine()
