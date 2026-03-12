import cv2
import numpy as np
import threading
from pathlib import Path
from collections import Counter
from sklearn.cluster import KMeans
from typing import List, Dict, Optional, Tuple

from core.config.settings import settings
from core.logger import logger
from ai.calibration.calibration_pnl import PnLCalibrator
from ai.identity.identity_reader import IdentityReader

class DetectorEngine:
    """
    Motor de detección especializado para fútbol.
    Gestiona modelos YOLO para jugadores, balón y campo.
    """
    def __init__(self):
        self._player_model = None
        self._ball_model = None
        self._pitch_model = None
        self._coco_model = None
        
        self._player_lock = threading.Lock()
        self._ball_lock = threading.Lock()
        self._pitch_lock = threading.Lock()
        self._coco_lock = threading.Lock()
        
        self.calibrator = PnLCalibrator()
        self.id_reader = IdentityReader()
        
        self.BALL_ID = 0
        self.GOALKEEPER_ID = 1
        self.PLAYER_ID = 2
        self.REFEREE_ID = 3
        self.ID_TO_NAME = {
            self.BALL_ID: "ball",
            self.GOALKEEPER_ID: "goalkeeper",
            self.PLAYER_ID: "player",
            self.REFEREE_ID: "referee"
        }

    def _load_model(self, model_attr: str, model_path: str, lock: threading.Lock, name: str):
        with lock:
            if getattr(self, model_attr) is None:
                from ultralytics import YOLO
                path = Path(model_path)
                if path.exists():
                    setattr(self, model_attr, YOLO(str(path)))
                    logger.info(f"Modelo {name} cargado desde {path.name}")
                else:
                    logger.warning(f"Modelo {name} no encontrado en {model_path}")
        return getattr(self, model_attr)

    @property
    def player_model(self):
        return self._load_model("_player_model", settings.PLAYER_MODEL_PATH, self._player_lock, "jugadores")

    @property
    def ball_model(self):
        return self._load_model("_ball_model", settings.BALL_MODEL_PATH, self._ball_lock, "balón")

    @property
    def pitch_model(self):
        return self._load_model("_pitch_model", settings.PITCH_MODEL_PATH, self._pitch_lock, "campo")

    def _build_detection_dict(self, frame, x_abs, y_abs, w, h, clase, confidence, extract_dorsal=True):
        dorsal_num = None
        if extract_dorsal and clase in ("player", "goalkeeper"):
            cx, cy = x_abs, y_abs
            x1c = max(0, cx - w // 4); y1c = max(0, cy - h // 3)
            x2c = min(frame.shape[1], cx + w // 4); y2c = min(frame.shape[0], cy + h // 6)
            roi = frame[y1c:y2c, x1c:x2c]
            if roi.size > 0:
                dorsal_num, _ = self.id_reader.extract_dorsal(roi)
        
        return {
            "x": int(x_abs), "y": int(y_abs), "w": int(w), "h": int(h),
            "clase": clase, "confianza": round(confidence, 3), "conf": round(confidence, 3),
            "torso_color": self.extract_torso_rgb(frame, int(x_abs), int(y_abs), int(w), int(h)),
            "pitch_coords": self.calibrator.transform_point(int(x_abs), int(y_abs)),
            "dorsal": dorsal_num, "mask": None,
        }

    def detect_frame(self, frame, confidence=0.35):
        import supervision as sv
        h_frame, w_frame = frame.shape[:2]
        detecciones = []

        # 1. Jugadores / Porteros / Árbitros
        pm = self.player_model
        if pm:
            try:
                r = pm.predict(frame, conf=confidence, verbose=False, device=settings.DEVICE)[0]
                d = sv.Detections.from_ultralytics(r).with_nms(threshold=0.5, class_agnostic=True)
                for i, xyxy in enumerate(d.xyxy):
                    x1, y1, x2, y2 = map(int, xyxy)
                    w = x2 - x1; h = y2 - y1; cx = (x1 + x2) // 2; cy = (y1 + y2) // 2
                    cls_id = int(d.class_id[i]) if d.class_id is not None else self.PLAYER_ID
                    conf_v = float(d.confidence[i]) if d.confidence is not None else confidence
                    clase = self.ID_TO_NAME.get(cls_id, "player")
                    
                    if clase == "ball": continue
                    if cy < h_frame * 0.15 or w * h < 400: continue
                    if h > 0 and (h / max(w, 1)) > 7.0: continue
                    
                    detecciones.append(self._build_detection_dict(frame, cx, cy, w, h, clase, conf_v))
            except Exception as e:
                logger.error(f"Error en detección de jugadores: {e}")

        # 2. Balón (1280px para mayor precisión en VEO)
        bm = self.ball_model
        if bm:
            try:
                scale = 1280 / w_frame
                fhd = cv2.resize(frame, (1280, int(h_frame * scale)))
                rb = bm.predict(fhd, conf=0.1, verbose=False, imgsz=1280, device=settings.DEVICE)[0]
                db = sv.Detections.from_ultralytics(rb).with_nms(threshold=0.3, class_agnostic=True)
                for i, xyxy in enumerate(db.xyxy):
                    x1, y1, x2, y2 = map(int, xyxy)
                    x1 = int(x1 / scale); y1 = int(y1 / scale)
                    x2 = int(x2 / scale); y2 = int(y2 / scale)
                    w = x2 - x1; h = y2 - y1; cx = (x1 + x2) // 2; cy = (y1 + y2) // 2
                    cls_id = int(db.class_id[i]) if db.class_id is not None else self.BALL_ID
                    conf_v = float(db.confidence[i]) if db.confidence is not None else 0.1
                    
                    if self.ID_TO_NAME.get(cls_id, "ball") != "ball": continue
                    if h > 0 and (h / max(w, 1)) > 3.0: continue
                    if w > int(w_frame * 0.07) or cy < h_frame * 0.15: continue
                    
                    detecciones.append(self._build_detection_dict(frame, cx, cy, max(w, 20), max(h, 20), "ball", conf_v, extract_dorsal=False))
            except Exception as e:
                logger.error(f"Error en detección de balón: {e}")

        return detecciones

    def extract_torso_rgb(self, frame, x, y, w, h):
        x1 = max(0, int(x - w * 0.25)); x2 = min(frame.shape[1], int(x + w * 0.25))
        y1 = max(0, int(y - h * 0.3)); y2 = min(frame.shape[0], int(y + h * 0.1))
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 2 or roi.shape[1] < 2: return None
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pixels = roi_rgb.reshape(-1, 3)
        if len(pixels) < 4: return None
        km = KMeans(n_clusters=2, random_state=42, n_init=3)
        labels = km.fit_predict(pixels)
        counts = Counter(labels)
        dom = counts.most_common(1)[0][0]
        dc = km.cluster_centers_[dom]
        r, g, b = dc
        is_green = (g > r + 30 and g > b + 30)
        if is_green and len(counts) > 1:
            dc = km.cluster_centers_[counts.most_common(2)[1][0]]
        return tuple(map(int, dc))

    def detect_pitch_homography(self, frame):
        pm = self.pitch_model
        if not pm: return None
        try:
            import supervision as sv
            from sports.configs.soccer import SoccerPitchConfiguration
            CONFIG = SoccerPitchConfiguration()
            result = pm.predict(frame, conf=0.3, verbose=False, device=settings.DEVICE)[0]
            kp = sv.KeyPoints.from_ultralytics(result)
            if not kp or len(kp.xy[0]) < 4: return None
            
            mask = kp.confidence[0] > 0.5
            frame_pts = kp.xy[0][mask]
            pitch_pts = np.array(CONFIG.vertices)[mask]
            if len(frame_pts) < 4: return None
            H, _ = cv2.findHomography(pitch_pts.astype(np.float32), frame_pts.astype(np.float32), cv2.RANSAC, 5.0)
            return H
        except Exception as e:
            logger.error(f"Error en homografía automática: {e}")
            return None
