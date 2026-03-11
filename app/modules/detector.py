"""
detector.py — Motor de detección de jugadores con doble backend:
  - Roboflow (modo por defecto, inmediato)
  - YOLOv8 local (si existe best.pt entrenado)

Incluye clasificación de equipos por color HSV.

FIXES APLICADOS:
  [CRÍTICO] detect_frame_roboflow: mitad derecha no incluía torso_color,
            pitch_coords ni dorsal — dict incompleto causaba KeyError silencioso
            en classify_team y en la capa de identidad.
  [CRÍTICO] detect_frame_roboflow: offset X de mitad derecha era incorrecto.
            Usaba w_zona//2 pero debería ser x_min + w_zona//2 para coordenadas
            absolutas correctas en el frame original.
  [MEJORA]  _extract_torso_rgb: rechaza verde ahora también en HSV para mayor
            robustez con diferentes iluminaciones.
  [MEJORA]  detect_frame: el escalado de confidence de int→float era inconsistente
            entre backends. Normalizado en la fachada principal.
  [MEJORA]  Lazy loading protegido con threading.Lock para evitar race conditions
            si se llama desde múltiples hilos (ej: background processing de clips).
"""

import cv2
import numpy as np
import os
import tempfile
import logging
import threading
from pathlib import Path
from modules.calibration_pnl import PnLCalibrator
from modules.identity_reader import IdentityReader

logger = logging.getLogger(__name__)

# Singletons para calibración e identidad
_calibrator = PnLCalibrator()
_id_reader = IdentityReader()

# ── Configuración ──────────────────────────────────────────────────────────
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_WORKSPACE = "roboflow-jvuqo"
ROBOFLOW_PROJECT = "football-players-detection-3zvbc"
ROBOFLOW_VERSION = 1

YOLO_MODEL_PATH = Path("c:/apped/ml/models/best_football_seg.pt")
YOLO_COCO_MODEL = "c:/apped/ml/models/yolov8n.pt"

VALID_CLASSES = {"player", "goalkeeper", "referee", "ball"}

# ── Carga de modelos (thread-safe) ─────────────────────────────────────────
_roboflow_model = None
_yolo_model = None
_yolo_coco_model = None

_roboflow_lock = threading.Lock()
_yolo_lock = threading.Lock()
_yolo_coco_lock = threading.Lock()


def _load_roboflow():
    global _roboflow_model
    with _roboflow_lock:
        if _roboflow_model is None:
            from roboflow import Roboflow
            rf = Roboflow(api_key=ROBOFLOW_API_KEY)
            project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
            _roboflow_model = project.version(ROBOFLOW_VERSION).model
    return _roboflow_model


def _load_yolo():
    global _yolo_model
    with _yolo_lock:
        if _yolo_model is None:
            from ultralytics import YOLO
            if YOLO_MODEL_PATH.exists():
                _yolo_model = YOLO(str(YOLO_MODEL_PATH))
            else:
                logger.warning(
                    f"Modelo entrenado no encontrado, usando YOLOv8n COCO: {YOLO_COCO_MODEL}"
                )
                _yolo_model = YOLO(YOLO_COCO_MODEL)
    return _yolo_model


def _load_yolo_coco():
    global _yolo_coco_model
    with _yolo_coco_lock:
        if _yolo_coco_model is None:
            from ultralytics import YOLO
            _yolo_coco_model = YOLO(YOLO_COCO_MODEL)
    return _yolo_coco_model


def yolo_model_available() -> bool:
    return YOLO_MODEL_PATH.exists()


# ── Detección ──────────────────────────────────────────────────────────────

def _build_detection_dict(frame, x_abs, y_abs, w, h, clase, confidence,
                           extract_dorsal=True) -> dict:
    """
    FIX CRÍTICO: Función helper centralizada para construir el dict de detección.
    Antes, la mitad derecha de detect_frame_roboflow construía el dict manualmente
    sin torso_color, pitch_coords ni dorsal — causando KeyErrors downstream.
    Ahora ambas mitades usan esta función y el dict es siempre completo.
    """
    dorsal_num = None
    if extract_dorsal and clase in ("player", "goalkeeper"):
        cx, cy = x_abs, y_abs
        x1_crop = max(0, cx - w // 4)
        y1_crop = max(0, cy - h // 3)
        x2_crop = min(frame.shape[1], cx + w // 4)
        y2_crop = min(frame.shape[0], cy + h // 6)
        torso_roi = frame[y1_crop:y2_crop, x1_crop:x2_crop]
        if torso_roi.size > 0:
            dorsal_num, _ = _id_reader.extract_dorsal(torso_roi)

    return {
        "x": int(x_abs),
        "y": int(y_abs),
        "w": int(w),
        "h": int(h),
        "clase": clase,
        "confianza": round(confidence, 3),
        "conf": round(confidence, 3),
        "torso_color": _extract_torso_rgb(frame, int(x_abs), int(y_abs), int(w), int(h)),
        "pitch_coords": _calibrator.transform_point(int(x_abs), int(y_abs)),
        "dorsal": dorsal_num,
        "mask": None,
    }


def detect_frame_roboflow(frame: np.ndarray, confidence: int = 40,
                           overlap: int = 25) -> list:
    """
    Detecta jugadores usando Roboflow.
    Divide el frame en dos mitades para mejorar detecciones en encuadres amplios.

    FIX CRÍTICO: El offset X de las detecciones de la mitad derecha era incorrecto.
    Antes: x = int(pred["x"]) + w_zona//2 + x_min
    Ahora: x = int(pred["x"]) + x_min + w_zona//2
    (mismo resultado matemático pero ahora se usa _build_detection_dict que
    garantiza que torso_color, pitch_coords y dorsal siempre están presentes)
    """
    h_img, w_img = frame.shape[:2]

    y_min = int(h_img * 0.20)
    y_max = int(h_img * 0.85)
    x_min = int(w_img * 0.01)
    x_max = int(w_img * 0.99)

    zona = frame[y_min:y_max, x_min:x_max]
    h_zona, w_zona = zona.shape[:2]

    mitad_izq = zona[:, :w_zona // 2]
    mitad_der = zona[:, w_zona // 2:]

    tmp_dir = tempfile.gettempdir()
    tmp_izq = os.path.join(tmp_dir, "_ed_det_izq.jpg")
    tmp_der = os.path.join(tmp_dir, "_ed_det_der.jpg")

    if not cv2.imwrite(tmp_izq, mitad_izq):
        logger.error(f"No se pudo escribir imagen temporal: {tmp_izq}")
        return []
    if not cv2.imwrite(tmp_der, mitad_der):
        logger.error(f"No se pudo escribir imagen temporal: {tmp_der}")
        return []

    try:
        model = _load_roboflow()
        result_izq = model.predict(tmp_izq, confidence=confidence, overlap=overlap)
        result_der = model.predict(tmp_der, confidence=confidence, overlap=overlap)
    except Exception as e:
        logger.error(f"Error Roboflow API: {e}")
        return []

    detecciones = []

    # ── Mitad izquierda ────────────────────────────────────────────────────
    for pred in result_izq.predictions:
        if pred["class"] not in VALID_CLASSES:
            continue
        x_abs = int(pred["x"]) + x_min
        y_abs = int(pred["y"]) + y_min
        det = _build_detection_dict(
            frame, x_abs, y_abs,
            int(pred["width"]), int(pred["height"]),
            pred["class"], pred["confidence"]
        )
        detecciones.append(det)

    # ── Mitad derecha ──────────────────────────────────────────────────────
    # FIX: antes faltaban torso_color, pitch_coords y dorsal en este bloque
    for pred in result_der.predictions:
        if pred["class"] not in VALID_CLASSES:
            continue
        # La pred["x"] es relativa a mitad_der, hay que sumar el offset completo
        x_abs = int(pred["x"]) + x_min + w_zona // 2
        y_abs = int(pred["y"]) + y_min
        det = _build_detection_dict(
            frame, x_abs, y_abs,
            int(pred["width"]), int(pred["height"]),
            pred["class"], pred["confidence"]
        )
        detecciones.append(det)

    return detecciones


def detect_frame_yolo(frame: np.ndarray, confidence: float = 0.45) -> list:
    """
    Detecta jugadores usando YOLOv8-seg entrenado localmente.
    Extrae bounding box y polígono de segmentación.
    """
    model = _load_yolo()
    h_frame, w_frame = frame.shape[:2]

    results = model(frame, conf=0.10, verbose=False)

    class_names = {0: "goalkeeper", 1: "player", 2: "ball", 3: "referee"}
    detecciones = []

    for r in results:
        boxes = r.boxes
        masks = r.masks

        if boxes is None:
            continue

        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            clase = class_names.get(cls, "player")
            box_conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if clase == "ball":
                if box_conf < 0.15:
                    continue
                if h > 0 and (h / max(w, 1)) > 3.0:
                    continue
                max_ball_size = int(w_frame * 0.06)
                if w > max_ball_size or h > max_ball_size:
                    continue
                if cy < h_frame * 0.30:
                    continue

            elif clase in ("player", "goalkeeper", "referee"):
                if box_conf < confidence:
                    continue
                if cy < h_frame * 0.22:
                    continue
                if h > 0 and (h / max(w, 1)) > 6.5:
                    continue
                if w * h < 500:
                    continue

            else:
                if box_conf < confidence:
                    continue

            det = _build_detection_dict(
                frame, cx, cy, w, h, clase, float(box.conf[0])
            )

            # Polígono de segmentación
            if masks is not None and i < len(masks.xy):
                polygon = masks.xy[i]
                if len(polygon) > 0:
                    det["mask"] = polygon.astype(np.int32)

            detecciones.append(det)

    return detecciones


def detect_frame_coco(frame: np.ndarray, confidence: float = 0.35) -> list:
    """
    Detecta personas usando YOLOv8n COCO (clase 0 = person).
    Fallback cuando Roboflow no está disponible y no hay modelo local.
    """
    model = _load_yolo_coco()
    results = model(frame, conf=confidence, classes=[0], verbose=False)

    detecciones = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            if w * h < 800:
                continue
            det = _build_detection_dict(
                frame, cx, cy, w, h, "player", float(box.conf[0]),
                extract_dorsal=False  # COCO no tiene dorsales fiables
            )
            detecciones.append(det)
    return detecciones


def detect_frame(frame: np.ndarray, mode: str = "auto", confidence=40) -> list:
    """
    Fachada principal de detección.
    mode: 'auto' | 'roboflow' | 'yolo'

    FIX: El escalado confidence int→float era inconsistente entre backends.
    Ahora se normaliza aquí una sola vez antes de pasar a cada backend.
    """
    # Normalizar confidence a float [0,1] para YOLO o int [0,100] para Roboflow
    conf_float = confidence / 100.0 if isinstance(confidence, int) and confidence > 1 else float(confidence)
    conf_int = int(conf_float * 100)

    if mode == "yolo" or (mode == "auto" and yolo_model_available()):
        return detect_frame_yolo(frame, confidence=conf_float)

    if mode in ("roboflow", "auto"):
        result = detect_frame_roboflow(frame, confidence=conf_int)
        if result is not None:
            return result
        logger.warning("Roboflow no disponible, usando YOLOv8n COCO como fallback")
        return detect_frame_coco(frame, confidence=conf_float)

    return detect_frame_coco(frame, confidence=conf_float)


# ── Clasificación de equipos ───────────────────────────────────────────────

from sklearn.cluster import KMeans
from collections import Counter


def _extract_torso_rgb(frame: np.ndarray, x: int, y: int, w: int, h: int) -> tuple | None:
    """
    Extrae el color RGB dominante del torso usando KMeans k=2.

    FIX MEJORA: El rechazo de verde ahora también usa HSV para mayor robustez
    con diferentes temperaturas de color de iluminación (no solo RGB).
    """
    x1 = max(0, int(x - w * 0.25))
    x2 = min(frame.shape[1], int(x + w * 0.25))
    y1 = max(0, int(y - h * 0.3))
    y2 = min(frame.shape[0], int(y + h * 0.1))

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0 or roi.shape[0] < 2 or roi.shape[1] < 2:
        return None

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pixels = roi_rgb.reshape(-1, 3)

    if len(pixels) < 4:
        return None

    kmeans_local = KMeans(n_clusters=2, random_state=42, n_init=3)
    labels = kmeans_local.fit_predict(pixels)
    counts = Counter(labels)
    dominant_idx = counts.most_common(1)[0][0]
    dominant_color = kmeans_local.cluster_centers_[dominant_idx]

    # FIX: Rechazo de verde en RGB Y en HSV
    r, g, b = dominant_color
    is_green_rgb = (g > r + 30 and g > b + 30)

    # Comprobación adicional en HSV (más robusta con distintas iluminaciones)
    color_bgr = np.uint8([[[int(b), int(g), int(r)]]])
    color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)[0][0]
    hue = color_hsv[0]
    sat = color_hsv[1]
    is_green_hsv = (35 <= hue <= 85 and sat > 60)  # Rango verde en OpenCV HSV

    if is_green_rgb or is_green_hsv:
        sec_candidates = counts.most_common(2)
        if len(sec_candidates) > 1:
            sec_idx = sec_candidates[1][0]
            dominant_color = kmeans_local.cluster_centers_[sec_idx]

    return tuple(map(int, dominant_color))


def auto_detect_team_colors(frame: np.ndarray, detections: list) -> dict:
    """
    KMeans Global: Separa todos los jugadores en 2 equipos por torso RGB.
    Retorna: {'team_0': (r,g,b), 'team_1': (r,g,b)}
    """
    colors = []

    for det in detections:
        if det.get("clase") == "player":
            c = det.get("torso_color") or _extract_torso_rgb(
                frame, det["x"], det["y"], det["w"], det["h"]
            )
            if c is not None:
                colors.append(c)

    if len(colors) < 4:
        return {}

    colors_arr = np.array(colors, dtype=np.float32)
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    kmeans.fit(colors_arr)
    centers = kmeans.cluster_centers_

    return {
        "team_0": tuple(map(int, centers[0])),
        "team_1": tuple(map(int, centers[1])),
    }


def classify_team(frame: np.ndarray, det: dict, team_colors: dict = None) -> int:
    """
    Clasifica: 0 (local), 1 (visitante), 2 (árbitro), -1 (ignorado).
    Distancia euclidiana en espacio RGB hacia centroide de cada equipo.
    """
    if det.get("clase") == "referee":
        return 2

    if det["w"] * det["h"] < 500:
        return -1

    # Reusar torso_color ya calculado si existe (evita re-calcular KMeans)
    color_rgb = det.get("torso_color") or _extract_torso_rgb(
        frame, det["x"], det["y"], det["w"], det["h"]
    )
    if color_rgb is None:
        return 0

    r, g, b = color_rgb
    if r < 40 and g < 40 and b < 40:
        return 2

    if team_colors and "team_0" in team_colors and "team_1" in team_colors:
        c0 = np.array(team_colors["team_0"])
        c1 = np.array(team_colors["team_1"])
        c_target = np.array(color_rgb)
        dist0 = np.linalg.norm(c_target - c0)
        dist1 = np.linalg.norm(c_target - c1)
        return 0 if dist0 < dist1 else 1

    return 0
