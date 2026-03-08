"""
detector.py — Motor de detección de jugadores con doble backend:
  - Roboflow (modo por defecto, inmediato)
  - YOLOv8 local (si existe best.pt entrenado)

Incluye clasificación de equipos por color HSV.
"""

import cv2
import numpy as np
import os
import tempfile
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Configuración ──────────────────────────────────────────────────────────
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_WORKSPACE = "roboflow-jvuqo"
ROBOFLOW_PROJECT = "football-players-detection-3zvbc"
ROBOFLOW_VERSION = 1

# Modelo entrenado localmente (YOLOv8 segmentación)
YOLO_MODEL_PATH = Path(__file__).parent / "best_football_seg.pt"
YOLO_COCO_MODEL = "yolov8n.pt"   # modelo base COCO, detecta 'person' sin entrenamiento

VALID_CLASSES = {"player", "goalkeeper", "referee", "ball"}


# ── Carga de modelos ───────────────────────────────────────────────────────

_roboflow_model = None
_yolo_model = None
_yolo_coco_model = None


def _load_roboflow():
    global _roboflow_model
    if _roboflow_model is None:
        from roboflow import Roboflow
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
        _roboflow_model = project.version(ROBOFLOW_VERSION).model
    return _roboflow_model


def _load_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        if YOLO_MODEL_PATH.exists():
            _yolo_model = YOLO(str(YOLO_MODEL_PATH))
        else:
            # Fallback al modelo COCO base
            logger.warning(f"Modelo entrenado no encontrado, usando YOLOv8n COCO: {YOLO_COCO_MODEL}")
            _yolo_model = YOLO(YOLO_COCO_MODEL)
    return _yolo_model


def _load_yolo_coco():
    """Carga YOLOv8n entrenado en COCO (detecta 'person' sin API key)."""
    global _yolo_coco_model
    if _yolo_coco_model is None:
        from ultralytics import YOLO
        _yolo_coco_model = YOLO(YOLO_COCO_MODEL)
    return _yolo_coco_model


def yolo_model_available() -> bool:
    return YOLO_MODEL_PATH.exists()


# ── Detección ──────────────────────────────────────────────────────────────

def detect_frame_roboflow(frame: np.ndarray, confidence: int = 40, overlap: int = 25) -> list:
    """
    Detecta jugadores en un frame usando Roboflow.
    Divide el frame en dos mitades para mejorar detecciones en encuadres amplios.
    Retorna lista de dicts: {x, y, w, h, clase, confianza}
    """
    h_img, w_img = frame.shape[:2]

    # Zona de campo (recortar marcadores y bancos)
    y_min = int(h_img * 0.20)
    y_max = int(h_img * 0.85)
    x_min = int(w_img * 0.01)
    x_max = int(w_img * 0.99)

    zona = frame[y_min:y_max, x_min:x_max]
    h_zona, w_zona = zona.shape[:2]

    mitad_izq = zona[:, :w_zona // 2]
    mitad_der = zona[:, w_zona // 2:]

    # Guardar temporalmente para enviar a Roboflow (API no acepta arrays directamente)
    # Usar tempfile para compatibilidad Windows/Linux/Mac
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

    for pred in result_izq.predictions:
        if pred["class"] in VALID_CLASSES:
            detecciones.append({
                "x": int(pred["x"]) + x_min,
                "y": int(pred["y"]) + y_min,
                "w": int(pred["width"]),
                "h": int(pred["height"]),
                "clase": pred["class"],
                "confianza": round(pred["confidence"], 3),
                "torso_color": get_torso_color(frame, int(pred["x"]) + x_min, int(pred["y"]) + y_min, int(pred["width"]), int(pred["height"]))
            })

    for pred in result_der.predictions:
        if pred["class"] in VALID_CLASSES:
            detecciones.append({
                "x": int(pred["x"]) + w_zona // 2 + x_min,
                "y": int(pred["y"]) + y_min,
                "w": int(pred["width"]),
                "h": int(pred["height"]),
                "clase": pred["class"],
                "confianza": round(pred["confidence"], 3),
            })

    return detecciones


def detect_frame_yolo(frame: np.ndarray, confidence: float = 0.45) -> list:
    """
    Detecta jugadores usando YOLOv8-seg entrenado localmente.
    Extrae tanto el bounding box como el polígono de segmentación (máscara).
    """
    model = _load_yolo()
    h_frame, w_frame = frame.shape[:2]

    # Predecimos con conf=0.10 para capturar el balón sin exceso de falsos positivos
    results = model(frame, conf=0.10, verbose=False)

    # Clases del modelo seg entrenado: {0:Goalkeeper, 1:Player, 2:ball, 3:referee}
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
                # ── Filtros específicos del balón ─────────────────────────
                # 1. Confianza mínima más baja para no perderlo si hay movimiento
                if box_conf < 0.15:
                    continue
                # 2. Aspect ratio: Permitimos un poco más de deformación (motion blur)
                if h > 0 and (h / max(w, 1)) > 3.0:
                    continue
                # 3. Tamaño máximo: un balón pequeño no ocupa más del 6% del ancho del frame
                max_ball_size = int(w_frame * 0.06)
                if w > max_ball_size or h > max_ball_size:
                    continue
                # 4. Zona vertical: ignorar el tercio superior del frame (cielo, postes, vallas)
                if cy < h_frame * 0.30:
                    continue

            elif clase in ("player", "goalkeeper", "referee"):
                # ── Filtros específicos de sujetos ────────────────────────
                # 1. Confianza según slider (por defecto aprox 0.25 en UI)
                if box_conf < confidence:
                    continue
                
                # 2. Filtro de "Posición Background" (feedback usuario: "el foco")
                # En planos generales de fútbol, los jugadores no suelen estar en el 20-25% superior
                # donde están los focos, vallas publicitarias lejanas y cielo.
                if cy < h_frame * 0.22:
                    continue
                
                # 3. Filtro de forma: un jugador suele tener un ratio H/W aprox 2-4.
                # Un poste o foco suele ser extremadamente delgado (ratio > 5-6).
                if h > 0 and (h / max(w, 1)) > 6.5:
                    continue

                # 4. Filtro de tamaño mínimo para evitar ruido en el horizonte
                if w * h < 500:
                    continue

            else:
                if box_conf < confidence:
                    continue

            det = {
                "x": cx, "y": cy,
                "w": w, "h": h,
                "clase": clase,
                "conf": round(float(box.conf[0]), 3),
                "confianza": round(float(box.conf[0]), 3),
                "mask": None,
                "torso_color": get_torso_color(frame, cx, cy, w, h)
            }

            # Si hay máscaras de segmentación, obtener el polígono
            if masks is not None and i < len(masks.xy):
                polygon = masks.xy[i]
                if len(polygon) > 0:
                    det["mask"] = polygon.astype(np.int32)

            detecciones.append(det)

    return detecciones



def detect_frame_coco(frame: np.ndarray, confidence: float = 0.35) -> list:
    """
    Detecta personas usando YOLOv8n entrenado en COCO (clase 0 = person).
    Fallback cuando Roboflow no está disponible.
    No requiere API key ni modelo entrenado propio.
    """
    model = _load_yolo_coco()
    results = model(frame, conf=confidence, classes=[0], verbose=False)  # clase 0 = person

    detecciones = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            # Filtrar por tamaño (evitar detecciones muy pequeñas)
            if w * h < 800:
                continue
            detecciones.append({
                "x": cx, "y": cy,
                "w": w, "h": h,
                "clase": "player",   # COCO solo detecta personas, las tratamos como players
                "confianza": round(float(box.conf[0]), 3),
            })
    return detecciones


def detect_frame(frame: np.ndarray, mode: str = "auto", confidence=40) -> list:
    """
    Fachada principal de detección.
    mode: 'auto' (usa YOLO si disponible, sino Roboflow, sino COCO), 'roboflow', 'yolo'
    """
    if mode == "yolo" or (mode == "auto" and yolo_model_available()):
        return detect_frame_yolo(frame, confidence=confidence / 100 if isinstance(confidence, int) else confidence)

    if mode == "roboflow" or mode == "auto":
        result = detect_frame_roboflow(frame, confidence=int(confidence))
        if result is not None:   # None = error de API
            return result
        # Fallback a COCO si Roboflow falla
        logger.warning("Roboflow no disponible, usando YOLOv8n COCO como fallback")
        return detect_frame_coco(frame, confidence=confidence / 100 if isinstance(confidence, int) else confidence)

    return detect_frame_coco(frame)


# ── Clasificación de equipos ───────────────────────────────────────────────

def get_torso_color(frame: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray | None:
    """Extrae el color HSV medio del torso de un jugador (zona central del bounding box)."""
    x1 = max(0, x - w // 4)
    x2 = min(frame.shape[1], x + w // 4)
    y1 = max(0, y - h // 4)
    y2 = min(frame.shape[0], y + h // 4)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    return np.mean(hsv, axis=(0, 1))


def auto_detect_team_colors(frame: np.ndarray, detections: list) -> dict:
    """
    Detecta automáticamente los 2 colores de equipo usando KMeans sobre
    los colores HSV de todos los jugadores detectados.
    Retorna: {'team_0': (h, s, v), 'team_1': (h, s, v)}
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        return {}

    colors = []
    for det in detections:
        if det.get("clase") in ("player", "goalkeeper"):
            c = get_torso_color(frame, det["x"], det["y"], det["w"], det["h"])
            if c is not None:
                # Filtrar colores que probablemente sean césped (Verde: H 35-85, S > 30)
                h, s, v = c
                if 35 <= h <= 85 and s > 30:
                    continue
                colors.append(c)

    if len(colors) < 4:
        return {}

    colors_arr = np.array(colors, dtype=np.float32)
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    kmeans.fit(colors_arr)
    centers = kmeans.cluster_centers_

    return {
        "team_0": centers[0].tolist(),
        "team_1": centers[1].tolist(),
    }


def classify_team(frame: np.ndarray, det: dict, team_colors: dict = None) -> int:
    """
    Clasifica a qué equipo pertenece un jugador.
    - Si hay team_colors (de auto_detect_team_colors), usa distancia al centroide.
    - Si no, usa reglas HSV por defecto.
    Retorna: 0 (equipo local), 1 (equipo visitante), 2 (árbitro), -1 (descartado)
    """
    # 1. El árbitro suele detectarse por clase primero
    if det.get("clase") == "referee":
        return 2

    # 2. Filtro por tamaño mínimo
    if det["w"] * det["h"] < 500:
        return -1

    color = get_torso_color(frame, det["x"], det["y"], det["w"], det["h"])
    if color is None:
        return 0

    h, s, v = color

    # 3. Detectar árbitro por color (típicamente Rojo, Negro o Flúor)
    # Rojo: H 0-10 o 170-180, S > 100
    if ((0 <= h <= 10 or 170 <= h <= 180) and s > 120):
        return 2
    # Negro/Muy oscuro: V < 50
    if v < 50:
        return 2

    # 4. Clasificar entre equipos usando KMeans si está disponible
    if team_colors and "team_0" in team_colors and "team_1" in team_colors:
        c0 = np.array(team_colors["team_0"])
        c1 = np.array(team_colors["team_1"])
        
        # Pesamos el Hue más que S y V para equipos con colores distintos
        # pero para Blanco vs Otro, Saturation es clave.
        dist0 = np.linalg.norm(color - c0)
        dist1 = np.linalg.norm(color - c1)
        return 0 if dist0 < dist1 else 1

    # 5. Reglas de fallback (Amarillo vs Blanco/Verde)
    # Amarillo
    if 20 <= h <= 45 and s > 70:
        return 0
    # Blanco o Verde
    if s < 70 or (50 <= h <= 95):
        return 1
        
    return 0
