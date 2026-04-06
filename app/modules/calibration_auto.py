"""
calibration_auto.py
===================
Calibracion automatica del campo de futbol usando keypoint detection.

Integra:
  - roboflow/sports SoccerPitchConfiguration (32 keypoints del campo)
  - roboflow/sports ViewTransformer (homografia robusta)
  - Modelo YOLO entrenado en football-field-detection (Roboflow Universe)

Reemplaza la calibracion manual de calibration_pnl.py con una version
completamente automatica que detecta las lineas del campo en el video.

Uso:
    from modules.calibration_auto import AutoCalibrator

    cal = AutoCalibrator()
    H = cal.calibrate(frame)                    # calcula homografia desde un frame
    pitch_xy = cal.pixel_to_pitch(px, py)       # convierte pixel a coordenadas campo
    frame_ann = cal.draw_keypoints(frame)       # visualiza keypoints detectados
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import os


# ── Configuracion del campo (roboflow/sports) ──────────────────────────────

class SoccerPitchConfiguration:
    """
    Campo de futbol estandar con 32 keypoints.
    Dimensiones en cm (campo estandar 105x68m).
    """
    width:  int = 6800   # cm
    length: int = 10500  # cm

    penalty_box_width:  int = 4100
    penalty_box_length: int = 2015
    goal_box_width:     int = 1832
    goal_box_length:    int = 550
    centre_circle_radius: int = 915
    penalty_spot_distance: int = 1100

    @property
    def vertices(self):
        w = self.width
        l = self.length
        pb_w = self.penalty_box_width
        pb_l = self.penalty_box_length
        gb_w = self.goal_box_width
        gb_l = self.goal_box_length
        cc_r = self.centre_circle_radius
        ps_d = self.penalty_spot_distance

        return [
            (0, 0),
            (0, (w - pb_w) / 2),
            (0, (w - gb_w) / 2),
            (0, (w + gb_w) / 2),
            (0, (w + pb_w) / 2),
            (0, w),
            (gb_l, (w - gb_w) / 2),
            (gb_l, (w + gb_w) / 2),
            (ps_d, w / 2),
            (pb_l, (w - pb_w) / 2),
            (pb_l, (w - gb_w) / 2),
            (pb_l, (w + gb_w) / 2),
            (pb_l, (w + pb_w) / 2),
            (l / 2, 0),
            (l / 2, w / 2 - cc_r),
            (l / 2, w / 2 + cc_r),
            (l / 2, w),
            (l - pb_l, (w - pb_w) / 2),
            (l - pb_l, (w - gb_w) / 2),
            (l - pb_l, (w + gb_w) / 2),
            (l - pb_l, (w + pb_w) / 2),
            (l - ps_d, w / 2),
            (l - gb_l, (w - gb_w) / 2),
            (l - gb_l, (w + gb_w) / 2),
            (l, 0),
            (l, (w - pb_w) / 2),
            (l, (w - gb_w) / 2),
            (l, (w + gb_w) / 2),
            (l, (w + pb_w) / 2),
            (l, w),
            (l / 2 - cc_r, w / 2),
            (l / 2 + cc_r, w / 2),
        ]


class ViewTransformer:
    """Homografia robusta entre dos planos (pixel → campo)."""

    def __init__(self, source: np.ndarray, target: np.ndarray):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m, _ = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
        if self.m is None:
            raise ValueError("No se pudo calcular la homografia.")

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self.m)
        return transformed.reshape(-1, 2).astype(np.float32)

    def transform_point(self, x: float, y: float) -> tuple:
        pts = np.array([[x, y]], dtype=np.float32)
        result = self.transform_points(pts)
        return float(result[0][0]), float(result[0][1])


# ── AutoCalibrator ─────────────────────────────────────────────────────────

class AutoCalibrator:
    """
    Calibrador automatico del campo de futbol.

    Detecta los keypoints del campo en el frame del video usando YOLO,
    y calcula la homografia para mapear pixeles a coordenadas reales.

    Coordenadas de salida en metros (0-105 x, 0-68 y).
    """

    # Dimensiones del campo en metros (para la salida)
    PITCH_WIDTH_M  = 68.0
    PITCH_LENGTH_M = 105.0

    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: Ruta al modelo YOLO de keypoints del campo.
                        Si es None, busca automaticamente en rutas conocidas.
        """
        self.pitch_config = SoccerPitchConfiguration()
        self.transformer:  Optional[ViewTransformer] = None
        self.last_keypoints_px  = []   # keypoints detectados en pixeles
        self.last_keypoints_idx = []   # indices de keypoints detectados
        self.model = None
        self._model_path = self._find_model(model_path)

        if self._model_path:
            self._load_model()
        else:
            print("[AutoCalibrator] Modelo de campo no encontrado — usando calibracion manual")

    def _find_model(self, model_path: Optional[str]) -> Optional[Path]:
        """Busca el modelo YOLO de keypoints del campo."""
        candidates = []

        if model_path:
            candidates.append(Path(model_path))

        base = Path(os.environ.get("APPED_ROOT", "C:/apped"))
        candidates += [
            base / "assets" / "weights" / "football-field-detection.pt",
            base / "assets" / "weights" / "field_keypoints.pt",
            base / "football-field-detection-15" / "football-field-detection" / "train" / "weights" / "best.pt",
            Path("football-field-detection-15") / "football-field-detection" / "train" / "weights" / "best.pt",
        ]

        for p in candidates:
            if p.exists():
                print(f"[AutoCalibrator] Modelo encontrado: {p}")
                return p
        return None

    def _load_model(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self._model_path))
            print(f"[AutoCalibrator] Modelo de campo cargado OK")
        except Exception as e:
            print(f"[AutoCalibrator] Error cargando modelo: {e}")
            self.model = None

    # ── Calibracion automatica ─────────────────────────────────────────────

    def calibrate(self, frame: np.ndarray,
                  min_keypoints: int = 4,
                  conf: float = 0.3) -> bool:
        """
        Detecta keypoints del campo en el frame y calcula la homografia.

        Args:
            frame:          Frame BGR del video
            min_keypoints:  Minimo de keypoints necesarios para calibrar
            conf:           Confianza minima para aceptar un keypoint

        Returns:
            True si la calibracion fue exitosa
        """
        if self.model is None:
            return False

        try:
            results = self.model(frame, conf=conf, verbose=False)
            if not results or results[0].keypoints is None:
                return False

            kps = results[0].keypoints
            if kps.xy is None or len(kps.xy) == 0:
                return False

            # Extraer keypoints detectados con suficiente confianza
            h, w = frame.shape[:2]
            src_pts = []   # puntos en pixeles
            dst_pts = []   # puntos en coordenadas de campo (metros)
            detected_idx = []

            vertices_m = self._vertices_in_meters()

            for i, (x, y) in enumerate(kps.xy[0]):
                if i >= len(vertices_m):
                    break
                conf_val = float(kps.conf[0][i]) if kps.conf is not None else 1.0
                if conf_val < conf:
                    continue
                if x < 1 or y < 1:
                    continue

                src_pts.append([float(x), float(y)])
                dst_pts.append(list(vertices_m[i]))
                detected_idx.append(i)

            if len(src_pts) < min_keypoints:
                return False

            src = np.array(src_pts, dtype=np.float32)
            dst = np.array(dst_pts, dtype=np.float32)

            self.transformer = ViewTransformer(src, dst)
            self.last_keypoints_px  = src_pts
            self.last_keypoints_idx = detected_idx

            print(f"[AutoCalibrator] Calibrado con {len(src_pts)} keypoints")
            return True

        except Exception as e:
            print(f"[AutoCalibrator] Error en calibracion: {e}")
            return False

    def calibrate_manual(self, pixel_points: list, field_points: list) -> bool:
        """
        Calibracion manual — el usuario indica puntos.

        Args:
            pixel_points: Lista de (x, y) en pixeles
            field_points: Lista de (x, y) en metros del campo
        """
        try:
            src = np.array(pixel_points, dtype=np.float32)
            dst = np.array(field_points,  dtype=np.float32)
            self.transformer = ViewTransformer(src, dst)
            return True
        except Exception as e:
            print(f"[AutoCalibrator] Error en calibracion manual: {e}")
            return False

    # ── Transformacion de puntos ───────────────────────────────────────────

    def pixel_to_pitch(self, x: float, y: float) -> tuple:
        """
        Convierte coordenadas de pixel a metros del campo.

        Returns:
            (x_metros, y_metros) o (0, 0) si no hay calibracion
        """
        if self.transformer is None:
            # Fallback naive
            return self._naive_transform(x, y)

        try:
            px, py = self.transformer.transform_point(x, y)
            # Clip a los limites del campo
            px = float(np.clip(px, 0, self.PITCH_LENGTH_M))
            py = float(np.clip(py, 0, self.PITCH_WIDTH_M))
            return px, py
        except Exception:
            return self._naive_transform(x, y)

    def pixels_to_pitch_batch(self, points: list) -> list:
        """Convierte una lista de puntos (x, y) a coordenadas del campo."""
        if not points:
            return []
        if self.transformer is None:
            return [self._naive_transform(x, y) for x, y in points]
        arr = np.array(points, dtype=np.float32)
        result = self.transformer.transform_points(arr)
        return [(float(np.clip(p[0], 0, self.PITCH_LENGTH_M)),
                 float(np.clip(p[1], 0, self.PITCH_WIDTH_M)))
                for p in result]

    def _naive_transform(self, x: float, y: float) -> tuple:
        """Transformacion naive cuando no hay homografia."""
        frame_w = 1920  # asumido
        frame_h = 1080
        px = np.clip(x / frame_w * self.PITCH_LENGTH_M, 0, self.PITCH_LENGTH_M)
        py = np.clip(y / frame_h * self.PITCH_WIDTH_M,  0, self.PITCH_WIDTH_M)
        return float(px), float(py)

    # ── Utilidades visuales ────────────────────────────────────────────────

    def draw_keypoints(self, frame: np.ndarray) -> np.ndarray:
        """Dibuja los keypoints detectados sobre el frame."""
        out = frame.copy()
        for i, (x, y) in enumerate(self.last_keypoints_px):
            idx = self.last_keypoints_idx[i] if i < len(self.last_keypoints_idx) else i
            cv2.circle(out, (int(x), int(y)), 6, (0, 255, 0), -1)
            cv2.putText(out, str(idx + 1), (int(x) + 8, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        return out

    def draw_pitch_minimap(self, detections: list,
                           width: int = 400, height: int = 267) -> np.ndarray:
        """
        Genera un minimapa del campo con las posiciones de los jugadores.

        Args:
            detections: Lista de dicts con 'pitch_pos' (x, y) y 'team'
            width, height: Dimensiones del minimapa en pixeles

        Returns:
            Imagen BGR del minimapa
        """
        minimap = np.zeros((height, width, 3), dtype=np.uint8)
        minimap[:] = (34, 139, 34)  # verde campo

        # Dibujar lineas del campo
        self._draw_field_lines(minimap, width, height)

        # Dibujar jugadores
        team_colors = {
            0: (255, 100, 50),   # equipo A — azul
            1: (50, 100, 255),   # equipo B — rojo
            2: (50, 255, 50),    # arbitro — verde
        }

        for det in detections:
            px, py = det.get("pitch_pos", (0, 0))
            team   = det.get("team", 0)
            # Convertir metros a pixeles del minimapa
            mx = int(px / self.PITCH_LENGTH_M * width)
            my = int(py / self.PITCH_WIDTH_M  * height)
            mx = np.clip(mx, 3, width - 3)
            my = np.clip(my, 3, height - 3)
            color = team_colors.get(team, (180, 180, 180))
            cv2.circle(minimap, (mx, my), 5, color, -1)
            cv2.circle(minimap, (mx, my), 5, (255, 255, 255), 1)

        return minimap

    def _draw_field_lines(self, img: np.ndarray, w: int, h: int):
        """Dibuja las lineas del campo en el minimapa."""
        c = (255, 255, 255)
        t = 1

        def m2p(x_m, y_m):
            return (int(x_m / self.PITCH_LENGTH_M * w),
                    int(y_m / self.PITCH_WIDTH_M  * h))

        L, W = self.PITCH_LENGTH_M, self.PITCH_WIDTH_M
        pb_l = self.pitch_config.penalty_box_length / 100
        pb_w = self.pitch_config.penalty_box_width  / 100
        gb_l = self.pitch_config.goal_box_length    / 100
        gb_w = self.pitch_config.goal_box_width     / 100
        cc_r = self.pitch_config.centre_circle_radius / 100

        # Bordes
        cv2.rectangle(img, m2p(0, 0), m2p(L, W), c, t)
        # Linea central
        cv2.line(img, m2p(L/2, 0), m2p(L/2, W), c, t)
        # Circulo central
        cx, cy = m2p(L/2, W/2)
        rx = int(cc_r / L * w)
        cv2.circle(img, (cx, cy), rx, c, t)
        # Areas penalti
        cv2.rectangle(img, m2p(0, (W-pb_w)/2), m2p(pb_l, (W+pb_w)/2), c, t)
        cv2.rectangle(img, m2p(L-pb_l, (W-pb_w)/2), m2p(L, (W+pb_w)/2), c, t)
        # Areas pequeñas
        cv2.rectangle(img, m2p(0, (W-gb_w)/2), m2p(gb_l, (W+gb_w)/2), c, t)
        cv2.rectangle(img, m2p(L-gb_l, (W-gb_w)/2), m2p(L, (W+gb_w)/2), c, t)

    def _vertices_in_meters(self) -> list:
        """Convierte los vertices del campo de cm a metros."""
        return [(x / 100, y / 100) for x, y in self.pitch_config.vertices]

    @property
    def is_calibrated(self) -> bool:
        return self.transformer is not None

    def get_summary(self) -> dict:
        return {
            "calibrated":       self.is_calibrated,
            "keypoints_detected": len(self.last_keypoints_px),
            "model_loaded":     self.model is not None,
        }
