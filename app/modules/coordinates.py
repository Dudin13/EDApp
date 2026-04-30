"""
coordinates.py — Transformación de coordenadas de píxeles a metros (campo real).
Usa la homografía calculada por el detector para mapear posiciones.
"""

import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

class FieldTransformer:
    """
    Transforma puntos de la imagen (píxeles) a coordenadas del campo (metros).
    Dimensiones estándar FIFA: 105m x 68m.
    """
    def __init__(self, pitch_width: float = 105.0, pitch_height: float = 68.0):
        self.pitch_width = pitch_width
        self.pitch_height = pitch_height
        self.H = None  # Matriz de homografía (3x3)
        self.inv_H = None

    def set_homography(self, H: np.ndarray):
        """Establece la matriz de homografía actual."""
        if H is not None and H.shape == (3, 3):
            self.H = H.astype(np.float32)
            try:
                self.inv_H = np.linalg.inv(self.H)
            except np.linalg.LinAlgError:
                self.inv_H = None
        else:
            self.H = None
            self.inv_H = None

    def pixel_to_pitch(self, x: float, y: float) -> tuple[float, float]:
        """
        Transforma un punto (x, y) de píxeles a metros en el campo.
        Si no hay homografía, usa un mapeo proporcional 'naive'.
        """
        if self.H is not None:
            # Perspective transform requiere un array de forma (1, 1, 2)
            pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, self.H)
            px, py = transformed[0][0]
            # Clip para asegurar que estamos dentro del rango razonable del campo
            return (
                float(np.clip(px, 0, self.pitch_width)),
                float(np.clip(py, 0, self.pitch_height))
            )
        else:
            # Fallback naive: asume 1280x720 como base si no hay calibración
            # Esto es mejor que nada, pero poco preciso.
            px = np.clip(x / 1280.0 * self.pitch_width, 0, self.pitch_width)
            py = np.clip(y / 720.0 * self.pitch_height, 0, self.pitch_height)
            return float(px), float(py)

    def pitch_to_pixel(self, x_m: float, y_m: float) -> tuple[float, float]:
        """Transforma metros de vuelta a píxeles (útil para dibujo en radar)."""
        if self.inv_H is not None:
            pt = np.array([[[float(x_m), float(y_m)]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, self.inv_H)
            px, py = transformed[0][0]
            return float(px), float(py)
        else:
            # Fallback naive inverso
            px = x_m / self.pitch_width * 1280.0
            py = y_m / self.pitch_height * 720.0
            return float(px), float(py)
