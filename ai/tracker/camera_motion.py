import cv2
import numpy as np
from core.logger import logger

class CameraMotionEstimator:
    """
    Estima el movimiento global de la cámara usando Optical Flow (Lucas-Kanade).
    Sigue puntos característicos del fondo para deducir el movimiento entre frames.
    """
    def __init__(self):
        self.prev_gray = None
        self.prev_points = None
        # Parámetros para Shi-Tomasi (Lucas-Kanade)
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        # Parámetros para detección de esquinas (Shi-Tomasi)
        self.feature_params = dict(
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=30,
            blockSize=3
        )

    def compute_offset(self, frame: np.ndarray) -> tuple:
        """
        Calcula el desplazamiento (dx, dy) de la imagen actual respecto al frame anterior.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            return (0.0, 0.0)

        if self.prev_points is None or len(self.prev_points) < 10:
            self.prev_points = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
            if self.prev_points is None:
                self.prev_gray = gray
                return (0.0, 0.0)

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None, **self.lk_params
        )

        if next_points is None or status is None:
            self.prev_gray = gray
            self.prev_points = None
            return (0.0, 0.0)

        good_new = next_points[status == 1]
        good_old = self.prev_points[status == 1]

        if len(good_new) < 5:
            self.prev_gray = gray
            self.prev_points = None
            return (0.0, 0.0)

        diff = good_new - good_old
        dx = np.median(diff[:, 0])
        dy = np.median(diff[:, 1])

        self.prev_gray = gray.copy()
        self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)

        return (float(dx), float(dy))
