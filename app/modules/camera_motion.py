import cv2
import numpy as np

class CameraMotionEstimator:
    """
    Estima el movimiento global de la cámara usando Optical Flow (Lucas-Kanade).
    Puede recibir keypoints del campo fijos (ej. de pitch.pt) para un tracking perfecto
    del entorno sin contaminación de jugadores. Como fallback, usa puntos característicos
    sobre una máscara verde de césped para no enganchar jugadores.
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

    def _get_green_mask(self, frame: np.ndarray) -> np.ndarray:
        """Crea una máscara que solo incluye los píxeles verdes (césped)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        return mask

    def compute_offset(self, frame: np.ndarray, custom_points: list = None) -> tuple[float, float]:
        """
        Calcula el desplazamiento (dx, dy) de la imagen actual respecto al frame anterior.
        Devuelve (dx, dy) = cuánto se ha movido el entorno. (Valores positivos/negativos).
        """
        # 1. Convertir a grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Inicialización
        if self.prev_gray is None:
            self.prev_gray = gray
            if custom_points and len(custom_points) > 0:
                self.prev_points = np.array(custom_points, dtype=np.float32).reshape(-1, 1, 2)
            else:
                mask = self._get_green_mask(frame)
                self.prev_points = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
            return (0.0, 0.0)

        # Si nos quedamos con muy pocos puntos (e.g. por paneo de la cámara)
        if self.prev_points is None or len(self.prev_points) < 5:
            if custom_points and len(custom_points) > 0:
                self.prev_points = np.array(custom_points, dtype=np.float32).reshape(-1, 1, 2)
            else:
                # Buscamos nuevos puntos de control pero SOLO en el césped (máscara verde)
                mask = self._get_green_mask(frame)
                self.prev_points = cv2.goodFeaturesToTrack(self.prev_gray, mask=mask, **self.feature_params)
            
            if self.prev_points is None:
                self.prev_gray = gray
                return (0.0, 0.0)

        # 3. Calcular Optical Flow usando pirámides de Lucas-Kanade
        next_points, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None, **self.lk_params
        )

        if next_points is None or status is None:
            self.prev_gray = gray
            self.prev_points = None
            return (0.0, 0.0)

        # 4. Filtrar solo los puntos fiables (trackeados con éxito)
        good_new = next_points[status == 1]
        good_old = self.prev_points[status == 1]

        # 5. Si hay muy pocos puntos buenos, abortar este frame (el siguiente re-inicializará)
        if len(good_new) < 5:
            self.prev_gray = gray
            self.prev_points = None
            return (0.0, 0.0)

        # 6. Calcular el desplazamiento promedio en X y en Y
        diff = good_new - good_old
        if len(diff) > 0:
            dx = np.median(diff[:, 0])
            dy = np.median(diff[:, 1])
        else:
            dx, dy = 0.0, 0.0

        # 7. Actualizar referencias para el siguiente ciclo
        self.prev_gray = gray.copy()
        
        # Guardamos los puntos trackeados como prev_points para continuar persiguiendo LOS MISMOS PUNTOS
        # (ya sean los custom_points inyectados, o los buenos filtrados sobre el césped)
        self.prev_points = good_new.reshape(-1, 1, 2)

        return (float(dx), float(dy))
