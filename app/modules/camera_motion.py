import cv2
import numpy as np

class CameraMotionEstimator:
    """
    Estima el movimiento global de la cámara usando Optical Flow (Lucas-Kanade).
    Rastrea puntos característicos del fondo (como las líneas del campo o publicidad)
    para deducir cómo se ha movido el plano entre dos frames consecutivos.
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

    def compute_offset(self, frame: np.ndarray) -> tuple[float, float]:
        """
        Calcula el desplazamiento (dx, dy) de la imagen actual respecto al frame anterior.
        Devuelve (dx, dy) = cuánto se ha movido el entorno. (Valores positivos/negativos).
        """
        # 1. Convertir a grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Inicialización en el primer frame
        if self.prev_gray is None:
            self.prev_gray = gray
            # Encontrar puntos interesantes para trackear (idealmente fondo estático)
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            return (0.0, 0.0)

        if self.prev_points is None or len(self.prev_points) < 10:
            # Si nos quedamos sin puntos, buscar nuevos
            self.prev_points = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
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

        # 5. Si hay muy pocos puntos buenos o falló demasiado, reiniciar
        if len(good_new) < 5:
            self.prev_gray = gray
            self.prev_points = None
            return (0.0, 0.0)

        # 6. Calcular el desplazamiento promedio en X y en Y
        # d_point = P_nuevo - P_viejo (cuánto se movió el píxel)
        # Una panorámica a la derecha hace que los objetos se muevan a la izquierda (dx < 0)
        # Queremos compensar el tracker, así que pasaremos cuánto se movió el "fondo".
        diff = good_new - good_old
        dx = np.median(diff[:, 0])
        dy = np.median(diff[:, 1])

        # 7. Actualizar referencias para el siguiente ciclo
        self.prev_gray = gray.copy()
        # Refrescar los puntos de control en cada frame para no perderlos si salen del encuadre
        # (Mejora: se podría hacer solo si le quedan pocos puntos a trackear, por rendimiento)
        self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)

        return (float(dx), float(dy))

