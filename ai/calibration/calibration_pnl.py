from core.config.settings import settings
from core.logger import logger

class PnLCalibrator:
    """
    Clase para manejar la calibración y transformación de coordenadas (Homografía).
    """
    def __init__(self):
        self.H = None # Homography matrix
        self.pitch_width = 105.0
        self.pitch_height = 68.0

    def set_homography(self, H):
        self.H = H
        logger.debug("Nueva matriz de homografía establecida")

    def transform_point(self, x, y):
        """
        Transforma un punto de píxeles a metros en el campo.
        """
        if self.H is None:
            return None
        
        # Implementación de cv2.perspectiveTransform para un punto
        import numpy as np
        pts = np.array([[[x, y]]], dtype=np.float32)
        # Nota: Normalmente H se calcula de campo a imagen, 
        # para ir de imagen a campo necesitamos la inversa o H_inv
        try:
            # Asumimos que H ya es la matriz que mapea imagen -> campo
            # Si H es campo -> imagen, necesitamos invertira.
            # En detector.py findHomography(pitch_pts, frame_pts) devuelve campo -> imagen.
            H_inv = getattr(self, "H_inv", None)
            if H_inv is None:
                import cv2
                _, self.H_inv = cv2.invert(self.H)
                H_inv = self.H_inv
            
            transformed = cv2.perspectiveTransform(pts, H_inv)
            return transformed[0][0].tolist()
        except:
            return None
