import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Generator, Tuple

from core.config.settings import settings
from core.logger import logger
from ai.detector.detector import DetectorEngine
from ai.tracker.tracker import PlayerTracker
from ai.tracker.camera_motion import CameraMotionEstimator

class VideoAnalysisPipeline:
    """
    Orquestador del análisis de video.
    Encapsula el flujo: Ingesta -> Movimiento -> Detección -> Tracking -> Analítica.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sample_rate = config.get("sample_rate", 2.0)
        self.confidence = config.get("confidence", 0.35)
        
        # Core Components
        self.detector = DetectorEngine()
        self.tracker = PlayerTracker(sample_rate=self.sample_rate)
        self.camera_estimator = CameraMotionEstimator()
        
        # State
        self.calibration_results = {}
        self.last_ball_pos = None
        self.ball_history = []

    def process(self, video_path: str) -> Generator[Tuple[int, str, Dict], None, None]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Captura de video fallida: {video_path}")
            yield 0, "No se pudo abrir el video", {}
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(fps * self.sample_rate)
        
        # --- Fase 1: Calibración ---
        yield 5, "Calibrando sistema...", {}
        # Aquí iría lógica de auto-homografía y colores si fuesen necesarios
        # Por ahora simplificamos para el MVP
        
        # --- Fase 2: Procesamiento principal ---
        frame_idx = 0
        analyzed_count = 0
        total_to_analyze = total_frames // frame_interval
        
        while cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: break
            
            minute = (frame_idx / fps) / 60.0
            progress = 10 + int(80 * (analyzed_count / max(1, total_to_analyze)))
            
            # 1. Movimiento de cámara
            dx, dy = self.camera_estimator.compute_offset(frame)
            
            # 2. Detección
            detections = self.detector.detect_frame(frame, confidence=self.confidence)
            
            # 3. Separación y Tracking
            player_dets = [d for d in detections if d["clase"] != "ball"]
            ball_dets = [d for d in detections if d["clase"] == "ball"]
            
            # Clasificación de equipos (simplificada para migración)
            # En producción esto usaría el TeamClassifierSigLIP
            equipo_map = [0] * len(player_dets) 
            
            tracks = self.tracker.update(
                player_dets, 
                equipo_map, 
                minute=minute, 
                camera_offset=(dx, dy)
            )
            
            # 4. Lógica de balón (simplificada)
            if ball_dets:
                self.last_ball_pos = (ball_dets[0]["x"], ball_dets[0]["y"])
                self.ball_history.append((minute, *self.last_ball_pos))

            analyzed_count += 1
            frame_idx += frame_interval
            
            yield progress, f"Procesando minuto {minute:.1f}...", {}

        cap.release()
        yield 100, "Procesamiento completado", self._compile_results()

    def _compile_results(self) -> Dict:
        """Compila los tracks y eventos en un formato para la UI."""
        # Esta lógica se expandirá para coincidir con lo que Streamlit espera
        return {
            "tracks": self.tracker.history,
            "ball_history": self.ball_history
        }
