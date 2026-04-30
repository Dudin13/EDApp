"""
video_pipeline.py — Orquestador del análisis de video (v2).
Flujo: Ingesta -> Movimiento -> Detección -> Tracking -> Calibración -> Post-Procesado (Eventos/Físico).
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Generator, Tuple
from collections import defaultdict

from core.config.settings import settings
from core.logger import logger

# Módulos de análisis
try:
    from modules.detector import detect_frame, detect_pitch_homography
    from modules.tracker import ProfessionalTracker
    from modules.camera_motion import CameraMotionEstimator
    from modules.coordinates import FieldTransformer
    from modules.performance_engine import PerformanceEngine
    from modules.event_engine import EventEngine
    from modules.team_classifier import TeamClassifier, Team
    from modules.ball_interpolation import BallInterpolator
    _MODULES_OK = True
except ImportError as e:
    logger.error(f"[VideoAnalysisPipeline] Error importando módulos: {e}")
    _MODULES_OK = False


class VideoAnalysisPipeline:
    def __init__(self, config: Dict[str, Any]):
        print("DEBUG: Pipeline init started")
        self.config = config
        self.sample_rate = config.get("sample_rate", 0.5)
        self.confidence = config.get("confidence", 0.25)

        if not _MODULES_OK:
            raise RuntimeError("Módulos de análisis no disponibles.")

        # Motores Core
        print("DEBUG: Initializing ProfessionalTracker")
        self.tracker = ProfessionalTracker(sample_rate=self.sample_rate)
        print("DEBUG: Initializing CameraMotionEstimator")
        self.camera_estimator = CameraMotionEstimator()
        print("DEBUG: Initializing FieldTransformer")
        self.transformer = FieldTransformer()
        print("DEBUG: Initializing PerformanceEngine")
        self.performance_engine = PerformanceEngine()
        print("DEBUG: Initializing EventEngine")
        self.event_engine = EventEngine()
        print("DEBUG: Initializing BallInterpolator")
        self.ball_interpolator = BallInterpolator()
        print("DEBUG: Initializing TeamClassifier")
        self.team_classifier = config.get("team_classifier") or TeamClassifier()

        # Estado
        self.team_colors = {}
        self.raw_ball_history = []
        self.all_minutes = []
        self.total_detecciones = 0
        self.frames_analizados = 0
        self._clf_fitted = self.team_classifier.colors.fitted
        print("DEBUG: Pipeline init finished")

        # Inyectar semillas manuales
        manual_seeds = config.get("manual_seeds", [])
        if manual_seeds:
            self.tracker.initialize_with_seeds(manual_seeds)

    def process(self, video_path: str) -> Generator[Tuple[int, str, Dict], None, None]:
        if not video_path or not Path(video_path).exists():
            yield 0, "Error: Archivo de vídeo no encontrado.", {}
            return

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(round(fps * self.sample_rate)))
        total_to_analyze = max(1, total_frames // frame_interval)

        yield 5, "Sincronizando modelos...", {}

        # ── Fase 0: Calibración de Colores (si es necesario) ───────────────
        if not self._clf_fitted:
            yield 7, "Calibrando colores de equipos...", {}
            # Tomar frame en el segundo 10 para calibración rápida
            cap.set(cv2.CAP_PROP_POS_MSEC, 10000)
            ret, cframe = cap.read()
            if ret:
                dets = detect_frame(cframe, mode="auto", confidence=self.confidence)
                player_bboxes = [d["bbox"] for d in dets if d.get("clase") == "player"]
                if self.team_classifier.fit(cframe, player_bboxes):
                    self._clf_fitted = True
                    logger.info("TeamClassifier calibrado automáticamente.")
            # Resetear puntero para empezar Fase 1 desde el inicio
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # ── Fase 1: Análisis Frame a Frame ──────────────────────────────────
        frame_idx = 0
        analyzed_count = 0
        last_adapt_minute = 0.0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            if frame_idx % frame_interval == 0:
                minute = (frame_idx / fps) / 60.0
                self.all_minutes.append(minute)
                progress = 10 + int(70 * (analyzed_count / total_to_analyze))

                # 1. Movimiento y Calibración
                dx, dy = self.camera_estimator.compute_offset(frame)
                
                if analyzed_count % 10 == 0:
                    H = detect_pitch_homography(frame)
                    if H is not None:
                        self.transformer.set_homography(H)

                # 2. Detección SOTA (Players + Ball)
                detections = detect_frame(frame, mode="auto", confidence=self.confidence)
                player_dets = [d for d in detections if d.get("clase") != "ball"]
                ball_dets   = [d for d in detections if d.get("clase") == "ball"]

                # 3. Clasificación de Equipo
                equipo_map = []
                for d in player_dets:
                    team = self.team_classifier.predict(frame, d["bbox"], is_referee=(d["clase"]=="referee"))
                    if team == Team.A: equipo_map.append(0)
                    elif team == Team.B: equipo_map.append(1)
                    elif team == Team.REFEREE: equipo_map.append(2)
                    else: equipo_map.append(-1)

                # 4. Tracking con compensación de cámara
                active_tracks = self.tracker.update(
                    player_dets, equipo_map, minute=minute, camera_offset=(dx, dy)
                )

                # Adaptación incremental de colores cada 2 minutos
                if self._clf_fitted and (minute - last_adapt_minute) > 2.0:
                    adapt_data = [{"bbox": t.last_box, "equipo": t.equipo} for t in active_tracks.values() if t.equipo in (0,1)]
                    if self.team_classifier.adapt(frame, adapt_data):
                        last_adapt_minute = minute

                # 5. Enriquecer tracks con coordenadas de campo
                for tid, track in active_tracks.items():
                    mx, my = self.transformer.pixel_to_pitch(track.last_box[0], track.last_box[1])
                    track.history_pitch_x.append(mx)
                    track.history_pitch_y.append(my)

                # 6. Registro de balón
                if ball_dets:
                    bx, by = ball_dets[0]["x"], ball_dets[0]["y"]
                    bmx, bmy = self.transformer.pixel_to_pitch(bx, by)
                    self.raw_ball_history.append({
                        "minute": minute, "pitch_x": bmx, "pitch_y": bmy
                    })

                self.total_detecciones += len(detections)
                self.frames_analizados += 1
                analyzed_count += 1
                
                yield progress, f"Analizando minuto {minute:.1f}...", {}

            frame_idx += 1

        cap.release()

        # ── Fase 2: Post-Procesado de Inteligencia ──────────────────────────
        yield 85, "Refinando trayectoria del balón...", {}
        interpolated_ball = self.ball_interpolator.process_with_timestamps(
            self.raw_ball_history, self.all_minutes
        )

        yield 90, "Detectando eventos tácticos (pases/robos)...", {}
        # Convertir historial de tracks a formato plano para los motores
        tracks_data = {}
        for tid, t in self.tracker._history.items():
            tracks_data[tid] = {
                "history_minute": t.history_minute,
                "pitch_x": t.history_pitch_x,
                "pitch_y": t.history_pitch_y,
                "equipo": t.equipo
            }

        events = self.event_engine.detect_events(tracks_data, interpolated_ball)

        yield 95, "Calculando métricas físicas reales...", {}
        performance_stats = self.performance_engine.process_all_tracks(tracks_data)

        # ── Fase 3: Compilación Final ───────────────────────────────────────
        yield 100, "Análisis completado ✓", self._compile_v2_results(
            tracks_data, interpolated_ball, events, performance_stats
        )

    def _compile_v2_results(self, tracks_data, ball_history, events, stats) -> Dict:
        """Formatea los resultados para la UI de Streamlit."""
        tracks_serializable = {}
        for tid, t in self.tracker._history.items():
            p_stats = stats.get(tid, {"distance_km": 0, "top_speed_kmh": 0, "sprint_count": 0})
            tracks_serializable[str(tid)] = {
                "track_id": tid,
                "equipo": t.equipo,
                "clase": t.clase,
                "history_x": t.history_x,
                "history_y": t.history_y,
                "history_minute": t.history_minute,
                "distance_km": p_stats["distance_km"],
                "top_speed": p_stats["top_speed_kmh"],
                "sprints": p_stats["sprint_count"],
                "total_actions": len([e for e in events if e["from_tid"] == tid or e.get("to_tid") == tid])
            }

        return {
            "tracks": tracks_serializable,
            "ball_events": events,
            "ball_history": ball_history,
            "frames_analizados": self.frames_analizados,
            "total_detecciones": self.total_detecciones,
            "heatmap_x": [x for t in self.tracker._history.values() for x in t.history_x],
            "heatmap_y": [y for t in self.tracker._history.values() for y in t.history_y],
        }
