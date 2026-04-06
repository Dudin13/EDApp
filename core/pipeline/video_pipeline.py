"""
video_pipeline.py — Orquestador del análisis de video.
Encapsula el flujo: Ingesta -> Movimiento -> Detección -> Tracking -> Analítica.

FIXES APLICADOS:
  [CRÍTICO] Fix 1: Eliminados imports de 'ai.detector.detector' y 'ai.tracker.*'
            (paquete ai/ no existe). Se usan los módulos reales en app/modules/.
  [CRÍTICO] Fix 3: equipo_map ya no es [0]*len (todos en equipo 0). Se usa
            auto_detect_team_colors + classify_team del detector real.
  [MEDIO]   Fix 4: frame_interval usa max(1, round(...)) y valida fps > 0.
  [BAJO]    Fix 8: Validación de video_path antes de abrir el cap.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Generator, Tuple

from core.config.settings import settings
from core.logger import logger

# ── Fix 1: Imports de módulos reales ────────────────────────────────────────
# El paquete 'ai/' no existe. Los módulos están en app/modules/ y son accesibles
# porque app.py añade el directorio app/ a sys.path antes de ejecutar cualquier página.
try:
    from modules.detector import (
        detect_frame,
        auto_detect_team_colors,
        classify_team,
    )
    from modules.tracker import ProfessionalTracker
    from modules.camera_motion import CameraMotionEstimator
    _MODULES_OK = True
except ImportError:
    # Fallback: intento con ruta absoluta de paquete (útil en tests / ejecución directa)
    try:
        import sys
        _app_dir = str(Path(__file__).resolve().parent.parent.parent / "app")
        if _app_dir not in sys.path:
            sys.path.insert(0, _app_dir)
        from modules.detector import detect_frame, auto_detect_team_colors, classify_team
        from modules.tracker import ProfessionalTracker
        from modules.camera_motion import CameraMotionEstimator
        _MODULES_OK = True
    except ImportError as _e:
        logger.error(f"[VideoAnalysisPipeline] No se pudieron importar módulos de análisis: {_e}")
        detect_frame = None
        auto_detect_team_colors = None
        classify_team = None
        ProfessionalTracker = None
        CameraMotionEstimator = None
        _MODULES_OK = False


class VideoAnalysisPipeline:
    """
    Orquestador del análisis de video.
    Encapsula el flujo: Ingesta -> Movimiento -> Detección -> Tracking -> Analítica.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sample_rate = config.get("sample_rate", 2.0)
        self.confidence = config.get("confidence", 0.35)

        if not _MODULES_OK:
            raise RuntimeError(
                "Los módulos de análisis (detector, tracker, camera_motion) "
                "no están disponibles. Revisa el sys.path y los imports."
            )

        # Core Components
        self.tracker = ProfessionalTracker(sample_rate=self.sample_rate)
        self.camera_estimator = CameraMotionEstimator()

        # State
        self.team_colors: Dict = {}
        self.last_ball_pos = None
        self.ball_history: List = []
        self.total_detecciones: int = 0
        self.frames_analizados: int = 0

        # Fix: Inyectar manual_seeds en el tracker ANTES de procesar el video.
        # Esto crea tracks pre-sembrados con confianza alta para que ByteTrack los
        # reconozca desde el primer frame real y mantenga los IDs asignados manualmente.
        manual_seeds = config.get("manual_seeds", [])
        if manual_seeds:
            logger.info(f"Inyectando {len(manual_seeds)} seeds manuales en el tracker")
            self.tracker.initialize_with_seeds(manual_seeds)

    def process(self, video_path: str) -> Generator[Tuple[int, str, Dict], None, None]:
        # ── Fix 8: Validar ruta antes de abrir ──────────────────────────────
        if not video_path:
            logger.error("video_path está vacío")
            yield 0, "Error: No se ha especificado ningún vídeo.", {}
            return
        if not Path(video_path).exists():
            logger.error(f"El archivo de vídeo no existe: {video_path}")
            yield 0, f"Error: El archivo '{Path(video_path).name}' no existe.", {}
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Captura de video fallida: {video_path}")
            yield 0, "No se pudo abrir el vídeo (formato no soportado).", {}
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # ── Fix 4: Validar fps antes de usar ────────────────────────────────
        if fps <= 0:
            logger.warning(f"FPS inválido ({fps}), usando 25.0 por defecto")
            fps = 25.0

        # max(1, ...) evita frame_interval=0 cuando sample_rate es muy bajo
        frame_interval = max(1, int(round(fps * self.sample_rate)))
        total_to_analyze = max(1, total_frames // frame_interval)

        logger.info(
            f"Iniciando análisis | fps={fps:.1f} | frames={total_frames} "
            f"| intervalo={frame_interval} | ~{total_to_analyze} frames a analizar"
        )

        # ── Fase 1: Preparación ──────────────────────────────────────────────
        yield 5, "Calibrando sistema...", {}

        # ── Fase 2: Procesamiento principal ─────────────────────────────────
        frame_idx = 0
        analyzed_count = 0
        team_colors_initialized = False

        # Normalizar confianza a [0, 1]
        conf_val = (self.confidence / 100.0
                    if isinstance(self.confidence, (int, float)) and self.confidence > 1
                    else float(self.confidence))

        while cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            minute = (frame_idx / fps) / 60.0
            progress = 10 + int(80 * (analyzed_count / total_to_analyze))

            # 1. Movimiento de cámara
            dx, dy = self.camera_estimator.compute_offset(frame)

            # 2. Detección
            try:
                detections = detect_frame(frame, mode="auto", confidence=conf_val)
            except Exception as e:
                logger.error(f"Error en detección frame {frame_idx}: {e}")
                detections = []

            # 3. Separación jugadores / balón
            player_dets = [d for d in detections if d.get("clase") != "ball"]
            ball_dets   = [d for d in detections if d.get("clase") == "ball"]

            # ── Fix 3: Clasificación de equipos real ─────────────────────────
            # Primer frame con suficientes jugadores: inicializar colores de equipo.
            # Frames posteriores: clasificar cada jugador con el mapa aprendido.
            if not team_colors_initialized and len(player_dets) >= 4:
                try:
                    self.team_colors = auto_detect_team_colors(
                        frame, player_dets, mode="kmeans"
                    )
                    team_colors_initialized = True
                    logger.info(f"Colores de equipo detectados: {self.team_colors}")
                except Exception as e:
                    logger.warning(f"No se pudo detectar colores de equipo: {e}")

            try:
                equipo_map = [
                    classify_team(frame, d, self.team_colors)
                    for d in player_dets
                ]
            except Exception as e:
                logger.warning(f"Error en classify_team: {e}")
                equipo_map = [0] * len(player_dets)

            # 4. Tracking
            try:
                self.tracker.update(
                    player_dets,
                    equipo_map,
                    minute=minute,
                    camera_offset=(dx, dy),
                )
            except Exception as e:
                logger.error(f"Error en tracker.update frame {frame_idx}: {e}")

            # 5. Lógica de balón
            if ball_dets:
                self.last_ball_pos = (ball_dets[0]["x"], ball_dets[0]["y"])
                self.ball_history.append((minute, *self.last_ball_pos))

            # Acumular métricas globales
            self.total_detecciones += len(detections)
            self.frames_analizados += 1
            analyzed_count += 1
            frame_idx += frame_interval

            yield progress, f"Procesando minuto {minute:.1f}...", {}

        cap.release()
        logger.info(f"Análisis completado: {analyzed_count} frames procesados")
        yield 100, "Procesamiento completado ✓", self._compile_results()

    def _compile_results(self) -> Dict:
        """Compila los tracks y eventos en un formato serializable para la UI."""
        tracks_serializable = {}
        for tid, track in self.tracker._history.items():
            tracks_serializable[str(tid)] = {
                "track_id":       track.track_id,
                "equipo":         track.equipo,
                "clase":          track.clase,
                "frames_seen":    track.frames_seen,
                "history_x":      track.history_x,
                "history_y":      track.history_y,
                "history_minute": track.history_minute,
            }

        # Serializar colores (tuplas -> listas para JSON)
        team_colors_serial = {
            k: list(v) if isinstance(v, tuple) else v
            for k, v in self.team_colors.items()
            if k != "_siglip_fitted"
        }

        # Fix: ball_events es el nombre que espera la UI (Step 5, collective_dashboard, etc.)
        # ball_history contiene tuplas (minute, x, y) → las convertimos a dicts legibles.
        ball_events_serializable = [
            {"minute": round(float(e[0]), 2), "x": int(e[1]), "y": int(e[2])}
            for e in self.ball_history
            if len(e) == 3
        ]

        # Heatmap coords para collective_dashboard
        heatmap_x = [t.history_x for t in self.tracker._history.values()]
        heatmap_y = [t.history_y for t in self.tracker._history.values()]
        heatmap_x_flat = [x for sublist in heatmap_x for x in sublist]
        heatmap_y_flat = [y for sublist in heatmap_y for y in sublist]

        return {
            "tracks":              tracks_serializable,
            "ball_events":         ball_events_serializable,   # ← nombre correcto para UI
            "ball_history":        self.ball_history,           # compatibilidad
            "team_colors":         team_colors_serial,
            "frames_analizados":   self.frames_analizados,
            "total_detecciones":   self.total_detecciones,
            "heatmap_x":           heatmap_x_flat,
            "heatmap_y":           heatmap_y_flat,
        }
