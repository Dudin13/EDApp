"""
tracker.py — Tracker avanzado basado en ByteTrack (via supervision).

FIXES APLICADOS:
  [CRÍTICO] frame_rate estaba hardcodeado a 2, independientemente del sample_rate
            real del análisis. Si el video se procesa a 0.5s de intervalo el
            tracker calculaba velocidades de Kalman incorrectas (factor 4x error).
            Ahora ProfessionalTracker acepta sample_rate en __init__ y calcula
            frame_rate = round(1.0 / sample_rate) automáticamente.
  [CRÍTICO] reset() recreaba ByteTrack sin los parámetros originales, perdiendo
            la configuración de lost_track_buffer etc. Ahora guarda los params.
  [MEJORA]  La compensación de cámara (optical flow offset) ahora también
            corrige lost_tracks además de tracked_tracks.
  [MEJORA]  initialize_with_seeds implementado correctamente — inyecta
            detecciones sintéticas en el primer update para pre-sembrar IDs.
"""

import logging
import numpy as np
import supervision as sv
from dataclasses import dataclass, field
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class Track:
    track_id: int
    clase: str
    equipo: int          # 0=local, 1=visitante, 2=árbitro (calculado por votación)
    last_box: tuple      # (cx, cy, w, h) último frame visto
    frames_seen: int = 1
    frames_lost: int = 0
    history_x: list = field(default_factory=list)
    history_y: list = field(default_factory=list)
    history_minute: list = field(default_factory=list)
    appearance_color: Optional[tuple] = None  # (r, g, b) medio del torso
    # Votación de equipo: buffer de las últimas N predicciones para evitar
    # que un frame malo fije el equipo incorrecto para toda la secuencia.
    _equipo_votes: list = field(default_factory=list)
    _VOTE_WINDOW: int = field(default=8, init=False, repr=False)

    def vote_equipo(self, equipo_pred: int) -> None:
        """Registra una nueva predicción de equipo y actualiza self.equipo por mayoría."""
        # Árbitros (equipo=2) son definitivos, no necesitan votación
        if self.clase == "referee":
            self.equipo = 2
            return
        self._equipo_votes.append(equipo_pred)
        if len(self._equipo_votes) > self._VOTE_WINDOW:
            self._equipo_votes.pop(0)
        # Mayoría simple sobre el buffer (ignora equipo=2 en jugadores de campo)
        votes = [v for v in self._equipo_votes if v != 2]
        if votes:
            self.equipo = max(set(votes), key=votes.count)


class ProfessionalTracker:
    """
    Tracker avanzado basado en ByteTrack (via supervision).
    Utiliza predicción de Kalman y lógica persistente para no perder IDs.
    """

    # Parámetros por defecto
    _DEFAULT_TRACK_ACTIVATION = 0.20
    _DEFAULT_LOST_BUFFER      = 30
    _DEFAULT_MATCH_THRESHOLD  = 0.8

    def __init__(self, sample_rate: float = 0.5):
        """
        Args:
            sample_rate: Intervalo en segundos entre frames analizados.
                         FIX: antes hardcodeado a frame_rate=2 (= 1/0.5s).
                         Ahora se calcula dinámicamente desde sample_rate real.
        """
        self.sample_rate = sample_rate
        # frame_rate para ByteTrack = cuántos "frames" por segundo procesamos
        # Si sample_rate=0.5s → procesamos ~2 frames/s
        # Si sample_rate=1.0s → procesamos ~1 frame/s
        self._frame_rate = max(1, round(1.0 / sample_rate))

        self._bt_params = dict(
            track_activation_threshold=self._DEFAULT_TRACK_ACTIVATION,
            lost_track_buffer=self._DEFAULT_LOST_BUFFER,
            minimum_matching_threshold=self._DEFAULT_MATCH_THRESHOLD,
            frame_rate=self._frame_rate,
        )
        self._tracker = sv.ByteTrack(**self._bt_params)
        self._tracks: Dict[int, Track] = {}
        self._history: Dict[int, Track] = {}

    def update(
        self,
        detecciones: list,
        equipo_map: list,
        minute: float = 0.0,
        camera_offset: tuple = (0.0, 0.0)
    ) -> Dict[int, Track]:
        """
        Actualiza el tracker con las detecciones del frame actual.

        Args:
            detecciones: Lista de dicts de detección (salida de detector.py).
            equipo_map:  Lista de ints (0/1/2) con el equipo de cada detección,
                         en el mismo orden que detecciones.
            minute:      Minuto de partido para el historial.
            camera_offset: (dx, dy) en píxeles de movimiento de cámara
                           (calculado externamente con optical flow).
        """
        if not detecciones:
            return self._tracks

        # ── Fix 6: Compensación de movimiento de cámara ───────────────────
        # Usamos getattr(..., None) para acceder a tracked_tracks / lost_tracks
        # de forma agnóstica a versiones de supervision (los atributos internos
        # de ByteTrack pueden cambiar entre releases sin warning).
        dx, dy = camera_offset
        if dx != 0.0 or dy != 0.0:
            try:
                for attr in ("tracked_tracks", "lost_tracks"):
                    track_list = getattr(self._tracker, attr, None)
                    if not track_list:
                        continue
                    for track in track_list:
                        mean = getattr(track, "mean", None)
                        if mean is None:
                            continue
                        # Proteger ante arrays de solo lectura o shapes inesperados
                        try:
                            if hasattr(mean, "__len__") and len(mean) >= 2:
                                track.mean[0] += dx
                                track.mean[1] += dy
                        except (TypeError, IndexError, ValueError):
                            pass  # mean no es mutable o tiene shape distinto
            except Exception:  # noqa: BLE001
                # Protección total ante cambios de versión de supervision
                logger.warning(
                    "[ProfessionalTracker] Compensación de cámara no aplicada "
                    "(API interna de ByteTrack puede haber cambiado de versión)"
                )

        # ── Convertir detecciones a formato supervision ───────────────────
        xyxy = []
        confidence = []
        class_id = []
        
        # Mapeo de nombres de clase a IDs para que ByteTrack no mezcle tipos
        # 0: player, 1: goalkeeper, 2: ball, 3: referee
        class_name_to_id = {"goalkeeper": 1, "player": 0, "referee": 3, "ball": 2}

        for d in detecciones:
            if "bbox" in d:
                xyxy.append(list(d["bbox"]))
            else:
                x1 = d["x"] - d["w"] / 2
                y1 = d["y"] - d["h"] / 2
                x2 = d["x"] + d["w"] / 2
                y2 = d["y"] + d["h"] / 2
                xyxy.append([x1, y1, x2, y2])
            
            confidence.append(d.get("conf", d.get("confianza", 0.5)))
            class_id.append(class_name_to_id.get(d.get("clase"), 0))

        sv_detections = sv.Detections(
            xyxy=np.array(xyxy, dtype=np.float32),
            confidence=np.array(confidence, dtype=np.float32),
            class_id=np.array(class_id, dtype=int),
        )

        # ── ByteTrack update ──────────────────────────────────────────────
        sv_detections = self._tracker.update_with_detections(sv_detections)

        # ── Mapear IDs ByteTrack → estructura Track ───────────────────────
        current_active = {}

        for i in range(len(sv_detections)):
            tid = int(sv_detections.tracker_id[i])
            x1, y1, x2, y2 = sv_detections.xyxy[i]
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2

            if tid not in self._history:
                # Track nuevo: buscar equipo y clase en detecciones originales por proximidad
                equipo, clase = self._match_equipo_clase(cx, cy, detecciones, equipo_map)
                new_track = Track(
                    track_id=tid,
                    clase=clase,
                    equipo=equipo,
                    last_box=(cx, cy, w, h),
                    history_x=[cx],
                    history_y=[cy],
                    history_minute=[minute],
                )
                new_track.vote_equipo(equipo)  # primer voto
                self._history[tid] = new_track
            else:
                track = self._history[tid]
                track.last_box = (cx, cy, w, h)
                track.history_x.append(cx)
                track.history_y.append(cy)
                track.history_minute.append(minute)
                track.frames_seen += 1
                # Actualizar equipo con la predicción del frame actual
                new_equipo, _ = self._match_equipo_clase(cx, cy, detecciones, equipo_map)
                track.vote_equipo(new_equipo)

            current_active[tid] = self._history[tid]

        self._tracks = current_active
        return self._tracks

    def _match_equipo_clase(self, cx: float, cy: float,
                            detecciones: list, equipo_map: list) -> tuple[int, str]:
        """
        Encuentra el equipo y la clase de la detección original más cercana al track.
        Returns: (equipo: int, clase: str)
        """
        if not detecciones or not equipo_map:
            return 0, "player"
        dist_min = float("inf")
        equipo = 0
        clase = "player"
        for d_idx, d in enumerate(detecciones):
            dist = np.hypot(cx - d["x"], cy - d["y"])
            if dist < dist_min:
                dist_min = dist
                equipo = equipo_map[d_idx] if d_idx < len(equipo_map) else 0
                clase = d.get("clase", "player")
        return equipo, clase

    def get_all_tracks(self) -> List[Track]:
        """Retorna todos los tracks registrados en la sesión completa."""
        return list(self._history.values())

    def reset(self):
        """
        FIX: Antes recreaba ByteTrack con parámetros por defecto, perdiendo
        la configuración original (lost_track_buffer, frame_rate, etc.).
        Ahora reutiliza self._bt_params guardados en __init__.
        """
        self._tracker = sv.ByteTrack(**self._bt_params)
        self._tracks = {}
        self._history = {}

    def initialize_with_seeds(self, seeds: List[Dict]):
        """
        FIX: Implementación real de la siembra manual de jugadores.
        Crea una detección sintética para cada seed y la inyecta en ByteTrack
        con confianza alta para forzar la creación del track desde el primer frame.

        Args:
            seeds: Lista de dicts con {"x": int, "y": int} en coordenadas de frame.
        """
        if not seeds:
            return

        # Crear detecciones sintéticas con tamaño estimado de jugador
        PLAYER_W, PLAYER_H = 40, 80  # píxeles estimados jugador en plano general

        xyxy = []
        confidence = []
        for s in seeds:
            x, y = s["x"], s["y"]
            xyxy.append([x - PLAYER_W/2, y - PLAYER_H/2,
                         x + PLAYER_W/2, y + PLAYER_H/2])
            confidence.append(0.95)  # Alta confianza para activar inmediatamente

        sv_seeds = sv.Detections(
            xyxy=np.array(xyxy, dtype=np.float32),
            confidence=np.array(confidence, dtype=np.float32),
            class_id=np.zeros(len(seeds), dtype=int),
        )

        # Primer update con las semillas — ByteTrack asignará track_id desde 1
        self._tracker.update_with_detections(sv_seeds)


# Alias de compatibilidad
SimpleTracker = ProfessionalTracker
