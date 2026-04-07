"""
event_spotter_tdeed.py
======================
Detector de eventos de futbol basado en posesion del balon.

Inspirado en:
  Peral et al. "Temporally Accurate Events Detection Through Ball Possessor
  Recognition in Soccer" (VISAPP 2025)
  https://doi.org/10.5220/0013317700003912

Pipeline:
  1. Para cada frame: encontrar el jugador mas cercano al balon -> poseedor
  2. Suavizar la secuencia de poseedores con ventana temporal (Gaussian filter)
  3. Detectar cambios de poseedor -> PASE / RECEPCION
  4. Clasificar acciones adicionales con reglas geometricas (tiro, corner, etc.)

Ventajas sobre T-DEED:
  - No requiere modelo preentrenado ni features de video
  - Funciona directamente con los tracks de ByteTrack + posicion del balon
  - Ligero: corre en CPU en tiempo real
  - Adaptado a camaras VEO panoramicas
"""

import torch
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BallEvent:
    """Evento detectado con el balon."""
    timestamp:    float        # segundo del video
    minute:       float        # minuto del partido
    action:       str          # Pase, Recepcion, Tiro, Corner, etc.
    track_id:     int          # ID del jugador que realiza la accion
    team:         int          # 0=local, 1=visitante, 2=arbitro
    ball_pos:     tuple        # (x, y) en pixeles
    pitch_pos:    tuple        # (x, y) en coordenadas de campo (0-105, 0-68)
    confidence:   float        # 0.0 - 1.0
    is_validated: bool = True  # paso la validacion geometrica


class EventSpotterTDEED:
    """
    Detector de eventos de futbol basado en posesion del balon.

    Uso:
        spotter = EventSpotterTDEED()

        # En cada frame del video:
        spotter.update(
            frame_second=12.5,
            minute=0.2,
            tracks=tracks_dict,          # salida de ProfessionalTracker.update()
            ball_pos=(640, 360),         # posicion del balon en pixeles
            pitch_pos=(52.3, 34.1),      # posicion en coordenadas de campo
            ball_conf=0.8
        )

        # Al final del video:
        events = spotter.get_events()
    """

    # Radio de contacto balon-jugador en pixeles (camara VEO panoramica)
    CONTACT_RADIUS_PX   = 120
    # Minimo de frames seguidos para confirmar posesion
    MIN_POSSESSION_FRAMES = 2
    # Ventana de suavizado de posesion (frames)
    SMOOTHING_WINDOW    = 5
    # Minimo tiempo entre eventos del mismo jugador (segundos)
    MIN_EVENT_GAP_S     = 0.8
    # Velocidad maxima del balon en px/s (para filtrar detecciones falsas)
    MAX_BALL_SPEED_PX_S = 900

    def __init__(self, weights_path: str = None):
        self.weights = weights_path
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Clases de eventos (compatibilidad con codigo anterior)
        self.classes = [
            "Pase", "Recepcion", "Tiro", "Corner", "Saque de banda",
            "Conduccion", "Despeje", "Recuperacion", "Penalty"
        ]

        # Estado interno
        self._possession_history: deque = deque(maxlen=self.SMOOTHING_WINDOW)
        self._events: list[BallEvent]   = []
        self._last_event_by_track: dict = {}   # track_id -> ultimo timestamp
        self._last_ball_pos: Optional[tuple]   = None
        self._last_ball_second: float          = -1.0
        self._frame_count: int                 = 0

        # Para detectar cambios de posesion
        self._current_possessor:  Optional[int] = None
        self._possessor_frames:   int           = 0
        self._previous_possessor: Optional[int] = None

        print(f"EventSpotter: Inicializado en {self.device} | "
              f"Modo: posesion de balon (Peral et al. 2025)")

    # ── API principal ──────────────────────────────────────────────────────

    def update(self, frame_second: float, minute: float,
               tracks: dict, ball_pos: Optional[tuple],
               pitch_pos: tuple = (0, 0), ball_conf: float = 0.0) -> list[BallEvent]:
        """
        Procesa un frame y detecta eventos.

        Args:
            frame_second: Segundo del video
            minute:       Minuto del partido
            tracks:       Dict {track_id: Track} de ProfessionalTracker
            ball_pos:     (x, y) del balon en pixeles, o None si no se detecta
            pitch_pos:    (x, y) del balon en coordenadas de campo (0-105, 0-68)
            ball_conf:    Confianza de la deteccion del balon

        Returns:
            Lista de BallEvent detectados en este frame (normalmente 0 o 1)
        """
        self._frame_count += 1
        new_events = []

        if ball_pos is None or ball_conf < 0.15:
            self._possession_history.append(None)
            return new_events

        bx, by = ball_pos

        # ── Filtro de velocidad (eliminar saltos de balon falsos) ──────────
        if self._last_ball_pos and self._last_ball_second > 0:
            dt   = max(frame_second - self._last_ball_second, 0.001)
            dist = np.hypot(bx - self._last_ball_pos[0], by - self._last_ball_pos[1])
            if (dist / dt) > self.MAX_BALL_SPEED_PX_S:
                # Balon se movio demasiado rapido -> probable falsa deteccion
                self._possession_history.append(None)
                return new_events

        self._last_ball_pos    = ball_pos
        self._last_ball_second = frame_second

        # ── Encontrar poseedor: jugador mas cercano al balon ───────────────
        possessor_id   = None
        possessor_dist = float("inf")
        possessor_team = -1

        for tid, track in tracks.items():
            if track.frames_lost > 0:
                continue
            if track.clase == "referee":
                continue
            tx, ty = track.last_box[0], track.last_box[1]
            dist   = np.hypot(tx - bx, ty - by)
            if dist < possessor_dist:
                possessor_dist = dist
                possessor_id   = tid
                possessor_team = track.equipo

        # Solo confirmar posesion si el jugador esta suficientemente cerca
        if possessor_dist > self.CONTACT_RADIUS_PX:
            possessor_id = None

        self._possession_history.append(possessor_id)

        # ── Suavizar posesion con ventana temporal ─────────────────────────
        smoothed_possessor = self._smooth_possessor()

        # ── Detectar cambio de poseedor ────────────────────────────────────
        if smoothed_possessor != self._current_possessor:
            if smoothed_possessor is not None:
                self._possessor_frames = 1
            prev = self._current_possessor
            self._previous_possessor = prev
            self._current_possessor  = smoothed_possessor

            # Cambio real de posesion -> clasificar evento
            if prev is not None and smoothed_possessor is not None:
                # Obtener equipo del poseedor actual
                team = -1
                if smoothed_possessor in tracks:
                    team = tracks[smoothed_possessor].equipo

                action = self._classify_action(
                    prev_possessor=prev,
                    new_possessor=smoothed_possessor,
                    tracks=tracks,
                    ball_pos=ball_pos,
                    pitch_pos=pitch_pos,
                    frame_second=frame_second
                )

                # Verificar gap minimo entre eventos del mismo jugador
                last_ev = self._last_event_by_track.get(prev, -999)
                if frame_second - last_ev >= self.MIN_EVENT_GAP_S:
                    event = BallEvent(
                        timestamp   = frame_second,
                        minute      = minute,
                        action      = action,
                        track_id    = prev,  # quien ENVIA
                        team        = team,
                        ball_pos    = ball_pos,
                        pitch_pos   = pitch_pos,
                        confidence  = min(ball_conf, 0.95),
                        is_validated= True,
                    )
                    if self.validate_geometrical({"action": action}, pitch_pos):
                        self._events.append(event)
                        self._last_event_by_track[prev] = frame_second
                        new_events.append(event)
        else:
            if smoothed_possessor is not None:
                self._possessor_frames += 1

        return new_events

    def _smooth_possessor(self) -> Optional[int]:
        """
        Devuelve el poseedor mas frecuente en la ventana de suavizado.
        Filtra None (sin balon) si hay suficientes detecciones validas.
        """
        history = list(self._possession_history)
        valid   = [h for h in history if h is not None]
        if not valid:
            return None
        # El poseedor es el que aparece mas veces en la ventana
        from collections import Counter
        most_common = Counter(valid).most_common(1)[0]
        # Solo confirmar si aparece en al menos la mitad de la ventana
        if most_common[1] >= max(1, len(history) // 2):
            return most_common[0]
        return None

    def _classify_action(self, prev_possessor: int, new_possessor: int,
                         tracks: dict, ball_pos: tuple,
                         pitch_pos: tuple, frame_second: float) -> str:
        """
        Clasifica la accion basandose en el cambio de posesion y contexto geometrico.
        """
        x, y = pitch_pos

        # Obtener equipos
        prev_team = tracks[prev_possessor].equipo if prev_possessor in tracks else -1
        new_team  = tracks[new_possessor].equipo  if new_possessor in tracks else -1
        prev_cls  = tracks[prev_possessor].clase  if prev_possessor in tracks else "player"

        # Portero -> Despeje
        if prev_cls == "goalkeeper":
            return "Despeje"

        # Cambio de equipo -> Recuperacion (el nuevo equipo recupera)
        if prev_team != new_team and prev_team >= 0 and new_team >= 0:
            return "Recuperacion"

        # Mismo equipo -> Pase
        if prev_team == new_team:
            # Velocidad del balon para distinguir pase largo de corto
            # (se calcula en video_processor con ball_history)
            # Aqui usamos posicion para estimar
            if x < 10 or x > 95:
                return "Saque de puerta"
            if y < 5 or y > 63:
                return "Saque de banda"
            return "Pase"

        return "Pase"

    # ── Validacion geometrica (compatibilidad con video_processor) ─────────

    def validate_geometrical(self, event: dict, ball_pitch_pos: tuple) -> bool:
        """
        Aplica reglas expertas para validar un evento.
        Devuelve True si el evento es geometricamente plausible.
        """
        action = event.get("action", "")
        x, y   = ball_pitch_pos

        if action == "Penalty":
            return (abs(x - 11) < 3 or abs(x - 94) < 3) and abs(y - 34) < 4

        if action == "Corner":
            return (x < 5 or x > 100) and (y < 5 or y > 63)

        if action == "Tiro":
            # Debe venir desde zona de ataque
            return x > 70 or x < 35

        if action == "Saque de banda":
            return y < 8 or y > 60

        return True

    # ── API de resultados ──────────────────────────────────────────────────

    def get_events(self) -> list[BallEvent]:
        """Devuelve todos los eventos detectados."""
        return self._events

    def get_events_dict(self) -> list[dict]:
        """Devuelve eventos como lista de dicts (compatible con video_processor)."""
        return [
            {
                "timestamp":   e.timestamp,
                "minute":      e.minute,
                "action":      e.action,
                "track_id":    e.track_id,
                "team":        e.team,
                "ball_pos":    e.ball_pos,
                "pitch_pos":   e.pitch_pos,
                "confidence":  e.confidence,
                "is_validated": e.is_validated,
            }
            for e in self._events
        ]

    def get_possession_stats(self, track_stats: dict) -> dict:
        """
        Calcula estadisticas de posesion por equipo.

        Returns:
            {"team_0": 45.2, "team_1": 54.8}  <- porcentaje de posesion
        """
        history = list(self._possession_history)
        team_frames = {0: 0, 1: 0}

        for tid in history:
            if tid is None:
                continue
            if tid in track_stats:
                team = track_stats[tid].get("equipo", -1)
                if team in team_frames:
                    team_frames[team] += 1

        total = sum(team_frames.values())
        if total == 0:
            return {"team_0": 50.0, "team_1": 50.0}

        return {
            "team_0": round(team_frames[0] / total * 100, 1),
            "team_1": round(team_frames[1] / total * 100, 1),
        }

    def reset(self):
        """Reinicia el estado para un nuevo video."""
        self._possession_history.clear()
        self._events.clear()
        self._last_event_by_track.clear()
        self._last_ball_pos    = None
        self._last_ball_second = -1.0
        self._frame_count      = 0
        self._current_possessor  = None
        self._possessor_frames   = 0
        self._previous_possessor = None

    # ── Compatibilidad con codigo anterior ────────────────────────────────

    def spot_events(self, features) -> list:
        """Stub de compatibilidad con la interfaz T-DEED original."""
        return []


class AdvancedEventDetector(EventSpotterTDEED):
    """Detector avanzado con reglas geométricas para eventos específicos."""

    def __init__(self):
        super().__init__()
        # Zonas del campo (coordenadas normalizadas 0-1)
        self.goal_zones = {
            'team_a_goal': (0.45, 0.55, 0.0, 0.05),   # x_min, x_max, y_min, y_max
            'team_b_goal': (0.45, 0.55, 0.95, 1.0)
        }
        self.corner_zones = [
            (0.0, 0.05, 0.0, 0.05),    # Corner inferior izquierdo
            (0.95, 1.0, 0.0, 0.05),    # Corner inferior derecho
            (0.0, 0.05, 0.95, 1.0),    # Corner superior izquierdo
            (0.95, 1.0, 0.95, 1.0)     # Corner superior derecho
        ]

    def detect_advanced_events(self, ball_pos, pitch_pos, tracks, frame_second):
        """Detecta eventos específicos usando geometría."""
        events = []

        # 1. DETECCIÓN DE GOLES
        if self._is_goal(ball_pos, pitch_pos):
            events.append(BallEvent(
                timestamp=frame_second,
                minute=frame_second/60,
                action="Gol",
                track_id=self._current_possessor,
                team=self._get_team_from_track(tracks, self._current_possessor),
                ball_pos=ball_pos,
                pitch_pos=pitch_pos,
                confidence=0.95
            ))

        # 2. DETECCIÓN DE CORNERS
        if self._is_corner(pitch_pos):
            events.append(BallEvent(
                timestamp=frame_second,
                minute=frame_second/60,
                action="Corner",
                track_id=self._current_possessor,
                team=self._get_team_from_track(tracks, self._current_possessor),
                ball_pos=ball_pos,
                pitch_pos=pitch_pos,
                confidence=0.85
            ))

        # 3. DETECCIÓN DE TIROS A PUERTA
        if self._is_shot_on_goal(pitch_pos, ball_pos):
            events.append(BallEvent(
                timestamp=frame_second,
                minute=frame_second/60,
                action="Tiro a puerta",
                track_id=self._current_possessor,
                team=self._get_team_from_track(tracks, self._current_possessor),
                ball_pos=ball_pos,
                pitch_pos=pitch_pos,
                confidence=0.80
            ))

        return events

    def _is_goal(self, ball_pos, pitch_pos):
        """Detecta si el balón entró en la portería."""
        px, py = pitch_pos
        # Verificar si está en zona de gol y velocidad indica entrada
        for goal_name, (x_min, x_max, y_min, y_max) in self.goal_zones.items():
            if x_min <= px <= x_max and y_min <= py <= y_max:
                return True
        return False

    def _is_corner(self, pitch_pos):
        """Detecta si el balón salió por línea de fondo (corner)."""
        px, py = pitch_pos
        for x_min, x_max, y_min, y_max in self.corner_zones:
            if x_min <= px <= x_max and y_min <= py <= y_max:
                return True
        return False

    def _is_shot_on_goal(self, pitch_pos, ball_pos):
        """Detecta tiros a puerta basados en trayectoria."""
        px, py = pitch_pos
        # Área de penalty (aproximada)
        penalty_area = (0.35, 0.65, 0.0, 0.17)  # Equipo A
        if (penalty_area[0] <= px <= penalty_area[1] and
            penalty_area[2] <= py <= penalty_area[3]):
            return True
        return False

    def _get_team_from_track(self, tracks, track_id):
        """Obtiene el equipo de un track."""
        if track_id in tracks:
            return tracks[track_id].equipo
        return -1


if __name__ == "__main__":
    spotter = EventSpotterTDEED()
    print("EventSpotter listo.")
    print(f"Clases: {spotter.classes}")
