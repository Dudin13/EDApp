"""
event_engine.py — Detección de eventos basada en reglas.
Posesión, Pases, Recuperaciones e Intercepciones.
"""

import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FootballEvent:
    timestamp: float
    minute: float
    type: str  # 'pass', 'tackle', 'interception', 'possession'
    from_tid: int
    to_tid: int | None = None
    team: int | None = None
    pitch_pos: tuple[float, float] | None = None

class EventEngine:
    """
    Analiza las trayectorias de jugadores y balón para detectar eventos tácticos.
    """
    def __init__(self, possession_radius: float = 2.0, change_threshold: int = 3):
        """
        Args:
            possession_radius: Distancia máxima (metros) para considerar posesión.
            change_threshold: Frames consecutivos para validar un cambio de posesión.
        """
        self.possession_radius = possession_radius
        self.change_threshold = change_threshold

    def detect_events(self, tracks: dict, interpolated_ball: list) -> list[dict]:
        """
        Analiza el partido completo para extraer eventos.
        
        Args:
            tracks: dict de track_id -> datos (incluyendo pitch_x, pitch_y)
            interpolated_ball: list de dicts con (minute, pitch_x, pitch_y)
        """
        if not interpolated_ball:
            return []

        # 1. Registro de posesión frame a frame
        possession_log = []
        for ball_frame in interpolated_ball:
            m = ball_frame["minute"]
            bx, by = ball_frame["pitch_x"], ball_frame["pitch_y"]

            closest_tid = None
            closest_dist = float("inf")
            closest_team = None

            for tid, track in tracks.items():
                # Encontrar posición del jugador en este minuto
                # (Búsqueda simple por ahora, asumiendo sincronía de índices)
                try:
                    # Encontrar el índice más cercano en el historial del jugador
                    idx = np.argmin(np.abs(np.array(track["history_minute"]) - m))
                    px, py = track["pitch_x"][idx], track["pitch_y"][idx]
                    
                    dist = np.sqrt((bx - px)**2 + (by - py)**2)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_tid = tid
                        closest_team = track.get("equipo")
                except (ValueError, IndexError):
                    continue

            if closest_dist <= self.possession_radius:
                possession_log.append({
                    "minute": m,
                    "tid": closest_tid,
                    "team": closest_team,
                    "dist": closest_dist,
                    "pos": (bx, by)
                })

        # 2. Detectar cambios de posesión con debouncing
        events = []
        if len(possession_log) < self.change_threshold:
            return []

        current_possessor = possession_log[0]["tid"]
        current_team = possession_log[0]["team"]
        
        candidate_tid = None
        candidate_count = 0

        for entry in possession_log[1:]:
            tid = entry["tid"]
            team = entry["team"]

            if tid == current_possessor:
                candidate_count = 0
                continue
            
            if tid == candidate_tid:
                candidate_count += 1
                if candidate_count >= self.change_threshold:
                    # Cambio de posesión detectado
                    event_type = "pass" if team == current_team else "turnover"
                    
                    events.append({
                        "minute": entry["minute"],
                        "from_tid": current_possessor,
                        "to_tid": tid,
                        "team": team,
                        "action": "Pase" if event_type == "pass" else "Recuperación",
                        "pitch_pos": entry["pos"]
                    })
                    
                    current_possessor = tid
                    current_team = team
                    candidate_count = 0
            else:
                candidate_tid = tid
                candidate_count = 1

        # 3. Refinar 'turnover' en 'Tackle' vs 'Intercepción'
        # (Lógica simplificada: si están cerca en el momento del cambio es tackle)
        for ev in events:
            if ev["action"] == "Recuperación":
                # Por ahora mantenemos la etiqueta genérica o refinamos si hay datos de dist. entre jugadores
                pass

        return events
