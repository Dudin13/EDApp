"""
event_engine.py — Detección de eventos basada en reglas.
Posesión, Pases, Recuperaciones e Intercepciones.
"""

import numpy as np
import logging
from dataclasses import dataclass
from math import sqrt, degrees, atan2

logger = logging.getLogger(__name__)

def calculate_xg(x, y, team):
    # Portería equipo B: x=105, y=34
    # Portería equipo A: x=0, y=34
    if team == 'A':
        gx, gy = 105, 34
    else:
        gx, gy = 0, 34
    
    distancia = sqrt((gx-x)**2 + (gy-y)**2)
    if distancia == 0:
        return 0.95
    angulo = degrees(atan2(3.66, distancia))
    xG = round(min(0.95, angulo / distancia * 8), 3)
    return xG

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

        BALL_PROXIMITY = 0.75  # metros

        # 1. Registro de posesión frame a frame y detección de duelos
        possession_log = []
        duels = []
        
        in_duel = False
        duel_players = set()
        duel_frames = []
        possessor_before = None
        last_possessor = None
        
        last_shot_minute = -999.0

        for idx_f, ball_frame in enumerate(interpolated_ball):
            m = ball_frame["minute"]
            bx, by = ball_frame["pitch_x"], ball_frame["pitch_y"]

            closest_tid = None
            closest_dist = float("inf")
            closest_team = None
            players_close = []

            for tid, track in tracks.items():
                try:
                    # Encontrar el índice más cercano en el historial del jugador
                    idx = np.argmin(np.abs(np.array(track["history_minute"]) - m))
                    if np.abs(track["history_minute"][idx] - m) > (1.0 / 60.0):
                        continue
                    px, py = track["pitch_x"][idx], track["pitch_y"][idx]
                    
                    dist = np.sqrt((bx - px)**2 + (by - py)**2)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_tid = tid
                        closest_team = track.get("equipo")
                    
                    if dist < BALL_PROXIMITY:
                        players_close.append({
                            "tid": tid,
                            "team": track.get("equipo"),
                            "dist": dist
                        })
                except (ValueError, IndexError):
                    continue

            # Registrar posesión si está dentro del radio de posesión
            if closest_dist <= self.possession_radius:
                possession_log.append({
                    "minute": m,
                    "tid": closest_tid,
                    "team": closest_team,
                    "dist": closest_dist,
                    "pos": (bx, by)
                })
                if not in_duel:
                    last_possessor = closest_tid

            # Detección de Tiro si supera 4 m/s hacia portería
            speed = 0.0
            heading_goal = False
            if idx_f > 0:
                prev_frame = interpolated_ball[idx_f - 1]
                dt = (m - prev_frame["minute"]) * 60.0  # en segundos
                if dt > 0:
                    dist = np.sqrt((bx - prev_frame["pitch_x"])**2 + (by - prev_frame["pitch_y"])**2)
                    speed = dist / dt
                    
                    # Determinar si va hacia la portería
                    if closest_team == 0:  # Equipo A (0) ataca portería B (x=105)
                        if bx > 90 and bx > prev_frame["pitch_x"]:
                            heading_goal = True
                    elif closest_team == 1:  # Equipo B (1) ataca portería A (x=0)
                        if bx < 15 and bx < prev_frame["pitch_x"]:
                            heading_goal = True

            if speed > 4.0 and heading_goal and (m - last_shot_minute > 5.0 / 60.0):
                team_str = 'A' if closest_team == 0 else 'B'
                xg_val = calculate_xg(bx, by, team_str)
                duels.append({
                    "minute": m,
                    "from_tid": closest_tid,
                    "to_tid": None,
                    "team": closest_team,
                    "action": "Tiro",
                    "pitch_pos": (bx, by),
                    "xg": xg_val,
                    "footpass_class": "Shot"
                })
                last_shot_minute = m

            # Detección de duelos (2 o más jugadores a < 0.75m del balón)
            if len(players_close) >= 2:
                if not in_duel:
                    in_duel = True
                    possessor_before = last_possessor
                    duel_players = set()
                    duel_frames = []
                for p in players_close:
                    duel_players.add((p["tid"], p["team"]))
                duel_frames.append({
                    "minute": m,
                    "players": players_close,
                    "pos": (bx, by)
                })
            else:
                if in_duel:
                    # El duelo ha terminado. Determinar ganador.
                    winner = None
                    if closest_dist <= self.possession_radius:
                        winner = closest_tid
                    
                    if winner is None:
                        # Buscar en los siguientes 5 frames
                        for next_f in interpolated_ball[idx_f : min(idx_f + 6, len(interpolated_ball))]:
                            nm = next_f["minute"]
                            nbx, nby = next_f["pitch_x"], next_f["pitch_y"]
                            next_closest_tid = None
                            next_closest_dist = float("inf")
                            for tid, track in tracks.items():
                                try:
                                    n_idx = np.argmin(np.abs(np.array(track["history_minute"]) - nm))
                                    if np.abs(track["history_minute"][n_idx] - nm) > (1.0 / 60.0):
                                        continue
                                    npx, npy = track["pitch_x"][n_idx], track["pitch_y"][n_idx]
                                    ndist = np.sqrt((nbx - npx)**2 + (nby - npy)**2)
                                    if ndist < next_closest_dist:
                                        next_closest_dist = ndist
                                        next_closest_tid = tid
                                except:
                                    continue
                            if next_closest_dist <= self.possession_radius:
                                winner = next_closest_tid
                                break
                    
                    # Fallback al jugador más cercano en el último frame del duelo
                    if winner is None and duel_frames:
                        last_df = duel_frames[-1]
                        last_df["players"].sort(key=lambda x: x["dist"])
                        winner = last_df["players"][0]["tid"]
                    
                    if duel_frames:
                        mid_idx = len(duel_frames) // 2
                        duel_min = duel_frames[0]["minute"]
                        duel_pos = duel_frames[mid_idx]["pos"]
                        
                        duels.append({
                            "minute": duel_min,
                            "from_tid": winner,
                            "to_tid": None,
                            "team": tracks[winner]["equipo"] if (winner is not None and winner in tracks) else -1,
                            "action": "Duelo",
                            "pitch_pos": duel_pos,
                            "duel_players": [{"tid": p[0], "team": p[1]} for p in duel_players],
                            "possessor_before": possessor_before,
                            "winner": winner,
                            "footpass_class": "Duel"
                        })
                    in_duel = False

        # 2. Detectar cambios de posesión con debouncing
        events = []
        if len(possession_log) < self.change_threshold:
            return sorted(duels, key=lambda x: x["minute"])

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
                    action_name = "Pase" if event_type == "pass" else "Recuperación"
                    
                    # Encontrar el inicio del pase en possession_log
                    start_pos = None
                    for prev_entry in reversed(possession_log[:possession_log.index(entry)]):
                        if prev_entry["tid"] == current_possessor:
                            start_pos = prev_entry["pos"]
                            break
                    if start_pos is None:
                        start_pos = entry["pos"]
                    
                    start_x, start_y = start_pos
                    dest_x, dest_y = entry["pos"]
                    
                    footpass_class = "Pass"
                    if event_type == "pass":
                        # Centro (CROSS): pitch_pos x > 80 o x < 25 e Y de destino en central (30<y<38)
                        if (start_x > 80 or start_x < 25) and (30 < dest_y < 38):
                            footpass_class = "Cross"
                        # Saque de banda (THROW_IN): start_y < 5 o start_y > 63
                        elif start_y < 5 or start_y > 63:
                            action_name = "Saque_banda"
                            footpass_class = "Throw-in"
                    else:
                        footpass_class = "Recovery"
                    
                    events.append({
                        "minute": entry["minute"],
                        "from_tid": current_possessor,
                        "to_tid": tid,
                        "team": team,
                        "action": action_name,
                        "pitch_pos": entry["pos"],
                        "footpass_class": footpass_class
                    })
                    
                    current_possessor = tid
                    current_team = team
                    candidate_count = 0
            else:
                candidate_tid = tid
                candidate_count = 1

        # Combinar y ordenar
        all_events = events + duels
        all_events.sort(key=lambda x: x["minute"])
        return all_events
