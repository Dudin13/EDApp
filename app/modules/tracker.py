"""
tracker.py — Tracker avanzado basado en ByteTrack (via supervision).

FIXES APLICADOS:
  [CRÍTICO] Inicialización híbrida: Soporta ByteTrack y UCMCTrack en paralelo.
  [MEJORA]  Adapter para UCMCTrack que mapea detecciones al plano del suelo.
"""

import logging
import sys
import os
import numpy as np
import supervision as sv

from pathlib import Path
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
from core.config_loader import config
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path

# ── Fix 8: Integración UCMCTrack ─────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_UCMC_PATH = _REPO_ROOT / "UCMCTrack"
if _UCMC_PATH.exists() and str(_UCMC_PATH) not in sys.path:
    sys.path.insert(0, str(_UCMC_PATH))
else:
    logging.warning(f"UCMC PATH NO EXISTE O YA ESTA: {_UCMC_PATH}")

try:
    from tracker.ucmc import UCMCTrack as UCMCInternal
except ImportError as e:
    UCMCInternal = None
    logging.warning(f"UCMCTrack no encontrado en {_UCMC_PATH}. Error: {e}")

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """
    Persistent state for a single tracked entity across frames.
    """
    track_id: int
    clase: str
    equipo: int          # 0=local, 1=visitante, 2=árbitro
    last_box: tuple      # (cx, cy, w, h)
    frames_seen: int = 1
    frames_lost: int = 0
    history_x: list = field(default_factory=list)
    history_y: list = field(default_factory=list)
    history_pitch_x: list = field(default_factory=list)
    history_pitch_y: list = field(default_factory=list)
    history_minute: list = field(default_factory=list)
    appearance_color: Optional[tuple] = None
    _equipo_votes: list = field(default_factory=list)
    _VOTE_WINDOW: int = field(default=8, init=False, repr=False)

    def vote_equipo(self, equipo_pred: int) -> None:
        if self.clase == "referee":
            self.equipo = 2
            return
        self._equipo_votes.append(equipo_pred)
        if len(self._equipo_votes) > self._VOTE_WINDOW:
            self._equipo_votes.pop(0)
        votes = [v for v in self._equipo_votes if v != 2]
        if votes:
            self.equipo = max(set(votes), key=votes.count)


class UCMCAdapter:
    """
    Bridge para UCMCTrack.
    """
    def __init__(self, fps=30, a=100.0, high_score=0.5):
        if UCMCInternal is None:
            raise ImportError("UCMCTrack no está disponible.")
        self.tracker = UCMCInternal(
            a1=a, a2=a, wx=5, wy=5, vmax=10, 
            max_age=30, fps=fps, dataset="MOT", 
            high_score=high_score, use_cmc=False, detector=None
        )
        self.frame_id = 1

    def update(self, detecciones: list):
        class UCMCDetection:
            def __init__(self, d):
                self.conf = d.get("conf", 0.5)
                # Asegurar que ucmc_y es un array 2x1 de floats
                y_raw = d.get("ucmc_y", [float(d["x"]), float(d["y"])])
                self.y = np.array(y_raw).reshape(2, 1).astype(np.float32)
                self.R = d.get("R", np.eye(2, dtype=np.float32))
                self.bb_width = d.get("w", 40)
                self.bb_height = d.get("h", 80)
                self.track_id = 0
        
        ucmc_dets = [UCMCDetection(d) for d in detecciones]
        self.tracker.update(ucmc_dets, self.frame_id)
        self.frame_id += 1
        return [d.track_id for d in ucmc_dets]


class ProfessionalTracker:
    """
    Tracker híbrido (ByteTrack / UCMCTrack).
    """
    _DEFAULT_TRACK_ACTIVATION = 0.20
    _DEFAULT_MATCH_THRESHOLD  = 0.8

    def __init__(self, sample_rate: float = 0.5, mode: str = None):
        self.sample_rate = sample_rate
        self.mode = mode if mode else config.tracking.engine
        self._frame_rate = max(1, round(1.0 / sample_rate))

        self._bt_params = dict(
            track_activation_threshold=self._DEFAULT_TRACK_ACTIVATION,
            lost_track_buffer=config.tracking.lost_track_buffer,
            minimum_matching_threshold=self._DEFAULT_MATCH_THRESHOLD,
            frame_rate=self._frame_rate,
        )
        self._bt_tracker = sv.ByteTrack(**self._bt_params)
        
        self._ucmc_tracker = None
        if self.mode == "ucmctrack" and UCMCInternal:
            try:
                self._ucmc_tracker = UCMCAdapter(fps=self._frame_rate)
            except Exception as e:
                logger.error(f"Error UCMCTrack: {e}")
                self.mode = "bytetrack"

        self._tracks: Dict[int, Track] = {}
        self._history: Dict[int, Track] = {}

    def update(self, detecciones, equipo_map, minute=0.0, camera_offset=(0,0)):
        if not detecciones:
            self._tracks = {}
            return self._tracks

        if self.mode == "ucmctrack" and self._ucmc_tracker:
            track_ids = self._ucmc_tracker.update(detecciones)
            current_active = {}
            for i, tid in enumerate(track_ids):
                if tid <= 0: continue
                d = detecciones[i]
                current_active[tid] = self._process_track_update(tid, d["x"], d["y"], d["w"], d["h"], detecciones, equipo_map, minute)
            self._tracks = current_active
            return self._tracks
        else:
            # ByteTrack path
            dx, dy = camera_offset
            if dx != 0 or dy != 0:
                for attr in ("tracked_tracks", "lost_tracks"):
                    track_list = getattr(self._bt_tracker, attr, None)
                    if track_list:
                        for track in track_list:
                            if hasattr(track, "mean") and len(track.mean) >= 2:
                                track.mean[0] += dx
                                track.mean[1] += dy

            class_name_to_id = {"player": 0, "goalkeeper": 1, "referee": 2, "ball": 3}
            xyxy, confs, clss = [], [], []
            for d in detecciones:
                xyxy.append([d["x"]-d["w"]/2, d["y"]-d["h"]/2, d["x"]+d["w"]/2, d["y"]+d["h"]/2])
                confs.append(d.get("conf", 0.5))
                clss.append(class_name_to_id.get(d.get("clase"), 0))

            sv_dets = sv.Detections(xyxy=np.array(xyxy, dtype=np.float32), 
                                   confidence=np.array(confs, dtype=np.float32), 
                                   class_id=np.array(clss, dtype=int))
            sv_dets = self._bt_tracker.update_with_detections(sv_dets)
            
            current_active = {}
            for i in range(len(sv_dets)):
                tid = int(sv_dets.tracker_id[i]) if sv_dets.tracker_id[i] is not None else -1
                if tid < 0: continue
                x1, y1, x2, y2 = sv_dets.xyxy[i]
                current_active[tid] = self._process_track_update(tid, (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1, detecciones, equipo_map, minute)
            
            self._tracks = current_active
            return self._tracks

    def _process_track_update(self, tid, cx, cy, w, h, detecciones, equipo_map, minute):
        if tid not in self._history:
            equipo, clase = self._match_equipo_clase(cx, cy, detecciones, equipo_map)
            track = Track(track_id=tid, clase=clase, equipo=equipo, last_box=(cx, cy, w, h),
                          history_x=[cx], history_y=[cy], history_minute=[minute])
            track.vote_equipo(equipo)
            self._history[tid] = track
        else:
            track = self._history[tid]
            track.last_box = (cx, cy, w, h)
            track.history_x.append(cx)
            track.history_y.append(cy)
            track.history_minute.append(minute)
            track.frames_seen += 1
            e, _ = self._match_equipo_clase(cx, cy, detecciones, equipo_map)
            track.vote_equipo(e)
        return track

    def _match_equipo_clase(self, cx, cy, detecciones, equipo_map):
        if not detecciones: return 0, "player"
        dists = [np.hypot(cx-d["x"], cy-d["y"]) for d in detecciones]
        idx = np.argmin(dists)
        return (equipo_map[idx] if idx < len(equipo_map) else 0), detecciones[idx].get("clase", "player")

    def get_all_tracks(self) -> List[Track]:
        return list(self._history.values())

    def reset(self):
        if self.mode == "ucmctrack" and self._ucmc_tracker:
            self._ucmc_tracker = UCMCAdapter(fps=self._frame_rate)
        else:
            self._bt_tracker = sv.ByteTrack(**self._bt_params)
        self._tracks = {}
        self._history = {}

    def initialize_with_seeds(self, seeds):
        if not seeds or self.mode == "ucmctrack": return
        xyxy = [[s["x"]-20, s["y"]-40, s["x"]+20, s["y"]+40] for s in seeds]
        sv_seeds = sv.Detections(xyxy=np.array(xyxy, dtype=np.float32), 
                                confidence=np.ones(len(seeds), dtype=np.float32)*0.95, 
                                class_id=np.zeros(len(seeds), dtype=int))
        self._bt_tracker.update_with_detections(sv_seeds)


SimpleTracker = ProfessionalTracker
