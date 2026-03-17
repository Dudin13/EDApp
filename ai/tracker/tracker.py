import numpy as np
import supervision as sv
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from core.logger import logger

@dataclass
class Track:
    track_id: int
    clase: str
    equipo: int          # 0=local, 1=visitante, 2=árbitro
    last_box: tuple      # (cx, cy, w, h)
    frames_seen: int = 1
    frames_lost: int = 0
    history_x: list = field(default_factory=list)
    history_y: list = field(default_factory=list)
    history_minute: list = field(default_factory=list)
    appearance_color: Optional[tuple] = None

class PlayerTracker:
    """
    Tracker profesional basado en ByteTrack con compensación de cámara.
    """
    def __init__(self, sample_rate: float = 0.5):
        self.sample_rate = sample_rate
        self.frame_rate = max(1, round(1.0 / sample_rate))
        
        self.bt_params = dict(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=self.frame_rate,
        )
        self.tracker = sv.ByteTrack(**self.bt_params)
        self.tracks: Dict[int, Track] = {}
        self.history: Dict[int, Track] = {}

    def update(
        self,
        detecciones: list,
        equipo_map: List[int],
        minute: float = 0.0,
        camera_offset: tuple = (0.0, 0.0)
    ) -> Dict[int, Track]:
        
        dx, dy = camera_offset
        if (dx != 0.0 or dy != 0.0):
            self._compensate_camera_motion(dx, dy)

        if not detecciones:
            return self.tracks

        # Convert to supervision format
        xyxy, confidence, class_id = [], [], []
        for d in detecciones:
            x1, y1 = d["x"] - d["w"] / 2, d["y"] - d["h"] / 2
            x2, y2 = d["x"] + d["w"] / 2, d["y"] + d["h"] / 2
            xyxy.append([x1, y1, x2, y2])
            confidence.append(d.get("conf", 0.5))
            class_id.append(0)

        sv_detections = sv.Detections(
            xyxy=np.array(xyxy, dtype=np.float32),
            confidence=np.array(confidence, dtype=np.float32),
            class_id=np.array(class_id, dtype=int),
        )

        sv_detections = self.tracker.update_with_detections(sv_detections)

        current_active = {}
        for i in range(len(sv_detections)):
            tid = int(sv_detections.tracker_id[i])
            x1, y1, x2, y2 = sv_detections.xyxy[i]
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w / 2, y1 + h / 2

            if tid not in self.history:
                equipo = self._match_equipo(cx, cy, detecciones, equipo_map)
                self.history[tid] = Track(
                    track_id=tid,
                    clase="player",
                    equipo=equipo,
                    last_box=(cx, cy, w, h),
                    history_x=[cx],
                    history_y=[cy],
                    history_minute=[minute],
                )
            else:
                track = self.history[tid]
                track.last_box = (cx, cy, w, h)
                track.history_x.append(cx)
                track.history_y.append(cy)
                track.history_minute.append(minute)
                track.frames_seen += 1

            current_active[tid] = self.history[tid]

        self.tracks = current_active
        return self.tracks

    def _compensate_camera_motion(self, dx: float, dy: float):
        try:
            for track_list in [getattr(self.tracker, "tracked_tracks", []), 
                              getattr(self.tracker, "lost_tracks", [])]:
                for track in track_list:
                    if hasattr(track, "mean") and len(track.mean) >= 2:
                        track.mean[0] += dx
                        track.mean[1] += dy
        except Exception as e:
            logger.warning(f"Error compensando movimiento de cámara: {e}")

    def _match_equipo(self, cx: float, cy: float, detecciones: list, equipo_map: List[int]) -> int:
        if not detecciones or not equipo_map: return 0
        distances = [np.hypot(cx - d["x"], cy - d["y"]) for d in detecciones]
        idx = np.argmin(distances)
        return equipo_map[idx] if idx < len(equipo_map) else 0

    def reset(self):
        self.tracker = sv.ByteTrack(**self.bt_params)
        self.tracks = {}
        self.history = {}
