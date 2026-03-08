import numpy as np
import supervision as sv
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class Track:
    track_id: int
    clase: str
    equipo: int          # 0, 1 o 2 (árbitro)
    last_box: tuple      # (x, y, w, h) último frame
    frames_seen: int = 1
    frames_lost: int = 0
    history_x: list = field(default_factory=list)
    history_y: list = field(default_factory=list)
    history_minute: list = field(default_factory=list)
    appearance_color: Optional[tuple] = None  # (h, s, v) medio del torso


class ProfessionalTracker:
    """
    Tracker avanzado basado en ByteTrack (via supervision).
    Utiliza predicción de Kalman y lógica persistente para no perder IDs.
    """

    def __init__(self):
        self._tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,  # Frames a esperar antes de borrar (aprox 15s a 0.5fps)
            minimum_matching_threshold=0.8, # IoU mínimo para asociar
            frame_rate=2 # Ajustado a nuestro sample rate típico
        )
        self._tracks: Dict[int, Track] = {}
        self._history: Dict[int, Track] = {} # Para persistencia final

    def update(self, detecciones: list, equipo_map: list, minute: float = 0.0, camera_offset: tuple = (0.0, 0.0)) -> Dict[int, Track]:
        """
        Actualiza el tracker usando ByteTrack.
        Aplica compensación de movimiento de cámara si se provee.
        """
        if not detecciones:
            return self._tracks

        # Aplicar Optical Flow (Compensación de cámara) a los estados internos del Kalman Filter si es posible
        dx, dy = camera_offset
        if (dx != 0.0 or dy != 0.0) and hasattr(self._tracker, 'tracked_tracks'):
            try:
                # ByteTrack guarda el estado [x, y, a, h, vx, vy, va, vh]
                for track_list in [self._tracker.tracked_tracks, self._tracker.lost_tracks]:
                    for track in track_list:
                        # ByteTrack usa Kalman Filter interno. track.mean es [x,y,a,h,vx,vy,va,vh]
                        if hasattr(track, 'mean') and len(track.mean) >= 2:
                            track.mean[0] += dx
                            track.mean[1] += dy
            except Exception:
                pass

        # Convertir nuestras detecciones al formato de supervision
        xyxy = []
        confidence = []
        class_id = []
        
        for d in detecciones:
            # cx, cy, w, h -> x1, y1, x2, y2
            x1 = d["x"] - d["w"] / 2
            y1 = d["y"] - d["h"] / 2
            x2 = d["x"] + d["w"] / 2
            y2 = d["y"] + d["h"] / 2
            xyxy.append([x1, y1, x2, y2])
            confidence.append(d.get("conf", 0.5))
            class_id.append(0) # Asumimos 'player' por simplicidad en el tracker

        sv_detections = sv.Detections(
            xyxy=np.array(xyxy),
            confidence=np.array(confidence),
            class_id=np.array(class_id)
        )

        # Ejecutar ByteTrack
        sv_detections = self._tracker.update_with_detections(sv_detections)

        # Mapear IDs de ByteTrack a nuestra estructura Track
        current_active = {}
        
        for i in range(len(sv_detections)):
            tid = int(sv_detections.tracker_id[i])
            box = sv_detections.xyxy[i]
            # Convertir de vuelta a cx, cy, w, h
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w/2, y1 + h/2

            # Buscar si ya existe o es nuevo
            if tid not in self._history:
                # Intentar encontrar equipo original (ByteTrack no cambia el orden de dets si coinciden 1:1)
                # NOTA: En produccion real usariamos detecciones originales vs tracks finales
                # Por ahora, buscamos la deteccion mas cercana espacialmente en este frame
                equipo = 0
                dist_min = float('inf')
                for d_idx, d in enumerate(detecciones):
                    dist = np.sqrt((cx - d["x"])**2 + (cy - d["y"])**2)
                    if dist < dist_min:
                        dist_min = dist
                        equipo = equipo_map[d_idx]

                self._history[tid] = Track(
                    track_id=tid,
                    clase="player",
                    equipo=equipo,
                    last_box=(cx, cy, w, h),
                    history_x=[cx],
                    history_y=[cy],
                    history_minute=[minute]
                )
            else:
                track = self._history[tid]
                track.last_box = (cx, cy, w, h)
                track.history_x.append(cx)
                track.history_y.append(cy)
                track.history_minute.append(minute)
                track.frames_seen += 1
            
            current_active[tid] = self._history[tid]

        self._tracks = current_active
        return self._tracks

    def get_all_tracks(self) -> List[Track]:
        """Retorna todos los tracks detectados en la sesión."""
        return list(self._history.values())

    def reset(self):
        self._tracker = sv.ByteTrack()
        self._tracks = {}
        self._history = {}

    def initialize_with_seeds(self, seeds: List[Dict]):
        """
        ByteTrack es difícil de pre-inicializar manualmente sin inyectar detecciones falsas.
        Por ahora, dejamos que ByteTrack cree los IDs automáticamente en el primer frame.
        """
        pass


# Alias para mantener compatibilidad si se usa SimpleTracker en otros archivos
SimpleTracker = ProfessionalTracker
