"""
tracker.py — Tracking simple de jugadores entre frames consecutivos.

Asocia detecciones de frames distintos usando IoU (Intersection over Union)
para mantener IDs consistentes a lo largo del vídeo. No requiere ByteTrack.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class Track:
    track_id: int
    clase: str
    equipo: int          # 0, 1 o 2 (árbitro)
    last_box: tuple      # (x, y, w, h) último frame
    nombre: Optional[str] = None  # Asignación manual de identidad
    frames_seen: int = 1
    frames_lost: int = 0
    history_x: list = field(default_factory=list)
    history_y: list = field(default_factory=list)
    history_minute: list = field(default_factory=list)


def _iou(box_a: tuple, box_b: tuple) -> float:
    """Calcula IoU entre dos bounding boxes (cx, cy, w, h)."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax1, ay1 = ax - aw / 2, ay - ah / 2
    ax2, ay2 = ax + aw / 2, ay + ah / 2
    bx1, by1 = bx - bw / 2, by - bh / 2
    bx2, by2 = bx + bw / 2, by + bh / 2

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def _distance(box_a: tuple, box_b: tuple) -> float:
    """Calcula la distancia euclídea entre los centros de dos bounding boxes."""
    return ((box_a[0] - box_b[0]) ** 2 + (box_a[1] - box_b[1]) ** 2) ** 0.5


class SimpleTracker:
    """
    Tracker ligero generalizado.
    Mantiene tracks activos durante MAX_LOST frames sin detección.
    Utiliza Distancia Euclídea para asociar tracks, dado que IoU 
    falla al usar intervalos grandes (ej 2s).
    """

    MAX_LOST = 3       # frames sin deteccion antes de eliminar el track
    MAX_DIST_PX = 400  # distancia maxima de centroide permitida para asociar

    def __init__(self):
        self._next_id = 1
        self._tracks: Dict[int, Track] = {}

    def update(self, detecciones: list, equipo_map: list, minute: float = 0.0) -> Dict[int, Track]:
        """
        Actualiza el tracker con las detecciones del frame actual.

        Args:
            detecciones: lista de dicts con {x, y, w, h, clase}
            equipo_map: lista de ints con el equipo de cada detección (misma longitud)
            minute: minuto del vídeo

        Returns:
            dict {track_id: Track} con los tracks activos
        """
        active_ids = set()

        if not detecciones:
            # Incrementar contador de frames perdidos para todos los tracks
            to_delete = []
            for tid, track in self._tracks.items():
                track.frames_lost += 1
                if track.frames_lost > self.MAX_LOST:
                    to_delete.append(tid)
            for tid in to_delete:
                del self._tracks[tid]
            return self._tracks

        # Construir matriz de Distancias
        track_list = list(self._tracks.values())
        n_tracks = len(track_list)
        n_dets = len(detecciones)

        dist_matrix = np.zeros((n_tracks, n_dets))
        for i, track in enumerate(track_list):
            for j, det in enumerate(detecciones):
                dist = _distance(track.last_box, (det["x"], det["y"], det["w"], det["h"]))
                # Penalización grande si la red neuronal dice que son de equipos diferentes
                # para evitar que el ID salte de un equipo a otro si se cruzan.
                if track.equipo != equipo_map[j]:
                    dist += 400
                dist_matrix[i, j] = dist

        matched_tracks = set()
        matched_dets = set()

        # Asignación greedy: menor Distancia primero
        if n_tracks > 0:
            flat_indices = np.argsort(dist_matrix, axis=None)
            for idx in flat_indices:
                i, j = divmod(int(idx), n_dets)
                if dist_matrix[i, j] > self.MAX_DIST_PX + 400: # max allow penalty included
                    break
                    
                # Strict check without penalty to make sure it's somewhat realistic 
                real_dist = _distance(track_list[i].last_box, (detecciones[j]["x"], detecciones[j]["y"]))
                if real_dist > self.MAX_DIST_PX:
                    continue

                if i not in matched_tracks and j not in matched_dets:
                    track = track_list[i]
                    det = detecciones[j]
                    track.last_box = (det["x"], det["y"], det["w"], det["h"])
                    track.frames_lost = 0
                    track.frames_seen += 1
                    track.history_x.append(det["x"])
                    track.history_y.append(det["y"])
                    track.history_minute.append(minute)
                    track.equipo = equipo_map[j]
                    active_ids.add(track.track_id)
                    matched_tracks.add(i)
                    matched_dets.add(j)

        # Crear nuevos tracks para detecciones no asignadas
        for j, det in enumerate(detecciones):
            if j not in matched_dets:
                new_track = Track(
                    track_id=self._next_id,
                    clase=det["clase"],
                    equipo=equipo_map[j],
                    last_box=(det["x"], det["y"], det["w"], det["h"]),
                    history_x=[det["x"]],
                    history_y=[det["y"]],
                    history_minute=[minute],
                )
                self._tracks[self._next_id] = new_track
                active_ids.add(self._next_id)
                self._next_id += 1

        # Eliminar tracks muy perdidos
        to_delete = []
        for tid, track in self._tracks.items():
            if tid not in active_ids:
                track.frames_lost += 1
                if track.frames_lost > self.MAX_LOST:
                    to_delete.append(tid)
        for tid in to_delete:
            del self._tracks[tid]

        return self._tracks

    def get_all_tracks(self) -> List[Track]:
        """Retorna todos los tracks, incluidos los ya eliminados (para estadísticas finales)."""
        return list(self._tracks.values())

    def reset(self):
        self._next_id = 1
        self._tracks = {}
