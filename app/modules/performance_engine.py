"""
performance_engine.py — Cálculo de métricas físicas reales.
Distancia (km), Velocidad Máxima (km/h) y Sprints.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

class PerformanceEngine:
    """
    Calcula estadísticas de rendimiento basadas en coordenadas del campo (metros).
    """
    def __init__(self, sprint_threshold_kmh: float = 25.2):
        """
        Args:
            sprint_threshold_kmh: Umbral de velocidad para considerar un sprint (FIFA: 25.2 km/h).
        """
        self.sprint_threshold_ms = sprint_threshold_kmh / 3.6

    def calculate_stats(self, track_history: dict) -> dict:
        """
        Procesa el historial de un track y devuelve sus métricas físicas.
        
        Args:
            track_history: dict con keys:
                - history_minute: list[float]
                - pitch_x: list[float] (metros)
                - pitch_y: list[float] (metros)
        """
        minutes = track_history.get("history_minute", [])
        px = track_history.get("pitch_x", [])
        py = track_history.get("pitch_y", [])

        # Sincronizar longitudes por seguridad
        min_len = min(len(minutes), len(px), len(py))
        if min_len < 2:
            return {
                "distance_km": 0.0,
                "top_speed_kmh": 0.0,
                "sprint_count": 0
            }
        
        minutes = minutes[:min_len]
        px = px[:min_len]
        py = py[:min_len]

        total_dist_m = 0.0
        max_speed_ms = 0.0
        sprint_count = 0
        is_sprinting = False

        for i in range(1, len(px)):
            dt_min = minutes[i] - minutes[i-1]
            dt_s = dt_min * 60.0
            
            # Evitar saltos temporales excesivos o divisiones por cero
            if dt_s <= 0 or dt_s > 10.0:
                is_sprinting = False
                continue

            dist = np.sqrt((px[i] - px[i-1])**2 + (py[i] - py[i-1])**2)
            
            # Filtro de outlier: si la velocidad es > 45 km/h (Usain Bolt), es ruido
            speed_ms = dist / dt_s
            if speed_ms > 12.5: # ~45 km/h
                continue

            total_dist_m += dist
            
            if speed_ms > max_speed_ms:
                max_speed_ms = speed_ms

            # Detección de Sprints con histéresis simple
            if speed_ms >= self.sprint_threshold_ms:
                if not is_sprinting:
                    sprint_count += 1
                    is_sprinting = True
            else:
                # Debounce: debe bajar de 23 km/h para resetear el sprint
                if speed_ms < (23.0 / 3.6):
                    is_sprinting = False

        return {
            "distance_km": round(total_dist_m / 1000.0, 3),
            "top_speed_kmh": round(max_speed_ms * 3.6, 1),
            "sprint_count": sprint_count
        }

    def process_all_tracks(self, tracks: dict) -> dict:
        """Calcula estadísticas para todos los tracks proporcionados."""
        results = {}
        for tid, track_data in tracks.items():
            results[tid] = self.calculate_stats(track_data)
        return results
