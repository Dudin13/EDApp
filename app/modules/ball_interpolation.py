"""
ball_interpolation.py — Relleno de huecos en el tracking del balón.
Asegura trayectorias continuas para la detección de eventos.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BallInterpolator:
    """
    Interpola posiciones del balón cuando se pierde la detección.
    """
    def __init__(self, max_gap_seconds: float = 2.0):
        self.max_gap = max_gap_seconds

    def interpolate(self, ball_history: list) -> list:
        """
        Args:
            ball_history: list de dicts con (minute, pitch_x, pitch_y)
        Returns:
            list de dicts con huecos rellenados.
        """
        if len(ball_history) < 2:
            return ball_history

        df = pd.DataFrame(ball_history)
        df = df.sort_values("minute").drop_duplicates("minute")
        
        # Crear un rango completo de minutos basado en el sample rate estimado
        # (Asumimos que el input ya tiene los minutos donde se detectó)
        # Para interpolar, necesitamos saber qué minutos faltan.
        
        # En el pipeline de EDApp, el balón se procesa en el bucle principal.
        # Si no hay detección, no hay entrada en ball_history.
        # El interpolador debería recibir todos los timestamps del análisis.
        
        return ball_history # Placeholder: se integrará más profundamente en el pipeline

    def process_with_timestamps(self, raw_history: list, all_minutes: list) -> list:
        """
        Rellena huecos usando la lista completa de minutos procesados.
        """
        if not raw_history:
            return []
            
        df_raw = pd.DataFrame(raw_history)
        df_all = pd.DataFrame({"minute": all_minutes})
        
        df = pd.merge(df_all, df_raw, on="minute", how="left")
        
        # Interpolar linealmente
        df["pitch_x"] = df["pitch_x"].interpolate(limit=int(self.max_gap * 2)) # asumiendo 0.5s rate
        df["pitch_y"] = df["pitch_y"].interpolate(limit=int(self.max_gap * 2))
        
        # Eliminar filas que sigan siendo NaN (gaps demasiado grandes)
        df = df.dropna(subset=["pitch_x", "pitch_y"])
        
        return df.to_dict("records")
