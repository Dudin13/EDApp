"""
pitch_config.py — Definición geométrica del campo de fútbol.
Basado en las 32 marcas estándar para homografía automática.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class SoccerPitchConfiguration:
    width: int = 6800   # [cm]
    length: int = 10500 # [cm]
    penalty_box_width: int = 4032
    penalty_box_length: int = 1650
    goal_box_width: int = 1832
    goal_box_length: int = 550
    centre_circle_radius: int = 915
    penalty_spot_distance: int = 1100

    @property
    def vertices(self) -> List[Tuple[float, float]]:
        # Escalar a metros para que la homografía devuelva metros directamente
        w = self.width / 100.0
        l = self.length / 100.0
        pbw = self.penalty_box_width / 100.0
        pbl = self.penalty_box_length / 100.0
        gbw = self.goal_box_width / 100.0
        gbl = self.goal_box_length / 100.0
        ccr = self.centre_circle_radius / 100.0
        psd = self.penalty_spot_distance / 100.0

        return [
            (0, 0), (0, (w - pbw)/2), (0, (w - gbw)/2), (0, (w + gbw)/2), (0, (w + pbw)/2), (0, w),
            (gbl, (w - gbw)/2), (gbl, (w + gbw)/2), (psd, w/2),
            (pbl, (w - pbw)/2), (pbl, (w - gbw)/2), (pbl, (w + gbw)/2), (pbl, (w + pbw)/2),
            (l/2, 0), (l/2, w/2 - ccr), (l/2, w/2 + ccr), (l/2, w),
            (l - pbl, (w - pbw)/2), (l - pbl, (w - gbw)/2), (l - pbl, (w + gbw)/2), (l - pbl, (w + pbw)/2),
            (l - psd, w/2), (l - gbl, (w - gbw)/2), (l - gbl, (w + gbw)/2),
            (l, 0), (l, (w - pbw)/2), (l, (w - gbw)/2), (l, (w + gbw)/2), (l, (w + pbw)/2), (l, w),
            (l/2 - ccr, w/2), (l/2 + ccr, w/2)
        ]
