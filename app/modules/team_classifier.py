"""
team_classifier.py
==================
Capa 2 del pipeline: clasifica jugadores en equipo A o equipo B
basandose en el color de su camiseta.

Dos modos:
  - AUTO: KMeans agrupa los colores sin saber nada de antemano
  - MANUAL: se le pasan los colores RGB de cada equipo

Uso:
    from modules.team_classifier import TeamClassifier, Team

    clf = TeamClassifier()
    clf.fit(frame, list_of_bboxes)      # aprende colores en el frame elegido
    team = clf.predict(frame, bbox)     # Team.A / Team.B / Team.REFEREE / Team.UNKNOWN
    print(clf.colors.name_a)            # "Equipo A"
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Team(str, Enum):
    A       = "team_a"
    B       = "team_b"
    REFEREE = "referee"
    UNKNOWN = "unknown"


@dataclass
class TeamColors:
    """Colores representativos de cada equipo en espacio HSV."""
    team_a:  np.ndarray = field(default_factory=lambda: np.zeros(3))
    team_b:  np.ndarray = field(default_factory=lambda: np.zeros(3))
    fitted:  bool = False
    name_a:  str = "Equipo A"
    name_b:  str = "Equipo B"


class TeamClassifier:
    """
    Clasifica jugadores en equipo A o equipo B por color de camiseta.

    Flujo:
        1. clf.fit(frame, bboxes)     -> aprende los 2 colores dominantes (KMeans)
           o bien:
           clf.set_colors(rgb_a, rgb_b) -> modo manual con colores conocidos
        2. clf.predict(frame, bbox)   -> Team enum
    """

    def __init__(self, torso_ratio: float = 0.35):
        """
        Args:
            torso_ratio: Fraccion vertical del bbox usada como zona de camiseta.
                         0.35 = coge el tercio central (evita cara y piernas)
        """
        self.torso_ratio = torso_ratio
        self.colors      = TeamColors()

    # ── Extraccion de color de camiseta ────────────────────────────────────

    def _extract_torso_crop(self, frame: np.ndarray, bbox: tuple) -> Optional[np.ndarray]:
        """
        Recorta la zona del torso del jugador.
        Toma el 35% central del alto del bbox y el 60% central del ancho,
        para evitar cara, piernas, fondo y brazos.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h = y2 - y1
        w = x2 - x1

        if h < 20 or w < 8:
            return None

        # Zona vertical: desde 25% hasta 60% del alto del bbox
        y_start = y1 + int(h * 0.25)
        y_end   = y1 + int(h * 0.60)

        # Zona horizontal: 20% margen a cada lado
        x_start = x1 + int(w * 0.20)
        x_end   = x2 - int(w * 0.20)

        crop = frame[y_start:y_end, x_start:x_end]
        if crop.size == 0:
            return None
        return crop

    def _dominant_color_hsv(self, crop: np.ndarray, k: int = 3) -> np.ndarray:
        """
        Extrae el color dominante de un recorte usando KMeans en HSV.
        Ignora pixeles muy oscuros (sombras) y muy claros (cielo/fondo).

        Returns:
            Array HSV [H, S, V] del color dominante
        """
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3).astype(np.float32)

        # Filtrar sombras (V < 40) y sobreexposicion (V > 230 y S < 30)
        mask = (pixels[:, 2] > 40) & ~((pixels[:, 2] > 230) & (pixels[:, 1] < 30))
        pixels = pixels[mask]

        if len(pixels) < 10:
            return hsv.reshape(-1, 3).mean(axis=0)

        # KMeans para encontrar color dominante
        k = min(k, len(pixels))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS
        )
        # Elegir el cluster mas grande
        counts   = np.bincount(labels.flatten())
        dominant = centers[np.argmax(counts)]
        return dominant

    # ── Fit (modo automatico) ──────────────────────────────────────────────

    def fit(self, frame: np.ndarray, player_bboxes: list[tuple],
            name_a: str = "Equipo A", name_b: str = "Equipo B") -> bool:
        """
        Aprende los colores de los dos equipos usando KMeans sobre
        los recortes de camiseta de los jugadores dados.

        Args:
            frame:         Frame BGR
            player_bboxes: Lista de (x1,y1,x2,y2) — SOLO jugadores, no porteros/arbitros
            name_a:        Nombre del equipo A (para mostrar en UI)
            name_b:        Nombre del equipo B

        Returns:
            True si el fit fue exitoso, False si hay pocos jugadores
        """
        if len(player_bboxes) < 4:
            return False

        colors_hsv = []
        for bbox in player_bboxes:
            crop = self._extract_torso_crop(frame, bbox)
            if crop is None:
                continue
            color = self._dominant_color_hsv(crop)
            colors_hsv.append(color)

        if len(colors_hsv) < 4:
            return False

        # KMeans con k=2 para separar los dos equipos
        data     = np.array(colors_hsv, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
        _, labels, centers = cv2.kmeans(
            data, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        self.colors.team_a = centers[0]
        self.colors.team_b = centers[1]
        self.colors.fitted = True
        self.colors.name_a = name_a
        self.colors.name_b = name_b
        return True

    def set_colors(self, rgb_a: tuple, rgb_b: tuple,
                   name_a: str = "Equipo A", name_b: str = "Equipo B"):
        """
        Modo manual: establece los colores de cada equipo directamente.

        Args:
            rgb_a: Color del equipo A en RGB, e.g. (0, 80, 160) para azul marino
            rgb_b: Color del equipo B en RGB, e.g. (255, 255, 255) para blanco
        """
        # Convertir RGB -> HSV para comparacion consistente
        def rgb_to_hsv(rgb):
            px = np.uint8([[list(rgb)]])
            px_bgr = cv2.cvtColor(px, cv2.COLOR_RGB2BGR)
            return cv2.cvtColor(px_bgr, cv2.COLOR_BGR2HSV)[0][0].astype(np.float32)

        self.colors.team_a = rgb_to_hsv(rgb_a)
        self.colors.team_b = rgb_to_hsv(rgb_b)
        self.colors.fitted = True
        self.colors.name_a = name_a
        self.colors.name_b = name_b

    # ── Predict ────────────────────────────────────────────────────────────

    def predict(self, frame: np.ndarray, bbox: tuple,
                is_referee: bool = False) -> Team:
        """
        Clasifica un jugador en equipo A, B, arbitro o desconocido.

        Args:
            frame:       Frame BGR completo
            bbox:        (x1, y1, x2, y2) del jugador
            is_referee:  Si YOLO ya lo detecto como arbitro, devuelve REFEREE directamente

        Returns:
            Team enum
        """
        if is_referee:
            return Team.REFEREE

        if not self.colors.fitted:
            return Team.UNKNOWN

        crop = self._extract_torso_crop(frame, bbox)
        if crop is None:
            return Team.UNKNOWN

        color = self._dominant_color_hsv(crop)

        # Distancia euclidea en espacio HSV a cada equipo
        # El canal H es circular (0-180 en OpenCV), hay que tratarlo
        def hsv_distance(c1: np.ndarray, c2: np.ndarray) -> float:
            dh = min(abs(float(c1[0]) - float(c2[0])),
                     180 - abs(float(c1[0]) - float(c2[0])))
            ds = float(c1[1]) - float(c2[1])
            dv = float(c1[2]) - float(c2[2])
            # H tiene mas peso porque es el canal de color principal
            return np.sqrt((dh * 2) ** 2 + ds ** 2 + (dv * 0.5) ** 2)

        dist_a = hsv_distance(color, self.colors.team_a)
        dist_b = hsv_distance(color, self.colors.team_b)

        return Team.A if dist_a <= dist_b else Team.B

    def predict_batch(self, frame: np.ndarray,
                      detections: list[dict]) -> list[dict]:
        """
        Clasifica una lista de detecciones enriqueciendo cada una con 'team'.

        Args:
            frame:      Frame BGR
            detections: Lista de dicts con keys: bbox, cls_name, [es_portero]

        Returns:
            Misma lista con 'team' añadido a cada dict
        """
        result = []
        for det in detections:
            bbox     = det.get("bbox", (0, 0, 0, 0))
            cls_name = det.get("cls_name", "")
            is_ref   = cls_name == "referee"
            team     = self.predict(frame, bbox, is_referee=is_ref)
            result.append({**det, "team": team.value})
        return result

    def adapt(self, frame: np.ndarray, detections_with_team: list[dict]) -> bool:
        """
        Re-calibración incremental: actualiza los centroides de color con los
        tracks estables del frame actual. Útil para adaptarse a cambios de
        iluminación durante el partido.

        Args:
            frame: Frame BGR actual
            detections_with_team: Lista de dicts con keys 'bbox' y 'equipo' (0 o 1)
                                  Solo se usan detecciones con equipo conocido (0 o 1).

        Returns:
            True si se actualizaron los centroides, False si no había suficientes datos.
        """
        if not self.colors.fitted:
            return False

        colors_a, colors_b = [], []
        for det in detections_with_team:
            eq = det.get("equipo", -1)
            if eq not in (0, 1):
                continue
            bbox = det.get("bbox")
            if bbox is None:
                x, y, w, h = det.get("x", 0), det.get("y", 0), det.get("w", 0), det.get("h", 0)
                bbox = (x - w/2, y - h/2, x + w/2, y + h/2)
            if bbox[3] - bbox[1] < 20:  # bbox demasiado pequeño
                continue
            crop = self._extract_torso_crop(frame, bbox)
            if crop is None:
                continue
            color = self._dominant_color_hsv(crop)
            (colors_a if eq == 0 else colors_b).append(color)

        if len(colors_a) < 3 or len(colors_b) < 3:
            return False

        # Media ponderada: 70% centroide previo + 30% media del frame actual
        # para que la adaptación sea gradual y no salte por un frame malo
        alpha = 0.30
        new_a = np.mean(colors_a, axis=0)
        new_b = np.mean(colors_b, axis=0)
        self.colors.team_a = (1 - alpha) * self.colors.team_a + alpha * new_a
        self.colors.team_b = (1 - alpha) * self.colors.team_b + alpha * new_b
        return True

    # ── Utilidades ─────────────────────────────────────────────────────────

    def get_team_color_bgr(self, team: Team) -> tuple:
        """Devuelve el color BGR representativo del equipo para dibujar en pantalla."""
        if not self.colors.fitted:
            return (128, 128, 128)

        ref_hsv = self.colors.team_a if team == Team.A else self.colors.team_b
        px      = np.uint8([[ref_hsv.astype(np.uint8)]])
        bgr     = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0][0]
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    def get_summary(self) -> dict:
        """Resumen del estado del clasificador para debug/UI."""
        if not self.colors.fitted:
            return {"fitted": False}

        def hsv_to_hex(hsv):
            px  = np.uint8([[hsv.astype(np.uint8)]])
            bgr = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0][0]
            return "#{:02x}{:02x}{:02x}".format(int(bgr[2]), int(bgr[1]), int(bgr[0]))

        return {
            "fitted":   True,
            "name_a":   self.colors.name_a,
            "name_b":   self.colors.name_b,
            "color_a":  hsv_to_hex(self.colors.team_a),
            "color_b":  hsv_to_hex(self.colors.team_b),
        }

    def draw_debug(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """
        Dibuja las detecciones sobre el frame con colores por equipo.
        Util para verificar que la clasificacion es correcta.
        """
        out = frame.copy()
        team_colors = {
            Team.A.value:       (255, 100,  50),   # azul
            Team.B.value:       ( 50, 200, 255),   # naranja
            Team.REFEREE.value: ( 50, 255,  50),   # verde
            Team.UNKNOWN.value: (180, 180, 180),   # gris
        }
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.get("bbox", (0,0,0,0))]
            team  = det.get("team", Team.UNKNOWN.value)
            label = det.get("label", team)
            color = team_colors.get(team, (180, 180, 180))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, label, (x1, max(y1-6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        return out
