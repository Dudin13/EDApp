"""
pages/player_identification.py
================================
Pagina de Streamlit para identificar jugadores antes de analizar un partido.

Flujo:
  1. Usuario sube el video (o usa uno ya cargado)
  2. Scrubba el video para elegir un frame limpio con todos los jugadores visibles
  3. El sistema detecta jugadores y los agrupa automaticamente por equipo (KMeans)
  4. Usuario puede corregir la asignacion de equipo si hace falta
  5. Para cada jugador: escribe nombre, dorsal y posicion
  6. Opcionalmente ajusta los colores de equipo manualmente
  7. Guarda la sesion y pasa al analisis

Los datos se guardan en st.session_state['match_players'] con estructura:
    {
        "team_a": {"name": "Cadiz CF", "color_hex": "#ffff00",
                   "players": [{"id": 1, "name": "Juan", "dorsal": 7, "position": "DC"}, ...]},
        "team_b": {"name": "Rival",    "color_hex": "#ffffff",
                   "players": [...]},
        "classifier": <TeamClassifier instance>
    }
"""

import cv2
import numpy as np
import streamlit as st
from pathlib import Path
import sys
import base64

# Asegurar que el modulo de la app esta en el path
APP_DIR = Path(__file__).parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from modules.team_classifier import TeamClassifier, Team

POSITIONS = ["Portero", "Lateral D", "Lateral I", "Central", "Pivote",
             "Mediocentro", "Mediapunta", "Extremo D", "Extremo I", "Delantero"]


# ── Helpers ────────────────────────────────────────────────────────────────

def frame_to_base64(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()

def get_frame_at(video_path: str, second: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(second * fps))
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def get_video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames / fps

def detect_players_in_frame(frame: np.ndarray) -> list[dict]:
    """Detecta jugadores en un frame usando el modelo YOLO cargado."""
    try:
        from modules.detector import DetectorEngine
        detector  = DetectorEngine()
        raw_dets  = detector.detect(frame)
        players   = []
        for det in raw_dets:
            cls_name = det.get("cls_name", "")
            if cls_name in ("player", "goalkeeper"):
                players.append(det)
        return players
    except Exception as e:
        st.warning(f"Detector no disponible: {e}. Usando detecciones de prueba.")
        # Fallback: bboxes de prueba para desarrollo
        h, w = frame.shape[:2]
        return [
            {"bbox": (w*0.1, h*0.3, w*0.15, h*0.8), "cls_name": "player",     "conf": 0.9},
            {"bbox": (w*0.3, h*0.2, w*0.35, h*0.7), "cls_name": "player",     "conf": 0.88},
            {"bbox": (w*0.5, h*0.3, w*0.55, h*0.8), "cls_name": "goalkeeper", "conf": 0.92},
            {"bbox": (w*0.7, h*0.2, w*0.75, h*0.7), "cls_name": "player",     "conf": 0.85},
        ]

def crop_player(frame: np.ndarray, bbox: tuple, size: int = 80) -> np.ndarray:
    """Recorta un jugador del frame y lo redimensiona."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.resize(crop, (size, int(size * (y2-y1) / max(x2-x1, 1))))

def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


# ── Pagina principal ───────────────────────────────────────────────────────

def render():
    st.title("Identificacion de jugadores")
    st.caption("Selecciona un frame, confirma los equipos e introduce los datos de cada jugador.")

    # -- Paso 0: video disponible? -----------------------------------------
    video_path = st.session_state.get("video_path")
    if not video_path or not Path(video_path).exists():
        st.warning("No hay ningun video cargado. Ve primero a 'Nuevo partido' y sube un video.")
        return

    duration = get_video_duration(video_path)

    # -- Paso 1: elegir frame ----------------------------------------------
    st.subheader("1 — Elige el momento del video")
    st.caption("Busca un momento donde todos los jugadores sean claramente visibles.")

    col_slider, col_preview = st.columns([2, 1])
    with col_slider:
        second = st.slider("Segundo del video", 0.0, duration, min(30.0, duration * 0.1),
                           step=0.5, format="%.1fs")
    with col_preview:
        st.caption(f"Posicion: {second:.1f}s / {duration:.0f}s")

    frame = get_frame_at(video_path, second)
    if frame is None:
        st.error("No se pudo leer el frame. Prueba otro momento.")
        return

    # Mostrar frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, caption=f"Frame en {second:.1f}s", use_container_width=True)

    if st.button("Detectar jugadores en este frame", type="primary"):
        with st.spinner("Detectando..."):
            detections = detect_players_in_frame(frame)
            st.session_state["id_detections"] = detections
            st.session_state["id_frame"]      = frame
            st.session_state["id_second"]     = second

            # Clasificar equipos automaticamente con KMeans
            clf = TeamClassifier()
            player_bboxes = [d["bbox"] for d in detections if d["cls_name"] == "player"]
            fitted = clf.fit(frame, player_bboxes)
            st.session_state["id_classifier"] = clf
            st.session_state["id_fitted"]     = fitted

            # Inicializar datos de jugadores
            players_data = []
            for i, det in enumerate(detections):
                team = clf.predict(frame, det["bbox"],
                                   is_referee=det["cls_name"] == "referee")
                players_data.append({
                    "det_id":   i,
                    "bbox":     det["bbox"],
                    "cls_name": det["cls_name"],
                    "team":     team.value,
                    "name":     "",
                    "dorsal":   "",
                    "position": "Mediocentro",
                })
            st.session_state["id_players_data"] = players_data
        st.rerun()

    # -- Paso 2: configurar colores de equipo ------------------------------
    if "id_classifier" not in st.session_state:
        return

    clf     = st.session_state["id_classifier"]
    summary = clf.get_summary()

    st.subheader("2 — Confirma los colores de equipo")
    st.caption("El sistema ha intentado separar los equipos automaticamente. Ajusta si hace falta.")

    col_a, col_b = st.columns(2)
    with col_a:
        name_a = st.text_input("Nombre equipo A", value="Equipo A", key="name_a")
        color_a = st.color_picker("Color camiseta equipo A",
                                   value=summary.get("color_a", "#3b82f6"), key="color_a")
    with col_b:
        name_b = st.text_input("Nombre equipo B", value="Equipo B", key="name_b")
        color_b = st.color_picker("Color camiseta equipo B",
                                   value=summary.get("color_b", "#ffffff"), key="color_b")

    if st.button("Actualizar colores"):
        clf.set_colors(hex_to_rgb(color_a), hex_to_rgb(color_b), name_a, name_b)
        # Re-clasificar todos los jugadores
        id_frame   = st.session_state["id_frame"]
        players_data = st.session_state["id_players_data"]
        for p in players_data:
            team = clf.predict(id_frame, p["bbox"],
                               is_referee=p["cls_name"] == "referee")
            p["team"] = team.value
        st.session_state["id_players_data"] = players_data
        st.rerun()

    # -- Paso 3: identificar cada jugador ----------------------------------
    st.subheader("3 — Identifica cada jugador")
    st.caption("Rellena los datos de cada jugador. Puedes corregir el equipo si el sistema se ha equivocado.")

    id_frame     = st.session_state.get("id_frame", frame)
    players_data = st.session_state.get("id_players_data", [])

    # Separar por equipo
    team_a_players = [p for p in players_data if p["team"] == Team.A.value]
    team_b_players = [p for p in players_data if p["team"] == Team.B.value]
    other_players  = [p for p in players_data if p["team"] not in (Team.A.value, Team.B.value)]

    def render_player_row(p: dict, idx: int):
        """Renderiza la fila de un jugador con su foto y campos de datos."""
        crop   = crop_player(id_frame, p["bbox"])
        crop_r = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        c_img, c_name, c_dorsal, c_pos, c_team = st.columns([1, 3, 1, 2, 2])
        with c_img:
            st.image(crop_r, width=60)
        with c_name:
            p["name"] = st.text_input("Nombre", value=p["name"],
                                       key=f"name_{idx}", label_visibility="collapsed",
                                       placeholder="Nombre jugador")
        with c_dorsal:
            p["dorsal"] = st.text_input("Dorsal", value=str(p["dorsal"]),
                                         key=f"dorsal_{idx}", label_visibility="collapsed",
                                         placeholder="#")
        with c_pos:
            pos_idx = POSITIONS.index(p["position"]) if p["position"] in POSITIONS else 0
            p["position"] = st.selectbox("Posicion", POSITIONS, index=pos_idx,
                                          key=f"pos_{idx}", label_visibility="collapsed")
        with c_team:
            teams_opts = [Team.A.value, Team.B.value, Team.REFEREE.value]
            team_idx   = teams_opts.index(p["team"]) if p["team"] in teams_opts else 0
            p["team"]  = st.selectbox("Equipo", teams_opts, index=team_idx,
                                       key=f"team_{idx}", label_visibility="collapsed")

    tab_a, tab_b, tab_other = st.tabs([
        f"{name_a} ({len(team_a_players)})",
        f"{name_b} ({len(team_b_players)})",
        f"Otros ({len(other_players)})"
    ])

    with tab_a:
        if team_a_players:
            st.caption("Nombre | Dorsal | Posicion | Equipo")
            for p in team_a_players:
                render_player_row(p, p["det_id"])
        else:
            st.info("No hay jugadores asignados a este equipo todavia.")

    with tab_b:
        if team_b_players:
            st.caption("Nombre | Dorsal | Posicion | Equipo")
            for p in team_b_players:
                render_player_row(p, p["det_id"])
        else:
            st.info("No hay jugadores asignados a este equipo todavia.")

    with tab_other:
        if other_players:
            for p in other_players:
                render_player_row(p, p["det_id"])

    # -- Paso 4: guardar y continuar ---------------------------------------
    st.divider()
    col_save, col_info = st.columns([1, 2])
    with col_save:
        if st.button("Guardar e ir al analisis", type="primary", use_container_width=True):
            # Guardar todo en session_state para el pipeline
            st.session_state["match_players"] = {
                "team_a": {
                    "name":      st.session_state.get("name_a", "Equipo A"),
                    "color_hex": st.session_state.get("color_a", "#3b82f6"),
                    "players":   [p for p in players_data if p["team"] == Team.A.value],
                },
                "team_b": {
                    "name":      st.session_state.get("name_b", "Equipo B"),
                    "color_hex": st.session_state.get("color_b", "#ffffff"),
                    "players":   [p for p in players_data if p["team"] == Team.B.value],
                },
                "classifier": clf,
            }
            st.success("Datos guardados. Continua en 'Analizar partido'.")
            st.balloons()
    with col_info:
        total_identified = sum(1 for p in players_data if p["name"])
        st.metric("Jugadores identificados", f"{total_identified} / {len(players_data)}")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    render()
else:
    render()
