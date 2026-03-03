import streamlit as st
import time
from pathlib import Path

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def render():
    st.header("🎬 Subir y Analizar Vídeo")

    # ── BLOQUE 1: VÍDEO ──────────────────────────────────────────
    st.subheader("1. Vídeo del partido")
    uploaded_file = st.file_uploader(
        "Sube el vídeo del partido",
        type=["mp4", "avi", "mkv", "mov"]
    )

    if uploaded_file:
        st.success(f"✅ {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.1f} MB)")
        video_path = UPLOAD_DIR / uploaded_file.name
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        st.session_state["video_path"] = str(video_path)
        st.session_state["video_name"] = uploaded_file.name

    st.markdown("---")

    # ── BLOQUE 2: EQUIPOS Y JUGADORES ────────────────────────────
    st.subheader("2. Equipos y jugadores convocados")

    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    team_local = col_info1.text_input("Equipo local", placeholder="Ej: Atletico Sanluqueno")
    team_visit = col_info2.text_input("Equipo visitante", placeholder="Ej: Real Jaen")
    match_date = col_info3.date_input("Fecha del partido")
    competition = col_info4.selectbox("Competicion", [
    "1a Division",
    "2a Division", 
    "1a RFEF",
    "2a RFEF",
    "3a RFEF",
    "Division de Honor",
    "1a Andaluza",
    "2a Andaluza",
    "Copa del Rey",
    "Otro"
])

    st.markdown("####")

    col_local, col_visit = st.columns(2)

    # ── EQUIPO LOCAL ──────────────────────────────────────────────
    with col_local:
        st.markdown(f"### {team_local if team_local else 'Equipo Local'}")
        st.markdown("**#  &nbsp;&nbsp;&nbsp; Nombre**")
        jugadores_local = []
        for i in range(20):
            c1, c2 = st.columns([1, 4])
            dorsal = c1.number_input("D", min_value=1, max_value=99, value=i+1,
                                      key=f"local_dorsal_{i}", label_visibility="collapsed")
            nombre = c2.text_input("Nombre", placeholder=f"Jugador {i+1}",
                                    key=f"local_nombre_{i}", label_visibility="collapsed")
            jugadores_local.append({
                "dorsal": dorsal,
                "nombre": nombre,
                "equipo": team_local
            })

    # ── EQUIPO VISITANTE ──────────────────────────────────────────
    with col_visit:
        st.markdown(f"### {team_visit if team_visit else 'Equipo Visitante'}")
        st.markdown("**#  &nbsp;&nbsp;&nbsp; Nombre**")
        jugadores_visit = []
        for i in range(20):
            c1, c2 = st.columns([1, 4])
            dorsal = c1.number_input("D", min_value=1, max_value=99, value=i+1,
                                      key=f"visit_dorsal_{i}", label_visibility="collapsed")
            nombre = c2.text_input("Nombre", placeholder=f"Jugador {i+1}",
                                    key=f"visit_nombre_{i}", label_visibility="collapsed")
            jugadores_visit.append({
                "dorsal": dorsal,
                "nombre": nombre,
                "equipo": team_visit
            })

    st.markdown("---")

    # ── BOTÓN ANALIZAR ────────────────────────────────────────────
    col_btn = st.columns([2, 1, 2])[1]
    with col_btn:
        analizar = st.button("Iniciar Analisis", type="primary", use_container_width=True)

    if analizar:
        if not uploaded_file:
            st.error("Primero sube un video")
        elif not team_local or not team_visit:
            st.error("Introduce los nombres de los dos equipos")
        else:
            st.session_state["analysis_config"] = {
                "team": team_local,
                "rival": team_visit,
                "match_date": str(match_date),
                "competition": competition,
                "jugadores_local": jugadores_local,
                "jugadores_visit": jugadores_visit,
            }
            st.session_state["processing"] = True
            run_analysis(team_local, team_visit)
            st.session_state["processing"] = False


def run_analysis(team_local, team_visit):
    st.markdown("---")
    st.subheader("Progreso del analisis")

    progress = st.progress(0)
    status = st.empty()

    steps = [
        (10, "Cargando modelo YOLOv8..."),
        (20, "Detectando jugadores en el video..."),
        (35, "Aplicando tracking ByteTrack..."),
        (50, "Calculando homografia del campo..."),
        (65, "Detectando eventos tecnicos..."),
        (80, "Asociando acciones a jugadores..."),
        (90, "Generando clips de video..."),
        (100, "Analisis completado"),
    ]

    for pct, status_msg in steps:
        progress.progress(pct)
        status.info(status_msg)
        time.sleep(0.8)

    st.success(f"Analisis {team_local} vs {team_visit} completado.")

    import numpy as np
    config = st.session_state["analysis_config"]
    todos_jugadores = config["jugadores_local"] + config["jugadores_visit"]

    np.random.seed(42)
    resultados_jugadores = {}
    for j in todos_jugadores:
        if j["nombre"]:
            resultados_jugadores[j["nombre"]] = {
                "equipo": j["equipo"],
                "dorsal": j["dorsal"],
                "total_actions": int(np.random.randint(20, 60)),
                "passes": int(np.random.randint(10, 40)),
                "key_passes": int(np.random.randint(0, 5)),
                "shots": int(np.random.randint(0, 4)),
                "duels_won": int(np.random.randint(2, 10)),
                "duels_lost": int(np.random.randint(1, 6)),
                "recoveries": int(np.random.randint(2, 8)),
                "losses": int(np.random.randint(1, 6)),
                "distance_km": round(np.random.uniform(8, 12), 1),
                "top_speed": round(np.random.uniform(24, 32), 1),
            }

    st.session_state["analysis_done"] = True
    st.session_state["resultados_jugadores"] = resultados_jugadores
    st.session_state["mock_results"] = {
        "player_name": "Analisis completo",
        "player_number": 0,
        "total_actions": sum(r["total_actions"] for r in resultados_jugadores.values()),
        "passes": sum(r["passes"] for r in resultados_jugadores.values()),
        "key_passes": sum(r["key_passes"] for r in resultados_jugadores.values()),
        "shots": sum(r["shots"] for r in resultados_jugadores.values()),
        "duels_won": sum(r["duels_won"] for r in resultados_jugadores.values()),
        "duels_lost": sum(r["duels_lost"] for r in resultados_jugadores.values()),
        "recoveries": sum(r["recoveries"] for r in resultados_jugadores.values()),
        "losses": sum(r["losses"] for r in resultados_jugadores.values()),
        "distance_km": 0,
        "top_speed": 0,
    }