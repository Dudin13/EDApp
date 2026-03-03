import streamlit as st
import time
import os
from pathlib import Path

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Cargar variables de entorno
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except Exception:
    pass

os.environ.setdefault("ROBOFLOW_API_KEY", os.getenv("ROBOFLOW_API_KEY", ""))


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
        "1a Division", "2a Division", "1a RFEF", "2a RFEF", "3a RFEF",
        "Division de Honor", "1a Andaluza", "2a Andaluza", "Copa del Rey", "Otro"
    ])

    st.markdown("####")
    col_local, col_visit = st.columns(2)

    POSICIONES = ["", "Portero", "Lateral D", "Lateral I", "Central", "Pivote",
                  "Mediocentro", "Interior D", "Interior I", "Mediapunta",
                  "Extremo D", "Extremo I", "Delantero Centro"]

    # ── EQUIPO LOCAL ──────────────────────────────────────────────
    with col_local:
        st.markdown(f"### {team_local if team_local else 'Equipo Local'}")
        st.markdown("**#  &nbsp;&nbsp;&nbsp; Nombre &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Posición**")
        jugadores_local = []
        for i in range(20):
            c1, c2, c3 = st.columns([1, 3, 2])
            dorsal = c1.number_input("D", min_value=1, max_value=99, value=i + 1,
                                     key=f"local_dorsal_{i}", label_visibility="collapsed")
            nombre = c2.text_input("Nombre", placeholder=f"Jugador {i + 1}",
                                   key=f"local_nombre_{i}", label_visibility="collapsed")
            posicion = c3.selectbox("Pos", POSICIONES,
                                    key=f"local_pos_{i}", label_visibility="collapsed")
            jugadores_local.append({
                "dorsal": dorsal, "nombre": nombre,
                "equipo": team_local, "posicion": posicion
            })

    # ── EQUIPO VISITANTE ──────────────────────────────────────────
    with col_visit:
        st.markdown(f"### {team_visit if team_visit else 'Equipo Visitante'}")
        st.markdown("**#  &nbsp;&nbsp;&nbsp; Nombre &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Posición**")
        jugadores_visit = []
        for i in range(20):
            c1, c2, c3 = st.columns([1, 3, 2])
            dorsal = c1.number_input("D", min_value=1, max_value=99, value=i + 1,
                                     key=f"visit_dorsal_{i}", label_visibility="collapsed")
            nombre = c2.text_input("Nombre", placeholder=f"Jugador {i + 1}",
                                   key=f"visit_nombre_{i}", label_visibility="collapsed")
            posicion = c3.selectbox("Pos", POSICIONES,
                                    key=f"visit_pos_{i}", label_visibility="collapsed")
            jugadores_visit.append({
                "dorsal": dorsal, "nombre": nombre,
                "equipo": team_visit, "posicion": posicion
            })

    st.markdown("---")

    # ── OPCIONES DE ANÁLISIS ──────────────────────────────────────
    with st.expander("⚙️ Opciones de análisis"):
        col_opt1, col_opt2, col_opt3 = st.columns(3)
        detection_mode = col_opt1.selectbox(
            "Motor de detección",
            ["roboflow", "yolo (local)", "auto"],
            help="'auto' usa YOLOv8 si está entrenado, sino Roboflow"
        )
        if detection_mode == "yolo (local)":
            detection_mode = "yolo"
        elif detection_mode == "auto":
            detection_mode = "auto"

        sample_rate = col_opt2.slider(
            "Analizar 1 frame cada (segundos)", 1, 10, 3,
            help="Menor = más preciso pero más lento y más llamadas a la API"
        )
        confidence = col_opt3.slider("Confianza de detección", 20, 80, 40)

    # ── BOTÓN ANALIZAR ────────────────────────────────────────────
    col_btn = st.columns([2, 1, 2])[1]
    with col_btn:
        analizar = st.button("🚀 Iniciar Análisis", type="primary", use_container_width=True)

    if analizar:
        if not st.session_state.get("video_path"):
            st.error("Primero sube un vídeo")
        elif not team_local or not team_visit:
            st.error("Introduce los nombres de los dos equipos")
        else:
            config = {
                "team": team_local,
                "rival": team_visit,
                "match_date": str(match_date),
                "competition": competition,
                "jugadores_local": jugadores_local,
                "jugadores_visit": jugadores_visit,
                "sample_rate": sample_rate,
                "detection_mode": detection_mode,
                "confidence": confidence,
            }
            st.session_state["analysis_config"] = config
            st.session_state["processing"] = True
            run_analysis_real(config)
            st.session_state["processing"] = False


def run_analysis_real(config: dict):
    """Pipeline de análisis real usando VideoProcessor."""
    st.markdown("---")
    st.subheader("Progreso del análisis")

    progress_bar = st.progress(0)
    status_box = st.empty()
    info_box = st.empty()

    video_path = st.session_state.get("video_path", "")
    if not video_path or not os.path.exists(video_path):
        st.error("❌ No se encontró el fichero de vídeo. Vuelve a subirlo.")
        return

    try:
        from modules.video_processor import VideoProcessor
        processor = VideoProcessor(video_path, config)

        resultados_finales = {}
        for progreso, estado, resultados in processor.process():
            progress_bar.progress(int(progreso))
            status_box.info(f"**{estado}**")
            if resultados:
                resultados_finales = resultados

        # Guardar en session_state
        if resultados_finales:
            st.session_state["analysis_done"] = True
            st.session_state["mock_results"] = resultados_finales.get("mock_results", {})
            st.session_state["resultados_jugadores"] = resultados_finales.get("resultados_jugadores", {})
            st.session_state["heatmap_x"] = resultados_finales.get("heatmap_x", [])
            st.session_state["heatmap_y"] = resultados_finales.get("heatmap_y", [])
            st.session_state["detecciones_por_minuto"] = resultados_finales.get("detecciones_por_minuto", {})

            team_local = config.get("team", "")
            team_visit = config.get("rival", "")
            total = resultados_finales.get("total_detecciones", 0)
            frames = resultados_finales.get("frames_analizados", 0)

            st.success(f"✅ Análisis **{team_local} vs {team_visit}** completado — "
                       f"{total:,} detecciones en {frames} frames")

            # Resumen rápido
            resultados_jug = resultados_finales.get("resultados_jugadores", {})
            if resultados_jug:
                col1, col2, col3 = st.columns(3)
                col1.metric("Jugadores detectados", len(resultados_jug))
                col2.metric("Total detecciones", f"{total:,}")
                col3.metric("Frames analizados", frames)
        else:
            st.warning("⚠️ El análisis terminó pero no se generaron resultados. "
                       "Comprueba que el vídeo contiene el partido.")

    except ImportError as e:
        st.error(f"❌ Dependencia no instalada: {e}")
        st.info("Instala las dependencias con: `pip install roboflow ultralytics opencv-python`")
    except Exception as e:
        st.error(f"❌ Error durante el análisis: {e}")
        import traceback
        with st.expander("Ver detalle del error"):
            st.code(traceback.format_exc())