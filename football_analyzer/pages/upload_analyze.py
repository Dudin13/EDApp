import streamlit as st
import time
import os
from pathlib import Path

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except Exception:
    pass

os.environ.setdefault("ROBOFLOW_API_KEY", os.getenv("ROBOFLOW_API_KEY", ""))


def render():
    # ── PAGE HEADER ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:28px;">
        <div>
            <h2 style="margin:0;font-size:20px;font-weight:700;color:#fff;">Análisis de Vídeo</h2>
            <p style="margin:2px 0 0;font-size:13px;color:#5a6a7e;">Sube el vídeo del partido y configura el análisis</p>
        </div>
        <div style="background:rgba(0,212,170,0.1);border:1px solid rgba(0,212,170,0.25);color:#00d4aa;padding:4px 14px;border-radius:20px;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1px;">
            Nuevo análisis
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── BLOQUE 1: VÍDEO ──────────────────────────────────────────────────────
    st.markdown('<div class="ws-section-header">01 — Vídeo del partido</div>', unsafe_allow_html=True)

    col_up, col_status = st.columns([2, 1])
    with col_up:
        uploaded_file = st.file_uploader(
            "Arrastra el vídeo aquí o haz clic para seleccionarlo",
            type=["mp4", "avi", "mkv", "mov"],
            label_visibility="collapsed"
        )
    with col_status:
        if st.session_state.get("video_name"):
            st.markdown(f"""
            <div style="background:#111827;border:1px solid #00d4aa44;border-radius:10px;padding:14px 16px;height:100%;display:flex;flex-direction:column;justify-content:center;">
                <div style="font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#00d4aa;font-weight:600;margin-bottom:4px;">Vídeo cargado</div>
                <div style="font-size:13px;font-weight:600;color:#fff;word-break:break-all;">{st.session_state['video_name']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#111827;border:1px solid #1e2a3a;border-radius:10px;padding:14px 16px;height:100%;display:flex;flex-direction:column;justify-content:center;">
                <div style="font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#5a6a7e;font-weight:600;margin-bottom:4px;">Estado</div>
                <div style="font-size:13px;color:#5a6a7e;">Sin vídeo cargado</div>
            </div>
            """, unsafe_allow_html=True)

    if uploaded_file:
        video_path = UPLOAD_DIR / uploaded_file.name
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        st.session_state["video_path"] = str(video_path)
        st.session_state["video_name"] = uploaded_file.name
        st.success(f"✅ **{uploaded_file.name}** cargado correctamente ({uploaded_file.size / 1024 / 1024:.1f} MB)")

    # ── BLOQUE 2: INFO DEL PARTIDO ───────────────────────────────────────────
    st.markdown('<div class="ws-section-header">02 — Datos del partido</div>', unsafe_allow_html=True)

    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    team_local = col_info1.text_input("Equipo local", placeholder="Ej: Atlético Sanluqueño")
    team_visit = col_info2.text_input("Equipo visitante", placeholder="Ej: Real Jaén")
    match_date = col_info3.date_input("Fecha del partido")
    competition = col_info4.selectbox("Competición", [
        "1ª División", "2ª División", "1ª RFEF", "2ª RFEF", "3ª RFEF",
        "División de Honor", "1ª Andaluza", "2ª Andaluza", "Copa del Rey", "Otro"
    ])

    # ── BLOQUE 3: JUGADORES ───────────────────────────────────────────────────
    st.markdown('<div class="ws-section-header">03 — Plantillas convocadas</div>', unsafe_allow_html=True)

    POSICIONES = ["", "Portero", "Lateral D", "Lateral I", "Central", "Pivote",
                  "Mediocentro", "Interior D", "Interior I", "Mediapunta",
                  "Extremo D", "Extremo I", "Delantero Centro"]

    col_local, col_visit = st.columns(2)

    with col_local:
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1e2a3a;border-radius:10px;padding:14px 16px;margin-bottom:12px;">
            <div style="font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#5a6a7e;font-weight:600;margin-bottom:2px;">Local</div>
            <div style="font-size:16px;font-weight:700;color:#fff;">{team_local or "Equipo Local"}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("**#** &nbsp;&nbsp; **Nombre** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Posición**")
        jugadores_local = []
        for i in range(20):
            c1, c2, c3 = st.columns([1, 3, 2])
            dorsal = c1.number_input("D", min_value=1, max_value=99, value=i + 1,
                                     key=f"local_dorsal_{i}", label_visibility="collapsed")
            nombre = c2.text_input("Nombre", placeholder=f"Jugador {i + 1}",
                                   key=f"local_nombre_{i}", label_visibility="collapsed")
            posicion = c3.selectbox("Pos", POSICIONES,
                                    key=f"local_pos_{i}", label_visibility="collapsed")
            jugadores_local.append({"dorsal": dorsal, "nombre": nombre,
                                    "equipo": team_local, "posicion": posicion})

    with col_visit:
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1e2a3a;border-radius:10px;padding:14px 16px;margin-bottom:12px;">
            <div style="font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#5a6a7e;font-weight:600;margin-bottom:2px;">Visitante</div>
            <div style="font-size:16px;font-weight:700;color:#fff;">{team_visit or "Equipo Visitante"}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("**#** &nbsp;&nbsp; **Nombre** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Posición**")
        jugadores_visit = []
        for i in range(20):
            c1, c2, c3 = st.columns([1, 3, 2])
            dorsal = c1.number_input("D", min_value=1, max_value=99, value=i + 1,
                                     key=f"visit_dorsal_{i}", label_visibility="collapsed")
            nombre = c2.text_input("Nombre", placeholder=f"Jugador {i + 1}",
                                   key=f"visit_nombre_{i}", label_visibility="collapsed")
            posicion = c3.selectbox("Pos", POSICIONES,
                                    key=f"visit_pos_{i}", label_visibility="collapsed")
            jugadores_visit.append({"dorsal": dorsal, "nombre": nombre,
                                    "equipo": team_visit, "posicion": posicion})

    # ── BLOQUE 4: OPCIONES DE ANÁLISIS ───────────────────────────────────────
    st.markdown('<div class="ws-section-header">04 — Configuración del análisis</div>', unsafe_allow_html=True)

    with st.expander("⚙️ Opciones avanzadas", expanded=False):
        col_opt1, col_opt2, col_opt3 = st.columns(3)
        detection_mode = col_opt1.selectbox(
            "Motor de detección",
            ["roboflow", "yolo (local)", "auto"],
            help="'auto' usa YOLOv8 si está entrenado, sino Roboflow"
        )
        if detection_mode == "yolo (local)":
            detection_mode = "yolo"

        sample_rate = col_opt2.slider("Analizar 1 frame cada (seg)", 1, 10, 3,
                                      help="Menor = más preciso pero más lento")
        confidence = col_opt3.slider("Confianza de detección (%)", 20, 80, 40)

    # ── BOTÓN ANALIZAR ───────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _, col_btn, _ = st.columns([2, 1, 2])
    with col_btn:
        analizar = st.button("🚀 Iniciar Análisis", type="primary", use_container_width=True)

    if analizar:
        if not st.session_state.get("video_path"):
            st.error("❌ Primero sube un vídeo")
        elif not team_local or not team_visit:
            st.error("❌ Introduce los nombres de los dos equipos")
        else:
            config = {
                "team": team_local, "rival": team_visit,
                "match_date": str(match_date), "competition": competition,
                "jugadores_local": jugadores_local, "jugadores_visit": jugadores_visit,
                "sample_rate": sample_rate, "detection_mode": detection_mode,
                "confidence": confidence,
            }
            st.session_state["analysis_config"] = config
            st.session_state["processing"] = True
            run_analysis_real(config)
            st.session_state["processing"] = False


def run_analysis_real(config: dict):
    st.markdown('<div class="ws-section-header">Progreso del análisis</div>', unsafe_allow_html=True)
    progress_bar = st.progress(0)
    status_box = st.empty()

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
            status_box.markdown(f"""
            <div style="background:#111827;border:1px solid #1e2a3a;border-radius:8px;padding:10px 16px;font-size:13px;color:#e8eaed;">
                {estado}
            </div>
            """, unsafe_allow_html=True)
            if resultados:
                resultados_finales = resultados

        if resultados_finales:
            st.session_state["analysis_done"] = True
            st.session_state["mock_results"] = resultados_finales.get("mock_results", {})
            st.session_state["resultados_jugadores"] = resultados_finales.get("resultados_jugadores", {})
            st.session_state["heatmap_x"] = resultados_finales.get("heatmap_x", [])
            st.session_state["heatmap_y"] = resultados_finales.get("heatmap_y", [])
            st.session_state["detecciones_por_minuto"] = resultados_finales.get("detecciones_por_minuto", {})
            st.session_state["ball_events"] = resultados_finales.get("ball_events", [])
            st.session_state["total_detecciones"] = resultados_finales.get("total_detecciones", 0)
            st.session_state["frames_analizados"] = resultados_finales.get("frames_analizados", 0)
            if not st.session_state.get("video_path"):
                st.session_state["video_path"] = video_path

            total = resultados_finales.get("total_detecciones", 0)
            frames = resultados_finales.get("frames_analizados", 0)
            ball_evs = len(resultados_finales.get("ball_events", []))

            # Resumen con tarjetas
            st.markdown("<br>", unsafe_allow_html=True)
            st.success(f"✅ Análisis **{config.get('team')} vs {config.get('rival')}** completado")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Jugadores detectados", len(resultados_finales.get("resultados_jugadores", {})))
            c2.metric("Total detecciones", f"{total:,}")
            c3.metric("Frames analizados", frames)
            c4.metric("⚽ Acciones con balón", ball_evs)
        else:
            st.warning("⚠️ El análisis terminó sin resultados. Comprueba que el vídeo contiene el partido.")

    except ImportError as e:
        st.error(f"❌ Dependencia no instalada: {e}")
    except Exception as e:
        st.error(f"❌ Error durante el análisis: {e}")
        import traceback
        with st.expander("Ver detalle del error"):
            st.code(traceback.format_exc())