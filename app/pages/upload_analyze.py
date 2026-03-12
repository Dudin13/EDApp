import streamlit as st
st.set_page_config(page_title="Football Analyzer", layout="wide", initial_sidebar_state="collapsed")
import time
import os
import json
from pathlib import Path
from PIL import Image
from streamlit_drawable_canvas import st_canvas

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except Exception:
    pass

os.environ.setdefault("ROBOFLOW_API_KEY", os.getenv("ROBOFLOW_API_KEY", ""))

@st.cache_data(show_spinner=False)
def _get_diagnostic_frame(video_path, target_second):
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, target_second * 1000)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame


def _render_model_diagnostics(video_path: str, detection_mode: str, confidence: int):
    """Muestra que detecta el modelo en un frame de muestra del video cargado."""
    import cv2
    import numpy as np
    from pathlib import Path
    from core.config.settings import settings
    from ai.detector.detector import DetectorEngine
    
    detector = DetectorEngine()
    
    yolo_ok = Path(settings.PLAYER_MODEL_PATH).exists()

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        if yolo_ok:
            st.markdown(f"""
            <div style="background:rgba(0, 212, 170, 0.1);border:1px solid rgba(0, 212, 170, 0.3);border-radius:12px;padding:16px;">
                <div style="font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#00d4aa;font-weight:700;">
                    MODELO YOLO V8 ACTIVO
                </div>
                <div style="font-size:13px;color:#fff;margin-top:4px;">{Path(settings.PLAYER_MODEL_PATH).name}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:rgba(255, 77, 109, 0.1);border:1px solid rgba(255, 77, 109, 0.3);border-radius:12px;padding:16px;">
                <div style="font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#ff4d6d;font-weight:700;">
                    FALLBACK: ROBOFLOW MODE
                </div>
                <div style="font-size:13px;color:#fff;margin-top:4px;">No se detectó el modelo local .pt</div>
            </div>
            """, unsafe_allow_html=True)

    with col_m2:
        st.markdown(f"""
        <div style="background:rgba(17, 24, 39, 0.4);border:1px solid rgba(30, 42, 58, 0.6);border-radius:12px;padding:16px;">
            <div style="font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#8899aa;font-weight:700;">
                MOTOR SELECCIONADO
            </div>
            <div style="font-size:13px;color:#fff;margin-top:4px;font-weight:700;">{detection_mode.upper()}</div>
        </div>
        """, unsafe_allow_html=True)

    if not video_path or not Path(video_path).exists():
        st.info("Carga un vídeo para ver el diagnóstico.")
        return

    # Optimizar lectura de frame con cache y búsqueda por msec
    target_second = 10.0 # Estático para diagnóstico rápido
    frame = _get_diagnostic_frame(video_path, target_second)

    if frame is None:
        st.error("No se pudo leer el frame para el diagnóstico.")
        return

    with st.spinner("⏳ Ejecutando diagnóstico..."):
        dets = detector.detect_frame(frame, confidence=confidence / 100)

    CLASS_COLORS = {"ball": (0, 255, 255), "player": (0, 212, 170), "goalkeeper": (255, 200, 0), "referee": (0, 0, 255)}
    vis_frame = frame.copy()
    for det in dets:
        x, y, w, h = det["x"], det["y"], det["w"], det["h"]
        clase = det.get("clase", "player")
        conf = det.get("confianza", 0)
        color = CLASS_COLORS.get(clase, (200, 200, 200))
        cv2.rectangle(vis_frame, (x - w//2, y - h//2), (x + w//2, y + h//2), color, 2)

    vis_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
    st.image(vis_rgb, caption=f"Frame ~{target_second:.0f}s | {len(dets)} detecciones", use_container_width=True)


def render():
    if "analysis_step" not in st.session_state:
        st.session_state["analysis_step"] = 1

    # --- Loading Persistence ---
    results_file = settings.OUTPUT_DIR / "results.json"
    if not st.session_state.get("analysis_done") and results_file.exists():
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
            st.session_state["resultados_jugadores"] = saved.get("resultados_jugadores", {})
            st.session_state["heatmap_x"] = saved.get("heatmap_x", [])
            st.session_state["heatmap_y"] = saved.get("heatmap_y", [])
            st.session_state["ball_events"] = saved.get("ball_events", [])
            st.session_state["total_detecciones"] = saved.get("total_detecciones", 0)
            st.session_state["frames_analizados"] = saved.get("frames_analizados", 0)
            st.session_state["analysis_done"] = True
            st.toast("⏳ Análisis previo cargado automáticamente", icon="💾")
        except Exception:
            pass

    # --- STEPPER UI ---
    steps = ["Entrada", "Calibración", "Marcado", "Motor", "Resumen"]
    cols = st.columns(len(steps))
    for i, s in enumerate(steps):
        step_num = i + 1
        is_active = st.session_state["analysis_step"] == step_num
        is_done = st.session_state["analysis_step"] > step_num
        
        color = "#00d4aa" if is_active else ("#fff" if is_done else "#5a6a7e")
        bg = "rgba(0, 212, 170, 0.15)" if is_active else "transparent"
        border = "1px solid #00d4aa" if is_active else "none"
        
        cols[i].markdown(f"""
        <div style="text-align:center; padding:10px; border-radius:10px; background:{bg}; border:{border}; transition: all 0.3s; cursor:pointer;">
            <div style="font-size:10px; color:{color}; font-weight:700; margin-bottom:4px;">PASO 0{step_num}</div>
            <div style="font-size:12px; color:{color}; font-weight:600;">{s.split()[-1]}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # --- STEP 1: VIDEO Y RIVAL ---
    if st.session_state["analysis_step"] == 1:
        st.markdown('<div class="ws-section-header">Configuración del Encuentro</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            uploaded_file = st.file_uploader("Sube vídeo", type=["mp4","avi","mkv","mov"], label_visibility="collapsed")
            VIDEO_DIR = Path(__file__).parent.parent / "videos"
            VIDEO_DIR.mkdir(exist_ok=True)
            # Combine videos from both 'videos' and 'uploads' folders
            local_v = [f.name for f in VIDEO_DIR.glob("*") if f.suffix.lower() in [".mp4",".avi",".mkv",".mov"]]
            upload_v = [f.name for f in UPLOAD_DIR.glob("*") if f.suffix.lower() in [".mp4",".avi",".mkv",".mov"]]
            all_videos = sorted(list(set(local_v + upload_v)))
            selected_local = st.selectbox("O selecciona uno existente:", [""] + all_videos)
            
            if uploaded_file:
                video_path = UPLOAD_DIR / uploaded_file.name
                # Buffering for large files (10GB+)
                with open(video_path, "wb") as f:
                    while True:
                        chunk = uploaded_file.read(8192)
                        if not chunk: break
                        f.write(chunk)
                st.session_state["video_path"] = str(video_path)
                st.session_state["video_name"] = uploaded_file.name
            elif selected_local:
                # Check where the selected video is located
                path_video = VIDEO_DIR / selected_local
                if not path_video.exists():
                    path_video = UPLOAD_DIR / selected_local
                
                st.session_state["video_path"] = str(path_video)
                st.session_state["video_name"] = selected_local

        with col2:
            st.text_input("Equipo Local", value="Cadiz CF", key="step1_local")
            st.text_input("Equipo Visitante", placeholder="Ej: Real Madrid", key="step1_visit")
            st.date_input("Fecha", key="step1_date")
            st.text_input("Jornada/Competición", placeholder="Ej: Jornada 14", key="step1_jornada")
            st.text_input("Resultado", placeholder="Ej: 2 - 1", key="step1_result")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Siguiente: Calibrar IA →", use_container_width=True):
            if not st.session_state.get("video_path"):
                st.error("No has seleccionado ningún vídeo.")
            else:
                st.session_state["analysis_step"] = 2
                st.rerun()

    # --- STEP 2: CALIBRACION ---
    elif st.session_state["analysis_step"] == 2:
        st.markdown('<div class="ws-section-header">Diagnóstico de Visión Artificial</div>', unsafe_allow_html=True)
        
        col_c1, col_c2 = st.columns([1, 2])
        with col_c1:
            st.markdown("""
            <div style="background:rgba(0, 212, 170, 0.05); border:1px solid rgba(0, 212, 170, 0.2); border-radius:12px; padding:20px;">
                <div style="font-size:14px; font-weight:700; color:#00d4aa; margin-bottom:12px; text-transform:uppercase; letter-spacing:1px;">CONFIGURACIÓN ÓPTIMA</div>
                <div style="font-size:13px; color:#8899aa; line-height:1.6;">
                    Los parámetros han sido calibrados para maximizar la precisión en este tipo de tomas:<br><br>
                    • <b>Precisión</b>: 0.5s (Análisis de alta densidad)<br>
                    • <b>Umbral Confianza</b>: 25% (Sensibilidad mejorada)<br><br>
                    Verifica en la imagen de la derecha que los jugadores y el balón están siendo detectados correctamente.
                </div>
            </div>
            """, unsafe_allow_html=True)
            # Parámetros fijos
            sample_rate = 0.5
            confidence = 25
            st.session_state["analysis_params"] = {"rate": sample_rate, "conf": confidence}
        
        with col_c2:
            _render_model_diagnostics(st.session_state.get("video_path"), "yolo", confidence)

        c_prev, c_next = st.columns(2)
        if c_prev.button("← Volver", use_container_width=True):
            st.session_state["analysis_step"] = 1
            st.rerun()
        if c_next.button("Siguiente: Marcado Manual →", use_container_width=True):
            st.session_state["analysis_step"] = 3
            st.rerun()

    # --- STEP 3: MARCADO MANUAL CON PLANTILLA ---
    elif st.session_state["analysis_step"] == 3:
        from pages.squad_players import PLANTILLA as PLANTILLA_LOCAL

        team_local = st.session_state.get("step1_local", "Local")
        team_visit = st.session_state.get("step1_visit", "Visitante")

        if "player_seeds" not in st.session_state:
            st.session_state["player_seeds"] = []
        if "pending_click" not in st.session_state:
            st.session_state["pending_click"] = None
        if "rival_squad" not in st.session_state:
            st.session_state["rival_squad"] = [
                {"dorsal": i, "nombre": f"Jugador {i}", "posicion": "Jugador"}
                for i in range(1, 12)
            ]

        st.markdown('<div class="ws-section-header">Identificación de Jugadores</div>', unsafe_allow_html=True)
        st.markdown(
            "<div style='color:#8899aa;font-size:13px;margin-bottom:16px;'>"
            "Haz clic sobre un jugador en el vídeo → selecciona su nombre en el panel lateral. "
            "Marca los 22 jugadores. El tracker los seguirá durante todo el partido."
            "</div>", unsafe_allow_html=True
        )

        col_canvas, col_panel = st.columns([3, 1])

        video_p = st.session_state.get("video_path", "")
        cap = cv2.VideoCapture(video_p)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_v = cap.get(cv2.CAP_PROP_FPS) or 30

        with col_canvas:
            marking_sec = st.slider("⏱ Momento del vídeo", 0.0, float(total_frames / fps_v), 3.0, step=0.5,
                help="Elige un momento donde todos los jugadores sean visibles (min 3-5 suele ser bueno)")
            zoom = st.slider("🔍 Zoom", 1.0, 3.0, 1.0, 0.5)

        cap.set(cv2.CAP_PROP_POS_MSEC, marking_sec * 1000)
        ret, frame_raw = cap.read()
        cap.release()

        if not ret:
            st.error("No se pudo leer el frame del vídeo.")
        else:
            frame_rgb = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB)
            h_f, w_f = frame_rgb.shape[:2]

            # Dibujar seeds ya marcados
            frame_annotated = frame_rgb.copy()
            TEAM_COLORS = {team_local: (0, 212, 170), team_visit: (255, 77, 109), "Árbitro": (255, 200, 0)}
            for seed in st.session_state["player_seeds"]:
                sx, sy = seed["x"], seed["y"]
                color = TEAM_COLORS.get(seed.get("equipo", team_local), (200, 200, 200))
                cv2.circle(frame_annotated, (sx, sy), 12, color, -1)
                cv2.circle(frame_annotated, (sx, sy), 14, (255, 255, 255), 2)
                label = f"{seed.get('dorsal','?')} {seed.get('nombre','')[:10]}"
                cv2.putText(frame_annotated, label, (sx + 15, sy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

            if st.session_state["pending_click"]:
                px, py = st.session_state["pending_click"]
                cv2.circle(frame_annotated, (px, py), 14, (255, 140, 0), -1)
                cv2.circle(frame_annotated, (px, py), 16, (255, 255, 255), 2)
                cv2.putText(frame_annotated, "?", (px + 16, py + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            base_w = 860
            display_w = int(base_w * zoom)
            scale = display_w / w_f
            display_h = int(h_f * scale)

            with col_canvas:
                canvas_result = st_canvas(
                    fill_color="rgba(255,140,0,0.4)", stroke_width=2, stroke_color="#ff8c00",
                    background_image=Image.fromarray(frame_annotated),
                    update_streamlit=True, height=display_h, width=display_w,
                    drawing_mode="point", point_display_radius=7,
                    key=f"canvas_step3_{int(marking_sec*10)}_{zoom}",
                )

            if canvas_result.json_data and canvas_result.json_data.get("objects"):
                objs = canvas_result.json_data["objects"]
                if objs:
                    last = objs[-1]
                    if last["type"] == "circle":
                        cx_click = int(last["left"] / scale)
                        cy_click = int(last["top"] / scale)
                        if st.session_state["pending_click"] != (cx_click, cy_click):
                            st.session_state["pending_click"] = (cx_click, cy_click)
                            st.rerun()

        # Panel lateral
        with col_panel:
            n_local = sum(1 for s in st.session_state["player_seeds"] if s.get("equipo") == team_local)
            n_visit = sum(1 for s in st.session_state["player_seeds"] if s.get("equipo") == team_visit)
            n_ref   = sum(1 for s in st.session_state["player_seeds"] if s.get("equipo") == "Árbitro")

            st.markdown(f"""
            <div style="background:rgba(0,212,170,0.08);border:1px solid rgba(0,212,170,0.2);
                        border-radius:10px;padding:12px;margin-bottom:12px;">
                <div style="font-size:11px;color:#8899aa;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">MARCADOS</div>
                <div style="display:flex;gap:6px;flex-wrap:wrap;">
                    <span style="background:#00d4aa22;color:#00d4aa;padding:3px 8px;border-radius:6px;font-size:12px;font-weight:700;">{team_local[:8]}: {n_local}/11</span>
                    <span style="background:#ff4d6d22;color:#ff4d6d;padding:3px 8px;border-radius:6px;font-size:12px;font-weight:700;">{team_visit[:8]}: {n_visit}/11</span>
                    <span style="background:#ffc80022;color:#ffc800;padding:3px 8px;border-radius:6px;font-size:12px;font-weight:700;">Árb: {n_ref}/3</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.session_state["pending_click"]:
                px, py = st.session_state["pending_click"]
                st.markdown(
                    f"<div style='background:rgba(255,140,0,0.12);border:1px solid #ff8c00;"
                    f"border-radius:8px;padding:10px;margin-bottom:10px;'>"
                    f"<div style='color:#ff8c00;font-size:11px;font-weight:700;'>📍 CLICK EN ({px}, {py})</div>"
                    f"<div style='color:#aaa;font-size:11px;margin-top:4px;'>Selecciona el jugador:</div>"
                    f"</div>", unsafe_allow_html=True
                )

                equipo_sel = st.selectbox("Equipo", [team_local, team_visit, "Árbitro"], key="sel_equipo")

                if equipo_sel == team_local:
                    ya = {s["nombre"] for s in st.session_state["player_seeds"] if s.get("equipo") == team_local}
                    jugadores_map = {f"{j['dorsal']} · {j['nombre']} ({j['posicion']})": j for j in PLANTILLA_LOCAL if j["nombre"] not in ya}
                elif equipo_sel == team_visit:
                    ya = {s["nombre"] for s in st.session_state["player_seeds"] if s.get("equipo") == team_visit}
                    jugadores_map = {f"{j['dorsal']} · {j['nombre']} ({j['posicion']})": j for j in st.session_state["rival_squad"] if j["nombre"] not in ya}
                else:
                    n_a = n_ref + 1
                    opts = [f"Árbitro {n_a}", "Árbitro asistente 1", "Árbitro asistente 2"]
                    jugadores_map = {o: {"dorsal": 0, "nombre": o, "posicion": "Árbitro"} for o in opts}

                opciones = list(jugadores_map.keys())
                if opciones:
                    sel = st.selectbox("Jugador", opciones, key="sel_jugador")
                    jugador_data = jugadores_map.get(sel, {})
                    if st.button("✅ Confirmar", use_container_width=True, type="primary"):
                        st.session_state["player_seeds"].append({
                            "x": px, "y": py,
                            "nombre": jugador_data.get("nombre", sel),
                            "dorsal": jugador_data.get("dorsal", 0),
                            "posicion": jugador_data.get("posicion", ""),
                            "equipo": equipo_sel,
                        })
                        st.session_state["pending_click"] = None
                        st.toast(f"✅ {jugador_data.get('nombre', sel)} marcado")
                        st.rerun()
                else:
                    st.info("Todos los jugadores de este equipo ya están marcados.")

                if st.button("❌ Cancelar", use_container_width=True):
                    st.session_state["pending_click"] = None
                    st.rerun()
            else:
                st.markdown(
                    "<div style='color:#8899aa;font-size:12px;text-align:center;padding:20px;'>"
                    "👆 Haz clic sobre un jugador en el vídeo para identificarlo</div>",
                    unsafe_allow_html=True
                )

            st.markdown("---")

            if st.session_state["player_seeds"]:
                st.markdown("<div style='font-size:11px;color:#8899aa;text-transform:uppercase;margin-bottom:8px;'>MARCADOS</div>", unsafe_allow_html=True)
                for i, seed in enumerate(st.session_state["player_seeds"]):
                    eq_color = "#00d4aa" if seed["equipo"] == team_local else ("#ff4d6d" if seed["equipo"] == team_visit else "#ffc800")
                    col_s1, col_s2 = st.columns([4, 1])
                    col_s1.markdown(
                        f"<div style='font-size:12px;color:{eq_color};font-weight:700;'>{seed.get('dorsal','?')} · {seed['nombre']}</div>"
                        f"<div style='font-size:10px;color:#5a6a7e;'>{seed['equipo']} · {seed.get('posicion','')}</div>",
                        unsafe_allow_html=True
                    )
                    if col_s2.button("✕", key=f"del_seed_{i}"):
                        st.session_state["player_seeds"].pop(i)
                        st.rerun()

            with st.expander(f"✏️ Editar plantilla {team_visit}"):
                rival_squad_new = []
                for j in st.session_state["rival_squad"]:
                    nombre_nuevo = st.text_input(f"#{j['dorsal']}", value=j["nombre"], key=f"rival_{j['dorsal']}")
                    rival_squad_new.append({**j, "nombre": nombre_nuevo})
                if st.button("Guardar plantilla rival", use_container_width=True):
                    st.session_state["rival_squad"] = rival_squad_new
                    st.toast("Plantilla rival guardada")

        # Guardar seeds en formato compatible con video_processor
        seeds_final = [{"x": s["x"], "y": s["y"], "nombre": s["nombre"],
                        "dorsal": s["dorsal"], "posicion": s["posicion"], "equipo": s["equipo"]}
                       for s in st.session_state["player_seeds"]]
        st.session_state["manual_seeds"] = seeds_final
        st.session_state["jugadores_local"] = [
            {"nombre": s["nombre"], "dorsal": s["dorsal"], "posicion": s["posicion"]}
            for s in seeds_final if s["equipo"] == team_local
        ]
        st.session_state["jugadores_visit"] = [
            {"nombre": s["nombre"], "dorsal": s["dorsal"], "posicion": s["posicion"]}
            for s in seeds_final if s["equipo"] == team_visit
        ]

        st.markdown("<br>", unsafe_allow_html=True)
        c_prev, c_next = st.columns(2)
        if c_prev.button("← Volver", use_container_width=True):
            st.session_state["analysis_step"] = 2
            st.rerun()
            if st.button("LANZAR ANÁLISIS AHORA", use_container_width=True, type="primary"):
                st.session_state["processing"] = True
                params = st.session_state.get("analysis_params", {"rate": 0.5, "conf": 25})
                config = {
                    "team": st.session_state.get("step1_local", "Cadiz CF"),
                    "rival": st.session_state.get("step1_visit", "Rival"),
                    "match_date": str(st.session_state.get("step1_date")),
                    "sample_rate": params["rate"],
                    "detection_mode": "yolo",
                    "confidence": params["conf"],
                    "manual_seeds": st.session_state.get("manual_seeds", []),
                    "jugadores_local": st.session_state.get("jugadores_local", []),
                    "jugadores_visit": st.session_state.get("jugadores_visit", []),
                }
                st.session_state["analysis_config"] = config
                
                # Reloj de arena animado durante el proceso
                with st.spinner("⏳ Analizando vídeo..."):
                    st.markdown("""
                        <div style="text-align:center;padding:24px;background:rgba(0,212,170,0.05);border-radius:12px;border:1px dashed #00d4aa;margin-bottom:20px;">
                            <div style="font-size:50px;animation: hourglass_spin 2.5s ease-in-out infinite;">⏳</div>
                            <style>
                                @keyframes hourglass_spin {
                                    0% { transform: rotate(0deg); }
                                    45% { transform: rotate(180deg); }
                                    55% { transform: rotate(180deg); }
                                    100% { transform: rotate(360deg); }
                                }
                            </style>
                            <div style="font-size:14px;color:#00d4aa;font-weight:700;margin-top:12px;letter-spacing:1px;">IA PROCESANDO MOVIMIENTOS...</div>
                        </div>
                    """, unsafe_allow_html=True)
                    run_analysis_real(config)
                
                st.session_state["processing"] = False
                st.session_state["analysis_step"] = 5
                st.rerun()
        else:
            st.warning("Ya hay un análisis en curso...")

        if st.button("← Cancelar / Volver", use_container_width=True):
            st.session_state["analysis_step"] = 3
            st.rerun()

    # --- STEP 5: RESUMEN ---
    elif st.session_state["analysis_step"] == 5:
        st.markdown('<div class="ws-section-header">Resultados del Análisis</div>', unsafe_allow_html=True)
        
        if st.session_state.get("analysis_done"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Jugadores", len(st.session_state.get("resultados_jugadores", {})))
            c2.metric("Eventos Balón", len(st.session_state.get("ball_events", [])))
            c3.metric("Frames", st.session_state.get("frames_analizados", 0))
            c4.metric("Detecciones", f"{st.session_state.get('total_detecciones', 0):,}")
            
            st.markdown("<br>", unsafe_allow_html=True)
            col_d1, col_d2 = st.columns(2)
            if col_d1.button("📂 Ir al Dashboard Colectivo", use_container_width=True, type="primary"):
                st.session_state["page"] = "datos_colectivos"
                st.rerun()
            if col_d2.button("✂️ Ver Clips Generados", use_container_width=True):
                st.session_state["page"] = "partido_clips"
                st.rerun()
        else:
            st.error("No hay resultados cargados.")

        if st.button("➕ Iniciar Nuevo Análisis", use_container_width=True):
            st.session_state["analysis_step"] = 1
            st.session_state["analysis_done"] = False
            st.rerun()


def run_analysis_real(config: dict):
    progress_bar = st.progress(0)
    status_box = st.empty()
    video_path = st.session_state.get("video_path", "")

    try:
        from core.pipeline.video_pipeline import VideoAnalysisPipeline
        pipeline = VideoAnalysisPipeline(config)
        resultados_finales = {}

        for progreso, estado, resultados in pipeline.process(video_path):
            progress_bar.progress(int(progreso))
            status_box.markdown(f"""
            <div style="background:rgba(17, 24, 25, 0.5);border:1px solid rgba(0, 212, 170, 0.2);border-radius:12px;padding:12px 20px;font-size:13px;color:#00d4aa; font-weight:600;">
                {estado}
            </div>
            """, unsafe_allow_html=True)
            if resultados: resultados_finales = resultados

        if resultados_finales:
            st.session_state["analysis_done"] = True
            st.session_state["resultados_jugadores"] = resultados_finales.get("tracks", {})
            st.session_state["ball_history"] = resultados_finales.get("ball_history", [])
            
            # Persistence save
            results_file = settings.OUTPUT_DIR / "results.json"
            results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(results_file, "w", encoding="utf-8") as f:
                # Convert Path objects to strings for JSON serializability if any
                json.dump(resultados_finales, f, ensure_ascii=False, indent=2, default=str)
                
            st.success("✅ Análisis completado y guardado.")
    except Exception as e:
        st.error(f"Error: {e}")