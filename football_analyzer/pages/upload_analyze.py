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


def _render_model_diagnostics(video_path: str, detection_mode: str, confidence: int):
    """Muestra que detecta el modelo en un frame de muestra del video cargado."""
    import cv2
    import numpy as np
    from pathlib import Path
    from modules.detector import yolo_model_available, detect_frame
    from modules.detector import YOLO_MODEL_PATH
    
    yolo_ok = yolo_model_available()

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        if yolo_ok:
            st.markdown(f"""
            <div style="background:rgba(0, 212, 170, 0.1);border:1px solid rgba(0, 212, 170, 0.3);border-radius:12px;padding:16px;">
                <div style="font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#00d4aa;font-weight:700;">
                    MODELO YOLO V8 ACTIVO
                </div>
                <div style="font-size:13px;color:#fff;margin-top:4px;">{Path(str(YOLO_MODEL_PATH)).name}</div>
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

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    target_second = min(180, total_frames / fps * 0.2) if fps > 0 else 0
    target_frame = int(target_second * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        st.error("No se pudo leer el frame.")
        return

    with st.spinner("⏳ Ejecutando diagnóstico..."):
        dets = detect_frame(frame, mode=detection_mode, confidence=confidence / 100)

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
    results_file = Path("c:/apped/football_analyzer/output/results.json")
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
            local_videos = [f.name for f in VIDEO_DIR.glob("*") if f.suffix.lower() in [".mp4",".avi",".mkv",".mov"]]
            selected_local = st.selectbox("O selecciona uno existente:", [""] + local_videos)
            
            if uploaded_file:
                video_path = UPLOAD_DIR / uploaded_file.name
                with open(video_path, "wb") as f: f.write(uploaded_file.read())
                st.session_state["video_path"] = str(video_path)
                st.session_state["video_name"] = uploaded_file.name
            elif selected_local:
                st.session_state["video_path"] = str(VIDEO_DIR / selected_local)
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

    # --- STEP 3: MARCADO MANUAL ---
    elif st.session_state["analysis_step"] == 3:
        st.markdown('<div class="ws-section-header">Marcado Manual (Semillas de Tracking)</div>', unsafe_allow_html=True)
        st.info("💡 **Tip Pro**: ¿Ves jugadores no detectados o duplicados en el diagnóstico anterior? Haz clic sobre su base aquí para marcarlos manualmente. El sistema usará estos puntos para identificarlos y rastrearlos por todo el vídeo.")
        
        video_p = st.session_state.get("video_path")
        import cv2
        cap = cv2.VideoCapture(video_p)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        marking_sec = st.slider("Momento del vídeo", 0.0, float(total_frames/fps), 1.0, step=0.1)
        cap.set(cv2.CAP_PROP_POS_MSEC, marking_sec * 1000)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            
            col_z1, col_z2 = st.columns([1, 2])
            with col_z1:
                zoom = st.slider("🔍 Zoom de imagen (Scroll para ampliar)", 1.0, 4.0, 1.0, 0.5, help="Aumenta el tamaño para marcar jugadores lejanos con precisión.")
            
            base_w = 900
            display_w = int(base_w * zoom)
            scale = display_w / w
            display_h = int(h * scale)
            
            # Contenedor con scroll para zoom
            st.markdown(f"""
            <style>
            .canvas-container {{
                width: 100%;
                max-height: 700px;
                overflow: auto;
                border: 1px solid rgba(0, 212, 170, 0.2);
                border-radius: 12px;
                background: #000;
            }}
            </style>
            """, unsafe_allow_html=True)

            with st.container():
                st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
                canvas_result = st_canvas(
                    fill_color="rgba(0, 212, 170, 0.3)", stroke_width=2, stroke_color="#00d4aa",
                    background_image=Image.fromarray(frame_rgb), update_streamlit=True,
                    height=display_h, width=display_w, drawing_mode="point",
                    point_display_radius=5, key="canvas_step3",
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            if canvas_result.json_data:
                seeds = [{"x": int(obj["left"]/scale), "y": int(obj["top"]/scale)} 
                         for obj in canvas_result.json_data["objects"] if obj["type"]=="circle"]
                st.session_state["manual_seeds"] = seeds
                if seeds: st.toast(f"📍 {len(seeds)} jugadores marcados")

        c_prev, c_next = st.columns(2)
        if c_prev.button("← Volver", use_container_width=True):
            st.session_state["analysis_step"] = 2
            st.rerun()
        if c_next.button("Siguiente: Iniciar Motor →", use_container_width=True, type="primary"):
            st.session_state["analysis_step"] = 4
            st.rerun()

    # --- STEP 4: PROCESAMIENTO ---
    elif st.session_state["analysis_step"] == 4:
        st.markdown('<div class="ws-section-header">Ejecución del Motor Analytics</div>', unsafe_allow_html=True)
        
        if not st.session_state.get("processing", False):
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
                    "manual_seeds": st.session_state.get("manual_seeds", [])
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
        from modules.video_processor import VideoProcessor
        processor = VideoProcessor(video_path, config)
        resultados_finales = {}

        for progreso, estado, resultados in processor.process():
            progress_bar.progress(int(progreso))
            status_box.markdown(f"""
            <div style="background:rgba(17, 24, 25, 0.5);border:1px solid rgba(0, 212, 170, 0.2);border-radius:12px;padding:12px 20px;font-size:13px;color:#00d4aa; font-weight:600;">
                {estado}
            </div>
            """, unsafe_allow_html=True)
            if resultados: resultados_finales = resultados

        if resultados_finales:
            st.session_state["analysis_done"] = True
            st.session_state["resultados_jugadores"] = resultados_finales.get("resultados_jugadores", {})
            st.session_state["heatmap_x"] = resultados_finales.get("heatmap_x", [])
            st.session_state["heatmap_y"] = resultados_finales.get("heatmap_y", [])
            st.session_state["ball_events"] = resultados_finales.get("ball_events", [])
            st.session_state["total_detecciones"] = resultados_finales.get("total_detecciones", 0)
            st.session_state["frames_analizados"] = resultados_finales.get("frames_analizados", 0)
            
            # Persistence save
            results_file = Path("c:/apped/football_analyzer/output/results.json")
            results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(resultados_finales, f, ensure_ascii=False, indent=2)
                
            st.success("✅ Análisis completado y guardado.")
    except Exception as e:
        st.error(f"Error: {e}")