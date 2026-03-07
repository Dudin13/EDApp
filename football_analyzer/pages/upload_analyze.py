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


def _render_model_diagnostics(video_path: str, detection_mode: str, confidence: int):
    """Muestra que detecta el modelo en un frame de muestra del video cargado."""
    import cv2
    import numpy as np
    from pathlib import Path

    # Estado del modelo
    try:
        from modules.detector import yolo_model_available, detect_frame, classify_team
        from modules.detector import YOLO_MODEL_PATH
        yolo_ok = yolo_model_available()
    except Exception as e:
        st.error(f"Error cargando detector: {e}")
        return

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        if yolo_ok:
            st.markdown(f"""
            <div style="background:#0d1f14;border:1px solid #00d4aa44;border-radius:8px;padding:12px 16px;">
                <div style="font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#00d4aa;font-weight:600;">
                    Modelo YOLO activo
                </div>
                <div style="font-size:12px;color:#8899aa;margin-top:4px;">
                    {Path(str(YOLO_MODEL_PATH)).name}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#1e1220;border:1px solid #ff4d6d44;border-radius:8px;padding:12px 16px;">
                <div style="font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#ff4d6d;font-weight:600;">
                    Sin modelo YOLO local
                </div>
                <div style="font-size:12px;color:#8899aa;margin-top:4px;">Usando Roboflow</div>
            </div>
            """, unsafe_allow_html=True)

    with col_m2:
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1e2a3a;border-radius:8px;padding:12px 16px;">
            <div style="font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#5a6a7e;font-weight:600;">
                Motor seleccionado
            </div>
            <div style="font-size:13px;color:#fff;margin-top:4px;font-weight:600;">{detection_mode.upper()}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if not video_path or not Path(video_path).exists():
        st.info("Carga un vídeo para ver el diagnóstico del modelo.")
        return

    if st.button("🔍 Analizar frame de muestra", key="btn_diag_frame"):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Saltar al minuto 3 del vídeo (o al 20% si es más corto)
        target_second = min(180, total_frames / fps * 0.2) if fps > 0 else 0
        target_frame = int(target_second * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            st.error("No se pudo leer el frame de muestra.")
            return

        # Detección raw con conf muy baja para diagnóstico
        with st.spinner("Ejecutando detección..."):
            try:
                dets = detect_frame(frame, mode=detection_mode, confidence=confidence / 100)
            except Exception as e:
                st.error(f"Error en detección: {e}")
                return

        # Dibujar detecciones sobre el frame
        CLASS_COLORS = {
            "ball":       (0, 255, 255),   # Amarillo
            "player":     (0, 212, 170),   # Teal
            "goalkeeper": (255, 200, 0),   # Naranja
            "referee":    (0, 0, 255),     # Rojo
        }
        vis_frame = frame.copy()
        for det in dets:
            x, y, w, h = det["x"], det["y"], det["w"], det["h"]
            clase = det.get("clase", "player")
            conf = det.get("confianza", 0)
            color = CLASS_COLORS.get(clase, (200, 200, 200))

            # Dibujar bounding box
            cv2.rectangle(vis_frame,
                          (x - w//2, y - h//2),
                          (x + w//2, y + h//2),
                          color, 2)
            # Etiqueta
            label = f"{clase} {conf:.2f}"
            cv2.putText(vis_frame, label, (x - w//2, y - h//2 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Convertir BGR → RGB para Streamlit
        vis_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
        st.image(vis_rgb, caption=f"Frame ~{target_second:.0f}s | {len(dets)} detecciones", use_container_width=True)

        # Tabla resumen por clase
        if dets:
            import pandas as pd
            resumen = {}
            for det in dets:
                clase = det.get("clase", "player")
                conf = det.get("confianza", 0)
                if clase not in resumen:
                    resumen[clase] = {"count": 0, "max_conf": 0.0, "min_conf": 1.0}
                resumen[clase]["count"] += 1
                resumen[clase]["max_conf"] = max(resumen[clase]["max_conf"], conf)
                resumen[clase]["min_conf"] = min(resumen[clase]["min_conf"], conf)

            rows = []
            class_icons = {"ball": "⚽", "player": "👤", "goalkeeper": "🧤", "referee": "🟨"}
            for clase, stats in sorted(resumen.items()):
                rows.append({
                    "Clase": f"{class_icons.get(clase, '•')} {clase}",
                    "Detectados": stats["count"],
                    "Conf. máx.": f"{stats['max_conf']:.3f}",
                    "Conf. mín.": f"{stats['min_conf']:.3f}",
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Alerta si no se detecta el balón
            clases_detectadas = set(d.get("clase") for d in dets)
            if "ball" not in clases_detectadas:
                st.markdown("""
                <div style="background:#1e1220;border:1px solid #ff6b3544;border-radius:8px;padding:10px 14px;font-size:12px;color:#ff6b35;margin-top:8px;">
                    ⚠️ <strong>Balón no detectado</strong> en este frame.
                    El modelo necesita más entrenamiento para detectar el balón de forma consistente.
                    Prueba a bajar la confianza mínima o selecciona un frame donde el balón sea visible.
                </div>
                """, unsafe_allow_html=True)
            else:
                ball_dets = [d for d in dets if d.get("clase") == "ball"]
                max_ball_conf = max(d.get("confianza", 0) for d in ball_dets)
                if max_ball_conf < 0.1:
                    st.markdown(f"""
                    <div style="background:#1e1a10;border:1px solid #FFD70044;border-radius:8px;padding:10px 14px;font-size:12px;color:#FFD700;margin-top:8px;">
                        ⚠️ <strong>Balón detectado pero con baja confianza</strong> (máx: {max_ball_conf:.3f}).
                        Se recomienda continuar entrenando el modelo para mejorar la detección del balón.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success(f"✅ Balón detectado con confianza {max_ball_conf:.3f}")
        else:
            st.warning("No se detectó nada en este frame. Prueba otro vídeo o baja el umbral de confianza.")


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
        # Option 1: Upload
        uploaded_file = st.file_uploader(
            "Sube un nuevo vídeo",
            type=["mp4", "avi", "mkv", "mov"],
            label_visibility="collapsed"
        )
        
        # Option 2: Select from 'videos/' folder
        VIDEO_DIR = Path(__file__).parent.parent / "videos"
        VIDEO_DIR.mkdir(exist_ok=True)
        local_videos = [f.name for f in VIDEO_DIR.glob("*") if f.suffix.lower() in [".mp4", ".avi", ".mkv", ".mov"]]
        
        if local_videos:
            selected_local = st.selectbox("O selecciona un vídeo ya existente:", [""] + local_videos)
            if selected_local:
                st.session_state["video_path"] = str(VIDEO_DIR / selected_local)
                st.session_state["video_name"] = selected_local


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
        st.success(f"✅ **{uploaded_file.name}** cargado correctamente")

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
    
    # Pre-inicializar valores vacíos para que los inputs no exploten al sobrescribir desde Pandas
    for col_prefix in ["local", "visit"]:
        for i in range(20):
            if f"{col_prefix}_dorsal_{i}" not in st.session_state:
                st.session_state[f"{col_prefix}_dorsal_{i}"] = i + 1
            if f"{col_prefix}_nombre_{i}" not in st.session_state:
                st.session_state[f"{col_prefix}_nombre_{i}"] = ""
            if f"{col_prefix}_pos_{i}" not in st.session_state:
                st.session_state[f"{col_prefix}_pos_{i}"] = ""

    st.markdown("""
    <div style="font-size:13px;color:#8899aa;margin-bottom:12px;">
        Puedes rellenar los datos manualmente o subir un archivo Excel/CSV con columnas: <code>Dorsal</code>, <code>Nombre</code>, <code>Posición</code>
    </div>
    """, unsafe_allow_html=True)

    POSICIONES = ["", "Portero", "Lateral D", "Lateral I", "Central", "Pivote",
                  "Mediocentro", "Interior D", "Interior I", "Mediapunta",
                  "Extremo D", "Extremo I", "Delantero Centro"]

    import pandas as pd
    col_upload1, col_upload2 = st.columns(2)
    template_local = col_upload1.file_uploader(f"Plantilla {team_local or 'Local'}", type=["csv", "xlsx"], key="template_local")
    template_visit = col_upload2.file_uploader(f"Plantilla {team_visit or 'Visitante'}", type=["csv", "xlsx"], key="template_visit")

    def normalize_str(s):
        import unicodedata
        if pd.isna(s): return ""
        s = str(s).strip().lower()
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    def map_pos(p_raw):
        p = normalize_str(p_raw)
        if "porter" in p: return "Portero"
        elif "lateral i" in p or ("lateral" in p and "izq" in p) or "carrilero i" in p: return "Lateral I"
        elif "lateral" in p or "carrilero" in p: return "Lateral D"
        elif "central" in p or "defensa" in p: return "Central"
        elif "pivote" in p or "mediocentro def" in p: return "Pivote"
        elif "interior i" in p: return "Interior I"
        elif "interior" in p: return "Interior D"
        elif "mediapunta" in p: return "Mediapunta"
        elif "med" in p or "centrocampista" in p: return "Mediocentro"
        elif "extremo i" in p or ("extremo" in p and "izq" in p): return "Extremo I"
        elif "extremo" in p: return "Extremo D"
        elif "delantero" in p or "punta" in p or "ariete" in p or "ataque" in p: return "Delantero Centro"
        return ""

    def process_df(df, prefix):
        # Mapear columnas dinámicamente ignorando tildes y mayúsculas
        cols = {normalize_str(c): c for c in df.columns}
        col_dor = cols.get("dorsal", cols.get("numero", None))
        col_nom = cols.get("nombre", cols.get("jugador", None))
        col_pos = cols.get("posicion", cols.get("demarcacion", None))
        
        for i in range(20):
            if i < len(df):
                if col_dor and pd.notna(df.iloc[i][col_dor]):
                    st.session_state[f"{prefix}_dorsal_{i}"] = int(pd.to_numeric(df.iloc[i][col_dor], errors='coerce') or (i+1))
                if col_nom and pd.notna(df.iloc[i][col_nom]):
                    st.session_state[f"{prefix}_nombre_{i}"] = str(df.iloc[i][col_nom])
                else:
                    st.session_state[f"{prefix}_nombre_{i}"] = ""
                    
                if col_pos and pd.notna(df.iloc[i][col_pos]):
                    pos_mapped = map_pos(df.iloc[i][col_pos])
                    if pos_mapped in POSICIONES:
                        st.session_state[f"{prefix}_pos_{i}"] = pos_mapped
            else:
                # Limpiar celdas sobrantes si el nuevo excel tiene menos jugadores
                st.session_state[f"{prefix}_dorsal_{i}"] = i + 1 # Reset dorsal to default
                st.session_state[f"{prefix}_nombre_{i}"] = ""
                st.session_state[f"{prefix}_pos_{i}"] = ""

    if template_local:
        try:
            df_local = pd.read_excel(template_local) if template_local.name.endswith(".xlsx") else pd.read_csv(template_local)
            if st.session_state.get("last_local_file") != template_local.name:
                process_df(df_local, "local")
                st.session_state["last_local_file"] = template_local.name
        except Exception as e:
            col_upload1.error(f"Error leyendo archivo local: {e}")
            
    if template_visit:
        try:
            df_visit = pd.read_excel(template_visit) if template_visit.name.endswith(".xlsx") else pd.read_csv(template_visit)
            if st.session_state.get("last_visit_file") != template_visit.name:
                process_df(df_visit, "visit")
                st.session_state["last_visit_file"] = template_visit.name
        except Exception as e:
            col_upload2.error(f"Error leyendo archivo visitante: {e}")

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
            
            dorsal = c1.number_input("D", min_value=1, max_value=99,
                                     key=f"local_dorsal_{i}", label_visibility="collapsed")
            nombre = c2.text_input("Nombre", placeholder=f"Jugador {i + 1}",
                                   key=f"local_nombre_{i}", label_visibility="collapsed")
            posicion = c3.selectbox("Pos", POSICIONES,
                                    key=f"local_pos_{i}", label_visibility="collapsed")
            
            if nombre.strip():
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
            
            dorsal = c1.number_input("D", min_value=1, max_value=99,
                                     key=f"visit_dorsal_{i}", label_visibility="collapsed")
            nombre = c2.text_input("Nombre", placeholder=f"Jugador {i + 1}",
                                   key=f"visit_nombre_{i}", label_visibility="collapsed")
            posicion = c3.selectbox("Pos", POSICIONES,
                                    key=f"visit_pos_{i}", label_visibility="collapsed")
            
            if nombre.strip():
                jugadores_visit.append({"dorsal": dorsal, "nombre": nombre,
                                        "equipo": team_visit, "posicion": posicion})

    # ── BLOQUE 4: OPCIONES DE ANÁLISIS ───────────────────────────────────────
    st.markdown('<div class="ws-section-header">04 — Configuración del análisis</div>', unsafe_allow_html=True)

    with st.expander("⚙️ Opciones avanzadas", expanded=False):
        col_opt1, col_opt2, col_opt3 = st.columns(3)
        detection_mode = col_opt1.selectbox(
            "Motor de detección",
            ["yolo (local)", "roboflow", "auto"],
            help="'auto' usa YOLOv8 si está entrenado, sino Roboflow"
        )
        if detection_mode == "yolo (local)":
            detection_mode = "yolo"

        sample_rate = col_opt2.slider("Analizar 1 frame cada (seg)", 1, 10, 2,
                                      help="Menor = más preciso pero más lento")
        confidence = col_opt3.slider("Confianza de detección (%)", 5, 80, 30)

    # ── BLOQUE 5: DIAGNÓSTICO DEL MODELO ─────────────────────────────────────
    st.markdown('<div class="ws-section-header">05 — Diagnóstico del modelo</div>', unsafe_allow_html=True)

    with st.expander("🔬 Ver qué detecta el modelo en el vídeo actual", expanded=False):
        _render_model_diagnostics(st.session_state.get("video_path"), detection_mode, confidence)

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
            st.session_state["manual_tagging_phase"] = True
            st.rerun()

    # ── FASE DE ETIQUETADO MANUAL ────────────────────────────────────────────
    if st.session_state.get("manual_tagging_phase", False):
        st.markdown('<div class="ws-section-header">06 — Asignación Manual Inicial (Frame 1)</div>', unsafe_allow_html=True)
        st.info("💡 Asigna de forma visual la identidad de los jugadores para mejorar la precisión del seguimiento.")
        
        config = st.session_state["analysis_config"]
        video_path = st.session_state.get("video_path", "")
        
        if "first_frame_dets" not in st.session_state:
            with st.spinner("🖼️ Extrayendo y analizando primer frame clave..."):
                import cv2
                from modules.detector import detect_frame
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * 2)) # Extraer segundo 2
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    # Guardamos la imagen en memoria para recortes
                    import numpy as np
                    from PIL import Image
                    st.session_state["first_frame_img"] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Detección
                    dets = detect_frame(frame, mode=config["detection_mode"], confidence=config["confidence"])
                    # Filtramos detecciones válidas
                    valid_dets = [d for d in dets if d["w"] > 10 and d["h"] > 10]
                    # Asignamos IDs temporales a las cajas
                    for idx, d in enumerate(valid_dets):
                        d["temp_id"] = idx
                    st.session_state["first_frame_dets"] = valid_dets
                    if "manual_tags" not in st.session_state:
                        st.session_state["manual_tags"] = {}
                else:
                    st.error("Error al leer el vídeo.")
                    st.session_state["manual_tagging_phase"] = False
                    st.rerun()
                    
        if "extra_manual_dets" not in st.session_state:
            st.session_state["extra_manual_dets"] = []
            st.session_state["last_click"] = None

        # Juntamos detecciones
        dets = st.session_state["first_frame_dets"]
        all_dets = list(dets) + st.session_state["extra_manual_dets"]
        img = st.session_state["first_frame_img"]

        import cv2
        from PIL import Image
        from streamlit_image_coordinates import streamlit_image_coordinates
        
        vis_img = img.copy()
        for det in all_dets:
            x, y, w, h = det["x"], det["y"], det["w"], det["h"]
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            
            color = (0, 0, 255) if det.get("is_manual") else (0, 212, 170)
            tid = det["temp_id"]
            
            # Dibujar caja
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            # Dibujar fondo para el texto
            cv2.rectangle(vis_img, (x1, max(0, y1 - 22)), (x1 + 32, max(0, y1)), color, -1)
            # Dibujar ID numérico
            cv2.putText(vis_img, str(tid), (x1 + 5, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
        st.markdown("### 🖱️ Haz clic sobre los jugadores que falten")
        st.caption("Si la IA no ha detectado a alguien (ej. el árbitro), **haz clic sobre él en la imagen** y aparecerá una nueva caja roja que podrás etiquetar abajo.")

        # Obtiene las coordenadas al ahcer click
        value = streamlit_image_coordinates(Image.fromarray(vis_img), key="img_click")
        
        if value is not None and value != st.session_state.get("last_click"):
            st.session_state["last_click"] = value
            new_id = len(dets) + len(st.session_state["extra_manual_dets"])
            st.session_state["extra_manual_dets"].append({
                "x": value["x"], "y": value["y"], "w": 40, "h": 60,
                "temp_id": new_id,
                "is_manual": True
            })
            st.rerun()
        
        # Opciones disponibles de los Excel/Formularios base
        plantilla_local = [f"[Local] {j['dorsal']} - {j['nombre']} ({j['posicion']})" for j in config["jugadores_local"] if j["nombre"]]
        plantilla_visit = [f"[Visit] {j['dorsal']} - {j['nombre']} ({j['posicion']})" for j in config["jugadores_visit"] if j["nombre"]]
        todas_las_opciones = ["Árbitro 🟨", "Balón ⚽"] + plantilla_local + plantilla_visit
        
        st.markdown("<br>### Asignar identidades por ID", unsafe_allow_html=True)
        
        num_cols = 4
        cols = st.columns(num_cols)
        
        # Tags guardados globalmente para deducir qué está "pillado"
        current_selections = list(st.session_state.get("manual_tags", {}).values())
        
        for idx_list, det in enumerate(all_dets):
            # idx real (puede ser de la IA o el manual que acabamos de añadir)
            tid = det["temp_id"]
            is_manual = det.get("is_manual", False)
            
            # Decorativo para ver de donde viene
            origen = "🔴 Manual" if is_manual else "🟢 IA"
            
            col = cols[idx_list % num_cols]
            with col:
                st.markdown(f"**Caja ID {tid}** [{origen}]")
                saved_val = st.session_state.get("manual_tags", {}).get(str(tid))
                
                # Construir opciones DENTRO del bucle para filtrar las ya elegidas
                mis_opciones = ["Descartar / No Asignar"]
                for opt in todas_las_opciones:
                    # Incluimos la opción si es la nuestra actual, o si está libre
                    if opt == saved_val or opt not in current_selections:
                        mis_opciones.append(opt)
                        
                default_idx = 0
                if saved_val and saved_val in mis_opciones:
                    default_idx = mis_opciones.index(saved_val)
                    
                sel = st.selectbox(
                    "Identidad", 
                    options=mis_opciones, 
                    key=f"tag_sel_{tid}",
                    index=default_idx,
                    label_visibility="collapsed"
                )
                
                # Actualizar el diccionario y forzar un rerun si hay una nueva asignación validada
                if sel != "Descartar / No Asignar" and sel != saved_val:
                    st.session_state["manual_tags"][str(tid)] = sel
                    st.rerun() # Fuerzo rerun para que las demas cajas vean que esto está "pillado"
                elif sel == "Descartar / No Asignar" and str(tid) in st.session_state.get("manual_tags", {}):
                    st.session_state["manual_tags"].pop(str(tid), None)
                    st.rerun() # Fuerzo rerun para liberar el nombre
                
                if is_manual:
                    if st.button("❌ Eliminar Caja", key=f"del_box_{tid}", use_container_width=True):
                        # Filter out the deleted box by its temp_id
                        st.session_state["extra_manual_dets"] = [d for d in st.session_state["extra_manual_dets"] if d["temp_id"] != tid]
                        # Remove from tags if it was selected
                        st.session_state["manual_tags"].pop(str(tid), None)
                        st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("⏭️ Omitir este paso (Auto-Tracking)"):
                st.session_state["manual_tags"] = {} # Limpiamos
                st.session_state["processing"] = True
                st.session_state["manual_tagging_phase"] = False
                st.rerun()
        with col2:
            if st.button("✅ Confirmar Etiquetas e Iniciar Análisis", type="primary"):
                # Transformamos los tags guardados al formato {"temp_id": "Arbitro" o dict del jugador local/visitante}
                final_seeds = {}
                for tid_str, text_val in st.session_state.get("manual_tags", {}).items():
                    tid = int(tid_str)
                    if text_val == "Árbitro 🟨": final_seeds[tid] = "referee"
                    elif text_val == "Balón ⚽": final_seeds[tid] = "ball"
                    elif text_val.startswith("[Local]"): final_seeds[tid] = text_val.replace("[Local] ", "")
                    elif text_val.startswith("[Visit]"): final_seeds[tid] = text_val.replace("[Visit] ", "")
                config["initial_seeds"] = final_seeds
                config["initial_dets"] = all_dets # guardamos TODAS las detecciones (IA + manuales)
                st.session_state["analysis_config"] = config
                
                st.session_state["processing"] = True
                st.session_state["manual_tagging_phase"] = False
                st.rerun()

    if st.session_state.get("processing", False) and not st.session_state.get("manual_tagging_phase", False):
        run_analysis_real(st.session_state.get("analysis_config", {}))
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
            
            # Limpiar memoria de tagging phase
            st.session_state.pop("first_frame_dets", None)
            st.session_state.pop("first_frame_img", None)
            st.session_state.pop("manual_tags", None)
            st.session_state.pop("extra_manual_dets", None)
            st.session_state.pop("last_click", None)
        else:
            st.warning("⚠️ El análisis terminó sin resultados. Comprueba que el vídeo contiene el partido.")

    except ImportError as e:
        st.error(f"❌ Dependencia no instalada: {e}")
    except Exception as e:
        st.error(f"❌ Error durante el análisis: {e}")
        import traceback
        with st.expander("Ver detalle del error"):
            st.code(traceback.format_exc())