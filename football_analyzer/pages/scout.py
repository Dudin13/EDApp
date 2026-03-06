import streamlit as st
st.set_page_config(page_title="Modo Analista", layout="wide", initial_sidebar_state="collapsed")
import pandas as pd
import os
import subprocess
from pathlib import Path
from datetime import datetime
import uuid
import json
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from dotenv import load_dotenv
import time

# Configuración de rutas y FFmpeg
load_dotenv(Path(__file__).parent.parent / ".env")
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "C:\\ffmpeg\\bin\\ffmpeg.exe")
CLIPS_DIR = Path("output") / "clips_manuales"
PROJECTS_DIR = Path("output") / "scout_projects"
TEMPLATES_FILE = Path("output") / "scout_templates.json"
CLIPS_DIR.mkdir(parents=True, exist_ok=True)
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

# Estilos CSS Avanzados (Glassmorphism + Analyst Mode + Floating Panel fixes)
st.markdown("""
    <style>
    .analyst-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 24px;
        border-radius: 16px;
        border-bottom: 3px solid #00d4aa;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .timeline-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(0, 212, 170, 0.1);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        transition: all 0.2s ease;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .timeline-card:hover {
        background: rgba(30, 41, 59, 0.7);
        border-color: rgba(0, 212, 170, 0.4);
    }
    .tag-badge {
        background: #00d4aa;
        color: #000;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 13px;
        font-weight: 800;
        text-transform: uppercase;
        display: inline-block;
    }
    .time-badge {
        font-family: monospace;
        color: #8899aa;
        font-size: 14px;
    }
    .timeline-info { flex: 1; }
    .timeline-actions { display: flex; gap: 8px; }
    
    .floating-btn-container {
        display: grid;
        grid-template-columns: 1fr;
        gap: 8px;
        margin-bottom: 15px;
    }
    
    /* Persistent Header Icon */
    [data-testid="stHeader"]::after {
        content: "⏳";
        position: fixed;
        right: 80px;
        top: 15px;
        font-size: 20px;
        z-index: 999999;
    }
    
    /* Compact Buttons */
    .stButton > button {
        border-radius: 8px !important;
        transition: all 0.2s !important;
        font-size: min(1.1vw, 14px) !important;
        padding: 0.4rem 0.2rem !important;
        height: auto !important;
        min-height: 40px !important;
        background: rgba(0, 212, 170, 0.1) !important;
        border: 1px solid rgba(0, 212, 170, 0.3) !important;
    }
    .stButton > button:hover {
        background: rgba(0, 212, 170, 0.2) !important;
        border-color: #00d4aa !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- FFmpeg Helpers ---
def _cortar_clip_manual(video_path, start, end, event_id):
    if not Path(FFMPEG_PATH).exists():
        return False, "❌ No se encontró FFmpeg en la ruta configurada."
    
    out_path = CLIPS_DIR / f"analista_{event_id}.mp4"
    duration = end - start
    
    cmd = [
        FFMPEG_PATH, "-y",
        "-ss", f"{start:.2f}",
        "-i", str(video_path),
        "-t", f"{duration:.2f}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26",
        "-c:a", "aac", "-movflags", "+faststart",
        str(out_path)
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True, str(out_path)
    except Exception as e:
        return False, str(e)

def _extract_frame(video_path, time_sec, out_path_img):
    cmd = [
        FFMPEG_PATH, "-y",
        "-ss", f"{time_sec:.2f}",
        "-i", str(video_path),
        "-vframes", "1",
        "-q:v", "2",
        str(out_path_img)
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        if Path(out_path_img).exists():
            return True
        return False
    except:
        return False

def _burn_overlay(video_path, overlay_path, out_path):
    cmd = [
        FFMPEG_PATH, "-y",
        "-i", str(video_path),
        "-i", str(overlay_path),
        "-filter_complex", "[0:v][1:v]overlay=0:0[out]",
        "-map", "[out]",
        "-map", "0:a?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "26",
        "-c:a", "copy",
        str(out_path)
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True, str(out_path)
    except Exception as e:
        return False, str(e)

# --- State Management Helpers ---
def init_session():
    if "scout_step" not in st.session_state:
        st.session_state.scout_step = "dashboard"
    if "manual_events" not in st.session_state:
        st.session_state.manual_events = []
    if "video_path" not in st.session_state:
        st.session_state.video_path = ""
    if "project_name" not in st.session_state:
        st.session_state.project_name = ""
    if "tag_templates" not in st.session_state:
        st.session_state.tag_templates = [
            {"id": "t1", "name": "Ataque", "cat": "Táctica", "pre": 5.0, "post": 4.0},
            {"id": "t2", "name": "Defensa", "cat": "Táctica", "pre": 5.0, "post": 4.0},
            {"id": "t3", "name": "Tiro", "cat": "Acción", "pre": 3.0, "post": 2.0},
            {"id": "t4", "name": "Falta", "cat": "Acción", "pre": 4.0, "post": 4.0},
        ]
    if "drawing_ev_id" not in st.session_state:
        st.session_state.drawing_ev_id = None
    if "editing_ev_id" not in st.session_state:
        st.session_state.editing_ev_id = None
    if "video_duration" not in st.session_state:
        st.session_state.video_duration = 0.0

def save_persistence():
    if st.session_state.project_name:
        p_file = PROJECTS_DIR / f"scout_{st.session_state.project_name}.json"
        data = {
            "video_path": st.session_state.video_path,
            "templates": st.session_state.tag_templates,
            "events": st.session_state.manual_events,
            "duration": st.session_state.video_duration
        }
        p_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

def save_template_global():
    TEMPLATES_FILE.write_text(json.dumps(st.session_state.tag_templates, indent=2), encoding="utf-8")
    st.toast("⏳ Plantilla guardada como predeterminada", icon="💾")

def load_template_global():
    if TEMPLATES_FILE.exists():
        try:
            st.session_state.tag_templates = json.loads(TEMPLATES_FILE.read_text(encoding="utf-8"))
        except: pass

def load_persistence(project_name):
    p_file = PROJECTS_DIR / f"scout_{project_name}.json"
    if p_file.exists():
        data = json.loads(p_file.read_text(encoding="utf-8"))
        st.session_state.video_path = data.get("video_path", "")
        # Fallback to check if video exists
        if not Path(st.session_state.video_path).exists():
           v_name = Path(st.session_state.video_path).stem
           # Try to find it generically
           pots = list(Path("uploads").glob(f"{v_name}.*")) + list(Path("videos").glob(f"{v_name}.*"))
           if pots: st.session_state.video_path = str(pots[0])
           
        st.session_state.tag_templates = data.get("templates", st.session_state.tag_templates)
        st.session_state.manual_events = data.get("events", [])
        st.session_state.video_duration = data.get("duration", 0.0)
        st.session_state.project_name = project_name
        return True
    return False

# --- Views ---
def view_dashboard():
    st.markdown('<div class="analyst-header"><h1>Modo Analista - Paso 1: Dashboard</h1><p>Inicia un nuevo análisis o continúa uno existente</p></div>', unsafe_allow_html=True)
    
    # Load default templates if empty
    if not st.session_state.manual_events and TEMPLATES_FILE.exists():
        load_template_global()
    
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("🆕 Nuevo Análisis")
            st.write("Carga un vídeo para crear un nuevo proyecto.")
            uploaded_file = st.file_uploader("Cargar vídeo (.mp4, .mov)", type=["mp4", "mov"])
            proj_name = st.text_input("Nombre del Proyecto (opcional)", placeholder="Ej. Partido_Final")
            
            if st.button("Continuar a Configuración ➡️", use_container_width=True, type="primary", disabled=not uploaded_file):
                save_path = Path("uploads") / uploaded_file.name
                save_path.parent.mkdir(exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Get real duration
                cap = cv2.VideoCapture(str(save_path))
                fps = cap.get(cv2.CAP_PROP_FPS)
                t_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                dur = t_frames / fps if fps > 0 else 600.0
                
                st.session_state.video_path = str(save_path)
                st.session_state.video_duration = dur
                st.session_state.project_name = proj_name if proj_name else Path(uploaded_file.name).stem
                st.session_state.manual_events = []
                st.session_state.scout_step = "config"
                save_persistence()
                st.rerun()

    with col2:
        with st.container(border=True):
            st.subheader("📂 Continuar Proyecto")
            projects = list(PROJECTS_DIR.glob("scout_*.json"))
            if not projects:
                st.info("No hay proyectos guardados.")
            else:
                p_names = [p.stem.replace("scout_", "") for p in projects]
                selected_project = st.selectbox("Selecciona un proyecto:", p_names)
                if st.button("Cargar y Continuar 🚀", use_container_width=True):
                    if load_persistence(selected_project):
                        st.session_state.scout_step = "editor"
                        st.rerun()
                    else:
                        st.error("Error al cargar el proyecto.")


def view_config():
    st.markdown('<div class="analyst-header"><h1>Modo Analista - Paso 2: Plantilla de Etiquetas</h1><p>Configura los botones y sus duraciones automáticas</p></div>', unsafe_allow_html=True)
    
    if st.button("🔙 Volver al Dashboard"):
        st.session_state.scout_step = "dashboard"
        st.rerun()
        
    st.write("Agrega o modifica los botones rápidos que usarás durante el análisis de vídeo. Define el tiempo previo y posterior que se cortará automáticamente al hacer clic en el botón.")
    
    with st.container(border=True):
        st.subheader("Botonera Actual")
        
        # Display editable list
        new_templates = []
        for i, t in enumerate(st.session_state.tag_templates):
            c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
            name = c1.text_input("Nombre Etiqueta", value=t['name'], key=f"t_name_{i}")
            pre = c2.number_input("Segundos Previos", value=float(t['pre']), min_value=0.0, step=0.5, key=f"t_pre_{i}")
            post = c3.number_input("Segundos Posteriores", value=float(t['post']), min_value=0.0, step=0.5, key=f"t_post_{i}")
            if c4.button("🗑️", key=f"t_del_{i}"):
                continue # Skip adding this to new_templates
            new_templates.append({"id": t['id'], "name": name, "cat": t.get('cat', ''), "pre": pre, "post": post})
            
        st.session_state.tag_templates = new_templates
        
        if st.button("➕ Añadir Botón"):
            st.session_state.tag_templates.append({
                "id": f"t_{uuid.uuid4().hex[:6]}", 
                "name": "Nueva Etiqueta", 
                "cat": "", 
                "pre": 5.0, 
                "post": 5.0
            })
            st.rerun()
            
    st.markdown("<br>", unsafe_allow_html=True)
    c_save, c_next = st.columns(2)
    if c_save.button("💾 Guardar como Plantilla Predeterminada", use_container_width=True):
        save_template_global()
    if c_next.button("✅ Terminar Configuración e Ir al Editor", use_container_width=True, type="primary"):
        save_persistence()
        st.session_state.scout_step = "editor"
        st.rerun()


def view_editor():
    if not st.session_state.video_path or not Path(st.session_state.video_path).exists():
        st.error("No se encontró el vídeo. Vuelve al Dashboard.")
        if st.button("Dashboard"):
            st.session_state.scout_step = "dashboard"
            st.rerun()
        return

    st.markdown('<div class="analyst-header"><h1>Modo Analista - Editor</h1><p>Analiza, etiqueta y recorta eventos del partido</p></div>', unsafe_allow_html=True)
    
    # Navigation / Dashboard button in sidebar (cleaner)
    with st.sidebar:
        if st.button("🔙 Dashboard"):
            st.session_state.scout_step = "dashboard"
            st.rerun()
        if st.button("⚙️ Configuración"):
            st.session_state.scout_step = "config"
            st.rerun()

    # Main Layout: 2 Columns (Video | Buttons)
    col_vid, col_tags = st.columns([4, 1])

    with col_vid:
        st.video(st.session_state.video_path)
        # Use simple caption or nothing as user said built-in bar is enough
        st.caption("ℹ️ Usa los controles del propio vídeo para navegar")

    with col_tags:
        st.markdown("### 🏷️ Etiquetas")
        
        # Link video current time to tags using JS or state if possible
        # Since we removed the slider, we might need a hidden way to get current time
        # Or just trust common streamling behavior. 
        # For now, keeping a small hidden or compact slider if really needed, 
        # but user said "already have a bar". Streamlit st.video doesn't sync back time easily without custom components.
        # I'll keep the session_state.current_time logic but maybe hide the slider label.
        st.session_state.current_time = st.number_input("Tiempo (s)", 0.0, float(st.session_state.video_duration), st.session_state.current_time, step=0.1, label_visibility="collapsed")
            
        for t in st.session_state.tag_templates:
            btn_html = f'<span class="tag-btn-text">{t["name"]}</span>'
            if st.button(t['name'], key=f"btn_tag_{t['id']}", use_container_width=True):
                ct = st.session_state.current_time
                new_ev = {
                    "id": str(uuid.uuid4())[:8],
                    "tag": t['name'],
                    "start": max(0, ct - t['pre']),
                    "end": min(st.session_state.video_duration, ct + t['post']),
                    "note": "",
                    "has_drawing": False,
                    "clip_path": ""
                }
                st.session_state.manual_events.append(new_ev)
                save_persistence()
                st.toast(f"⏳ '{t['name']}' registrado", icon="⏳")
    
    # Bottom Layout: Timeline
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown("### 📊 Timeline de Eventos Recortados")
    
    if st.session_state.editing_ev_id:
        # Inline Edit Panel
        ev_to_edit = next((e for e in st.session_state.manual_events if e['id'] == st.session_state.editing_ev_id), None)
        if ev_to_edit:
            with st.container(border=True):
                st.write(f"**Editando:** {ev_to_edit['tag']}")
                c1, c2, c3 = st.columns(3)
                ns = c1.number_input("Inicio (s)", value=float(ev_to_edit['start']), step=1.0)
                ne = c2.number_input("Fin (s)", value=float(ev_to_edit['end']), step=1.0)
                nn = c3.text_input("Nota", value=ev_to_edit['note'] or "")
                
                c_btn1, c_btn2 = st.columns(2)
                if c_btn1.button("Guardar Cambios", type="primary"):
                    ev_to_edit['start'] = ns
                    ev_to_edit['end'] = ne
                    ev_to_edit['note'] = nn
                    save_persistence()
                    st.session_state.editing_ev_id = None
                    st.rerun()
                if c_btn2.button("Cancelar"):
                    st.session_state.editing_ev_id = None
                    st.rerun()

    if not st.session_state.manual_events:
        st.info("No hay eventos en el timeline. Usa la barra lateral para crear eventos.")
    else:
        for ev in reversed(st.session_state.manual_events):
            with st.container():
                st.markdown(f"""
                <div class="timeline-card">
                    <div class="timeline-info">
                        <div>
                            <span class="tag-badge">{ev['tag']}</span> 
                            <span class="time-badge ml-2">[{ev['start']:.1f}s - {ev['end']:.1f}s]</span>
                        </div>
                        <div style="font-size:13px; color:#a2b9ce; margin-top:5px;">
                            {ev.get('note', '') or 'Sin notas'}
                            { ' 🎨 (Anotación/Dibujo aplicado)' if ev.get('has_drawing') else ''}
                            { ' 🎞️ (Clip MP4)' if ev.get('clip_path') else ''}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                t1, t2, t3, t4, t5 = st.columns(5)
                if t1.button("✏️ Editar", key=f"edit_{ev['id']}", use_container_width=True):
                    st.session_state.editing_ev_id = ev['id']
                    st.rerun()
                if t2.button("🗑️ Borrar", key=f"del_{ev['id']}", use_container_width=True):
                    st.session_state.manual_events = [e for e in st.session_state.manual_events if e['id'] != ev['id']]
                    save_persistence()
                    st.rerun()
                if t3.button("📹 Recortar MP4", key=f"clip_{ev['id']}", use_container_width=True):
                    with st.spinner("⏳ Cortando vídeo..."):
                        ok, res = _cortar_clip_manual(st.session_state.video_path, ev['start'], ev['end'], ev['id'])
                        if ok:
                            ev['clip_path'] = res
                            save_persistence()
                            st.success(f"Clip listo.")
                            st.rerun()
                        else:
                            st.error(f"Error: {res}")
                if t4.button("✍️ KlipDraw", key=f"draw_{ev['id']}", use_container_width=True):
                    if not ev.get('clip_path') or not Path(ev['clip_path']).exists():
                        st.warning("Debes 'Recortar MP4' primero para poder dibujar sobre el clip.")
                    else:
                        st.session_state.drawing_ev_id = ev['id']
                        st.session_state.scout_step = "drawing"
                        st.rerun()
                if t5.button("👁️ Cabezal a Inicio", key=f"pre_{ev['id']}", use_container_width=True):
                    st.session_state.current_time = float(ev['start'])
                    st.rerun()


def view_drawing():
    ev_id = st.session_state.drawing_ev_id
    ev = next((e for e in st.session_state.manual_events if e['id'] == ev_id), None)
    
    if not ev or not ev.get('clip_path'):
        st.error("Clip no válido.")
        if st.button("Volver al Editor"):
            st.session_state.scout_step = "editor"
            st.rerun()
        return

    clip_path = ev['clip_path']
    st.markdown('<div class="analyst-header"><h1>Modo Analista - Pizarra KlipDraw</h1><p>Dibuja anotaciones sobre el clip. Estas se fusionarán con el vídeo.</p></div>', unsafe_allow_html=True)
    
    if st.button("🔙 Cancelar y Volver al Editor"):
        st.session_state.scout_step = "editor"
        st.rerun()

    c1, c2 = st.columns([1, 2])
    with c1:
        st.write("### Opciones de Dibujo")
        stroke_width = st.slider("Grosor del Trazo", 1, 15, 3)
        stroke_color = st.color_picker("Color", "#E31212")
        drawing_mode = st.radio("Modo", ("freedraw", "line", "circle", "rect", "arrow"))
        st.caption("Al guardar, el dibujo se 'quemará' estáticamente en todo el clip.")
        
        if st.button("💾 Guardar y Fusionar (FFmpeg)", type="primary", use_container_width=True):
            canvas_data = st.session_state.get(f"canvas_data_{ev_id}")
            if canvas_data is not None and canvas_data.image_data is not None:
                # Extract image array and save as transparent PNG
                img_data = canvas_data.image_data
                img = Image.fromarray((img_data).astype(np.uint8))
                
                overlay_path = CLIPS_DIR / f"overlay_{ev_id}.png"
                img.save(overlay_path, "PNG")
                
                # Burn into video
                out_burned = CLIPS_DIR / f"analista_{ev_id}_burned.mp4"
                with st.spinner("⏳ Fusionando dibujo en el clip MP4..."):
                    ok, res = _burn_overlay(clip_path, str(overlay_path), str(out_burned))
                    if ok:
                        # Replace clip path with burned clip
                        if Path(clip_path).exists():
                            Path(clip_path).unlink(missing_ok=True) # delete old
                        ev['clip_path'] = res
                        ev['has_drawing'] = True
                        save_persistence()
                        st.success("¡Vídeo anotado guardado!")
                        st.session_state.scout_step = "editor"
                        time.sleep(1) # Visual feedback
                        st.rerun()
                    else:
                        st.error(f"Error fusionando: {res}")
            else:
                st.warning("El lienzo está vacío o no se ha modificado.")

    with c2:
        # Extract a middle frame to draw upon
        frame_path = CLIPS_DIR / f"frame_{ev_id}.jpg"
        if not frame_path.exists():
            duration = ev['end'] - ev['start']
            mid_time = duration / 2.0
            _extract_frame(clip_path, mid_time, str(frame_path))
            
        bg_image = Image.open(str(frame_path)) if frame_path.exists() else None
        
        # We need to scale canvas to fit layout but keep aspect ratio, a full 1080p canvas is too large for browser
        w, h = 800, 450
        if bg_image:
            w, h = bg_image.size
            # scale down if too big
            max_w = 800
            if w > max_w:
                ratio = max_w / float(w)
                h = int(float(h) * float(ratio))
                w = max_w
                bg_image = bg_image.resize((w, h))

        canvas = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_image=bg_image,
            update_streamlit=True,
            height=h,
            width=w,
            drawing_mode="line" if drawing_mode == "arrow" else drawing_mode, # custom arrow logic can be complex, fallback to line if needed, but streamlit-drawable-canvas doesn't strictly have 'arrow' out of box without extension, standard is line. Let's map arrow to line or custom plugin.
            key=f"canvas_data_{ev_id}",
        )


def render():
    init_session()
    
    if st.session_state.scout_step == "dashboard":
        view_dashboard()
    elif st.session_state.scout_step == "config":
        view_config()
    elif st.session_state.scout_step == "editor":
        view_editor()
    elif st.session_state.scout_step == "drawing":
        view_drawing()


if __name__ == "__main__":
    render()
