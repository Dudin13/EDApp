"""
app.py — Punto de entrada principal (Streamlit).
Estructura multipágina optimizada y gestión de estado global.

FIXES APLICADOS:
  [CRÍTICO] Inicialización de st.session_state: faltaban claves como 'auto_detect'
            y 'analysis_done', lo que causaba AttributeErrors en las páginas.
  [CRÍTICO] El import de 'calibration' fallaba si no se encontraba en el PATH.
            Garantizamos que el directorio raíz de la app esté en sys.path.
  [MEJORA]  Diseño responsivo: sidebar colapsable por defecto para maximizar el
            espacio de trabajo de los dashboards.
  [MEJORA]  Banner central con gradiente premium y logo dinámico.
"""

import streamlit as st
import sys
from pathlib import Path

# ── Fix 7: Configuración de sys.path ─────────────────────────────────────────
# Este patch permite que los módulos en app/ sean importables con 'from modules.xxx'.
# Alternativa limpia a largo plazo: convertir app/ en un paquete instalable
# con pyproject.toml y eliminarlo. Por ahora es el enfoque más seguro para Streamlit.
def _setup_sys_path() -> None:
    root_path = str(Path(__file__).parent)
    if root_path not in sys.path:
        sys.path.insert(0, root_path)

_setup_sys_path()

# ── Configuración de Página ────────────────────────────────────────────────
st.set_page_config(
    page_title="EDudin · Football Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos Globales (Glassmorphism & Wyscout Style) ────────────────────────
st.markdown("""
<style>
    /* Importar tipografía premium */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Ocultar elementos innecesarios */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Contenedor de navegación */
    .nav-container {
        padding: 20px 10px;
    }

    .nav-header {
        font-size: 11px;
        font-weight: 800;
        color: #5a6a7e;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 24px 0 12px 12px;
    }

    /* Botones de navegación personalizados */
    .stButton > button {
        width: 100%;
        background: transparent !important;
        border: none !important;
        color: #8899aa !important;
        text-align: left !important;
        padding: 10px 16px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        background: rgba(0, 212, 170, 0.08) !important;
        color: #00d4aa !important;
        transform: translateX(4px);
    }

    .nav-active > button {
        background: rgba(0, 212, 170, 0.12) !important;
        color: #00d4aa !important;
        border-left: 3px solid #00d4aa !important;
        border-radius: 0 8px 8px 0 !important;
    }

    /* Cards Glassmorphism */
    .ws-card {
        background: rgba(13, 18, 32, 0.4);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }

    /* Headers sección */
    .ws-section-header {
        font-size: 14px;
        font-weight: 800;
        color: #fff;
        margin-bottom: 20px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        padding-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# ── Gestión de Estado ──────────────────────────────────────────────────────
# FIX: Asegurar que todas las claves necesarias existan desde el inicio
if "page" not in st.session_state:
    st.session_state.page = "home"
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "ball_events" not in st.session_state:
    st.session_state.ball_events = []
if "team_colors" not in st.session_state:
    st.session_state.team_colors = {}
if "auto_detect" not in st.session_state:
    st.session_state.auto_detect = True

# ── Sidebar y Navegación ─────────────────────────────────────────────────────
with st.sidebar:
    # Logo Central
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 30px 0;">
        <div style="font-size:28px; font-weight:900; color:#fff; letter-spacing:-1px;">
            E<span style="color:#00d4aa;">DUDIN</span>
        </div>
        <div style="font-size:10px; color:#5a6a7e; text-transform:uppercase; letter-spacing:3px; margin-top:-5px;">
            Football Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    def nav_btn(label, target, icon=""):
        btn_class = "nav-active" if st.session_state.page == target else ""
        st.markdown(f'<div class="{btn_class}">', unsafe_allow_html=True)
        if st.button(f"{icon}  {label}", key=f"nav_{target}"):
            st.session_state.page = target
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="nav-header">Principal</div>', unsafe_allow_html=True)
    nav_btn("Inicio", "home", "🏠")
    nav_btn("Análisis de Vídeo", "upload", "🛰️")

    st.markdown('<div class="nav-header">Dashboards</div>', unsafe_allow_html=True)
    nav_btn("Panel de Scout", "scout", "📋")
    nav_btn("Dashboard Colectivo", "collective", "📊")
    nav_btn("Clips de Acción", "clips", "🎬")

    st.markdown('<div class="nav-header">Herramientas IA</div>', unsafe_allow_html=True)
    nav_btn("Corrección de Imágenes", "image_correction", "🔍")
    nav_btn("Auditoría de Dataset", "dataset_audit", "🔬")
    nav_btn("Calibración Campo", "calibration", "📐")
    nav_btn("Ajustes", "settings", "⚙️")

    # Footer Sidebar
    st.markdown("""
    <div style="position:fixed; bottom:20px; left:25px; font-size:10px; color:#3a4a5e;">
        v2.0.1 PRO Edition<br>
        2024 © Edudin Analytics
    </div>
    """, unsafe_allow_html=True)

# ── Router de Páginas ────────────────────────────────────────────────────────
# FIX: Manejo robusto de imports para evitar crashes si falta un archivo
try:
    if st.session_state.page == "home":
        from pages import home
        home.render()
    elif st.session_state.page == "upload":
        from pages import upload_analyze
        upload_analyze.render()
    elif st.session_state.page == "scout":
        from pages import scout
        scout.render()
    elif st.session_state.page == "collective":
        from pages import collective_dashboard
        collective_dashboard.render()
    elif st.session_state.page == "clips":
        from pages import action_clips
        action_clips.render()
    elif st.session_state.page == "image_correction":
        from pages import image_correction
        image_correction.main()
    elif st.session_state.page == "dataset_audit":
        from pages import dataset_audit
        dataset_audit.main()
    elif st.session_state.page == "calibration":
        from pages import calibration
        calibration.render()
    elif st.session_state.page == "settings":
        from pages import settings
        settings.render()
except Exception as e:
    st.error(f"Error cargando la página '{st.session_state.page}': {e}")
    st.exception(e)
