import streamlit as st
import base64
from pathlib import Path
from modules.themes import CSS_IMPECCABLE, CSS_WYSCOUT

st.set_page_config(
    page_title="ED Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS GLOBAL DINAMICO ───────────────────────────────────────────────────────
selected_theme = st.session_state.get("ui_theme", "Estilo Impeccable")
if selected_theme == "Estilo Impeccable":
    st.markdown(CSS_IMPECCABLE, unsafe_allow_html=True)
else:
    st.markdown(CSS_WYSCOUT, unsafe_allow_html=True)


def get_logo_base64():
    base_dir = Path(__file__).parent
    logo_path = base_dir / "logo.png"
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


# ══════════════════════════════════════════════════════════════════════════════
# LOGIN
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.get("logged_in"):
    st.markdown("<style>[data-testid='stSidebar']{display:none!important;}</style>", unsafe_allow_html=True)
    logo_b64 = get_logo_base64()

    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        st.markdown("<br><br>", unsafe_allow_html=True)

        if logo_b64:
            st.markdown(
                f'<div style="text-align:center;margin-bottom:10px;">'
                f'<img src="data:image/png;base64,{logo_b64}" style="height:72px;border-radius:12px;"></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown('<div style="text-align:center;font-size:56px;margin-bottom:10px;">⚽</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align:center;margin-bottom:6px;">
            <div style="font-size:26px;font-weight:800;color:#ffffff;letter-spacing:-0.5px;">ED Analytics</div>
            <div style="font-size:13px;color:#5a6a7e;margin-top:4px;">Plataforma de análisis de fútbol profesional</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr style='border-top:1px solid #1e2a3a;margin:26px 0 18px;'>", unsafe_allow_html=True)

        st.markdown('<span style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:#5a6a7e;">Usuario</span>', unsafe_allow_html=True)
        usuario = st.text_input("Usuario", placeholder="Introduce tu usuario", label_visibility="collapsed")
        st.markdown('<span style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:#5a6a7e;margin-top:12px;display:block;">Contraseña</span>', unsafe_allow_html=True)
        password = st.text_input("Contraseña", type="password", placeholder="••••••••", label_visibility="collapsed")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Entrar →", use_container_width=True, type="primary"):
            st.session_state["logged_in"] = True
            st.session_state["usuario"] = usuario.strip() or "Scout"
            st.rerun()

        st.markdown('<div style="text-align:center;margin-top:20px;font-size:11px;color:#3a4a5e;">© 2025 ED Analytics · Todos los derechos reservados</div>', unsafe_allow_html=True)
    st.stop()


from modules.navigation import render_sidebar
from modules.utils import get_logo_base64

# ... (Previous code remains up to get_logo_base64 which is now imported)

# ... (Previous LOGIN logic remains unchanged for now to preserve session context)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    render_sidebar()
    
    st.markdown('<div class="nav-group-label" style="text-align: center; margin-top: 10px;">🎨 TEMA VISUAL</div>', unsafe_allow_html=True)
    theme_choice = st.selectbox(
        "Tema UI", 
        ["Estilo Impeccable", "Estilo Neon"], 
        index=0 if st.session_state.get("ui_theme", "Estilo Impeccable") == "Estilo Impeccable" else 1,
        label_visibility="collapsed"
    )
    if theme_choice != st.session_state.get("ui_theme", "Estilo Impeccable"):
        st.session_state["ui_theme"] = theme_choice
        st.rerun()
        
    st.markdown("<br>", unsafe_allow_html=True)
    if st.sidebar.button("↩  Cerrar sesión", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# ROUTING
# ══════════════════════════════════════════════════════════════════════════════
PAGES = {
    "home": "pages.home",
    "equipo_config": "pages.settings",
    "partido_nuevo": "pages.upload_analyze",
    "partido_clips": "pages.action_clips",
    "datos_jugadores": "pages.squad_players",
    "datos_colectivos": "pages.collective_dashboard",
    "scout": "pages.scout",
    "calibracion": "pages.calibration",
    "player_tracking": "pages.player_tracking"
}

current_page = st.session_state.get("page", "home")
# Map legacy routes
if current_page in ("colectivo_mapa", "colectivo_stats", "colectivo_metricas"):
    current_page = "datos_colectivos"

if current_page in PAGES:
    module = __import__(PAGES[current_page], fromlist=["render"])
    module.render()
else:
    from pages.upload_analyze import render
    render()
