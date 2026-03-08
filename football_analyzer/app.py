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
    st.markdown(CSS_WYSCOUT, unsafe_allow_html=True), unsafe_allow_html=True)


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


# ══════════════════════════════════════════════════════════════════════════════
# NAVEGACIÓN — definición de estructura
# ══════════════════════════════════════════════════════════════════════════════
# page_key es el identificador interno que usamos para el routing
NAV_TOP = [
    {"key": "datos_jugadores",  "icon": "🏆", "label": "Mi Equipo",      "group": "EQUIPO"},
    {"key": "equipo_config",  "icon": "⚙️", "label": "Configuración",      "group": "EQUIPO"},
    {"key": "partido_nuevo",  "icon": "🎬", "label": "Nuevo Análisis",  "group": "PARTIDOS"},
    {"key": "partido_clips",  "icon": "✂️",  "label": "Clips de Acción","group": "PARTIDOS"},
]

# Sub-items bajo el grupo "DATOS"
DATO_JUGADORES = {"key": "datos_jugadores",    "icon": "👤", "label": "Datos Jugadores"}
DATO_COLECTIVO_ITEMS = [
    {"key": "colectivo_mapa",     "icon": "🗻️", "label": "Mapa Táctico"},
    {"key": "colectivo_stats",    "icon": "👥", "label": "Análisis Colectivo"},
    {"key": "colectivo_metricas", "icon": "📊", "label": "Métricas de Partido"},
]

# Estado de navegación
if "page" not in st.session_state:
    st.session_state["page"] = "home"

current_page = st.session_state["page"]


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    logo_b64 = get_logo_base64()
    logo_html = (f'<img src="data:image/png;base64,{logo_b64}" style="height:30px;border-radius:6px;">'
                 if logo_b64 else '<span style="font-size:22px;">⚽</span>')

    st.markdown(f"""
    <div class="sidebar-logo">
        <div style="margin-top: -10px;">{logo_html}</div>
        <div>
            <div class="sidebar-logo-text">ED Analytics</div>
            <div class="sidebar-logo-sub">Scout Platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-group-label">PRINCIPAL</div>', unsafe_allow_html=True)
    if st.button("🏠", key="nav_home", use_container_width=True):
        st.session_state["page"] = "home"
        st.rerun()

    # ── EQUIPO ──
    st.markdown('<div class="nav-group-label">EQUIPO</div>', unsafe_allow_html=True)
    if st.button("Mi Equipo", key="nav_equipo", use_container_width=True):
        st.session_state["page"] = "datos_jugadores"
        st.rerun()
    
    # Nuevo botón Modo Analista bajo Mi Equipo
    if st.button("Modo Analista", key="nav_analista", use_container_width=True):
        st.session_state["page"] = "scout"
        st.session_state["scout_step"] = "dashboard"
        st.rerun()

    # ── PARTIDOS ──
    st.markdown('<div class="nav-group-label">PARTIDOS</div>', unsafe_allow_html=True)
    if st.button("Nuevo Análisis", key="nav_partido_nuevo", use_container_width=True):
        st.session_state["page"] = "partido_nuevo"
        st.rerun()
    if st.button("Clips de Acción", key="nav_partido_clips", use_container_width=True):
        st.session_state["page"] = "partido_clips"
        st.rerun()
    if st.button("Calibración 📍", key="nav_calibracion", use_container_width=True):
        st.session_state["page"] = "calibracion"
        st.rerun()

    # ── DATOS (dos sub-items directos) ──
    st.markdown('<div class="nav-group-label">DATOS</div>', unsafe_allow_html=True)

    if st.button("Datos Jugadores", key="nav_datos_jugadores", use_container_width=True):
        st.session_state["page"] = "datos_jugadores"
        st.session_state["squad_selected_player"] = None
        st.rerun()

    if st.button("Datos Colectivos", key="nav_datos_colectivos", use_container_width=True):
        st.session_state["page"] = "datos_colectivos"
        st.rerun()

    # Partido activo
    if st.session_state.get("analysis_config"):
        cfg = st.session_state["analysis_config"]
        st.markdown(f"""
        <div class="sidebar-match-card">
            <div class="sidebar-match-title">Partido activo</div>
            <div class="sidebar-match-teams">{cfg.get('team','')} <span style="color:#5a6a7e;font-weight:400;">vs</span> {cfg.get('rival','')}</div>
            <div class="sidebar-match-meta">{cfg.get('match_date','')} · {cfg.get('competition','')}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("Configuración", key="nav_config", use_container_width=True):
        st.session_state["page"] = "equipo_config"
        st.rerun()
    
    st.markdown('<div class="nav-group-label" style="text-align: center; margin-top: 10px;">🎨 TEMA VISUAL</div>', unsafe_allow_html=True)
    theme_choice = st.selectbox(
        "Tema UI", 
        ["Estilo Impeccable", "Estilo Wyscout Neon"], 
        index=0 if st.session_state.get("ui_theme", "Estilo Impeccable") == "Estilo Impeccable" else 1,
        label_visibility="collapsed"
    )
    if theme_choice != st.session_state.get("ui_theme", "Estilo Impeccable"):
        st.session_state["ui_theme"] = theme_choice
        st.rerun()
        
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("↩  Cerrar sesión", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ROUTING
# ══════════════════════════════════════════════════════════════════════════════
page = st.session_state.get("page", "home")

if page == "home":
    from pages.home import render
    render()

elif page == "equipo_config":
    from pages.settings import render
    render()

elif page == "partido_nuevo":
    from pages.upload_analyze import render
    render()

elif page == "partido_clips":
    from pages.action_clips import render
    render()

elif page == "datos_jugadores":
    from pages.squad_players import render
    render()

elif page == "datos_colectivos":
    from pages.collective_dashboard import render
    render()

elif page == "scout":
    from pages.scout import render
    render()

elif page == "calibracion":
    from pages.calibration import render
    render()

# Compatibilidad con rutas antiguas
elif page in ("colectivo_mapa", "colectivo_stats", "colectivo_metricas"):
    from pages.collective_dashboard import render
    render()

elif page == "jugador_perfil":
    from pages.player_tracking import render
    render()

else:
    from pages.upload_analyze import render
    render()