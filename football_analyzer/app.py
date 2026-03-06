import streamlit as st
import base64
from pathlib import Path

st.set_page_config(
    page_title="ED Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS GLOBAL WYSCOUT ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: #0b0f1a; }
.main .block-container { padding: 1.5rem 2rem 2rem; max-width: 1400px; }

/* ── Ocultar la navegación automática de Streamlit (pages/) ── */
[data-testid="stSidebarNav"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #0d1220 !important;
    border-right: 1px solid #1e2a3a;
    min-width: 230px !important; max-width: 230px !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0.75rem 0.75rem; }

.sidebar-logo {
    display: flex; align-items: center; gap: 10px;
    padding: 12px 6px 16px; border-bottom: 1px solid #1e2a3a; margin-bottom: 6px;
}
.sidebar-logo-text { font-size: 16px; font-weight: 700; color: #fff; letter-spacing: 0.5px; }
.sidebar-logo-sub { font-size: 10px; color: #5a6a7e; font-weight: 400; text-transform: uppercase; letter-spacing: 1px; }

/* User row */
.sidebar-user {
    display: flex; align-items: center; gap: 8px;
    padding: 10px 6px 14px; border-bottom: 1px solid #1e2a3a; margin-bottom: 4px;
}
.sidebar-avatar {
    width: 28px; height: 28px;
    background: linear-gradient(135deg,#00d4aa,#0077ff);
    border-radius: 50%; display: flex; align-items: center; justify-content:center;
    font-size: 12px; font-weight: 700; color: #000; flex-shrink: 0;
}
.sidebar-user-name { font-size: 12px; font-weight: 600; color: #fff; }
.sidebar-user-role { font-size: 10px; color: #5a6a7e; }

/* Section group header */
.nav-group-label {
    font-size: 9px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.8px; color: #3a4a5e; padding: 14px 8px 4px;
}

/* Nav item buttons */
.nav-item {
    display: flex; align-items: center; gap: 9px;
    padding: 9px 10px; border-radius: 8px; cursor: pointer;
    color: #8899aa; font-size: 13px; font-weight: 500;
    transition: all 0.15s; border: 1px solid transparent; margin-bottom: 2px;
    text-decoration: none;
}
.nav-item:hover { background: rgba(0,212,170,0.07); color: #fff; border-color: rgba(0,212,170,0.12); }
.nav-item.active { background: rgba(0,212,170,0.12); color: #00d4aa !important; border-color: rgba(0,212,170,0.22); }
.nav-item .nav-icon { font-size: 14px; width: 18px; text-align: center; flex-shrink:0; }
.nav-item .nav-sub { font-size: 10px; color: #5a6a7e; margin-top:1px; }

/* Sub-items indented */
.nav-subitem {
    display: flex; align-items: center; gap: 9px;
    padding: 7px 10px 7px 34px; border-radius: 8px; cursor: pointer;
    color: #6a7a8e; font-size: 12px; font-weight: 500;
    transition: all 0.15s; border: 1px solid transparent; margin-bottom: 1px;
}
.nav-subitem:hover { background: rgba(0,212,170,0.05); color: #ccc; }
.nav-subitem.active { color: #00d4aa !important; background: rgba(0,212,170,0.08); }

/* Match badge */
.sidebar-match-card {
    background: #111827; border: 1px solid #1e2a3a; border-radius: 10px;
    padding: 10px 12px; margin: 8px 4px 0;
}
.sidebar-match-title { font-size: 9px; text-transform: uppercase; letter-spacing: 1.5px; color: #3a4a5e; margin-bottom: 6px; font-weight: 700; }
.sidebar-match-teams { font-size: 13px; font-weight: 700; color: #fff; margin-bottom: 2px; }
.sidebar-match-meta { font-size: 10px; color: #5a6a7e; }

/* ── METRICS ── */
[data-testid="metric-container"] {
    background: rgba(17, 24, 39, 0.4) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(0, 212, 170, 0.2) !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    transition: all 0.3s ease;
}
[data-testid="metric-container"]:hover {
    border-color: rgba(0, 212, 170, 0.6) !important;
    box-shadow: 0 0 20px rgba(0, 212, 170, 0.2) !important;
    transform: translateY(-2px);
}
[data-testid="stMetricLabel"] { font-size: 11px !important; text-transform: uppercase !important; letter-spacing: 1.2px !important; color: #8899aa !important; font-weight: 700 !important; }
[data-testid="stMetricValue"] { font-size: 28px !important; font-weight: 800 !important; color: #fff !important; text-shadow: 0 0 10px rgba(0, 212, 170, 0.3); }

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, #00d4aa 0%, #0077ff 100%) !important;
    color: #000 !important; border: none !important; border-radius: 8px !important;
    font-weight: 700 !important; font-size: 13px !important; padding: 10px 24px !important;
    transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.stButton > button:hover {
    opacity: 1 !important;
    transform: scale(1.02) !important;
    box-shadow: 0 0 25px rgba(0, 212, 170, 0.5) !important;
}
.stButton > button[kind="secondary"] {
    background: rgba(30, 42, 58, 0.3) !important;
    backdrop-filter: blur(8px);
    color: #e8eaed !important;
    border: 1px solid rgba(0, 212, 170, 0.3) !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] { background: transparent; border-bottom: 2px solid rgba(30, 42, 58, 0.5); gap: 10px; }
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #5a6a7e; font-size: 14px; font-weight: 600;
    padding: 12px 24px; border-bottom: 3px solid transparent; border-radius: 8px 8px 0 0;
}
.stTabs [aria-selected="true"] {
    background: rgba(0, 212, 170, 0.05) !important;
    color: #00d4aa !important;
    border-bottom: 3px solid #00d4aa !important;
}

/* ── INPUTS ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div,
.stTextInput > div > div > input, .stNumberInput > div > div > input {
    background: rgba(17, 24, 39, 0.6) !important;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(30, 42, 58, 0.8) !important;
    border-radius: 10px !important;
    color: #e8eaed !important;
    padding: 8px 12px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #00d4aa !important;
    box-shadow: 0 0 10px rgba(0, 212, 170, 0.2) !important;
}

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    background: rgba(17, 24, 39, 0.4) !important;
    backdrop-filter: blur(15px) !important;
    border: 1px solid rgba(30, 42, 58, 0.6) !important;
    border-radius: 14px !important;
}

/* ── REUSABLE COMPONENTS ── */
.ws-player-header {
    background: rgba(17, 24, 39, 0.5);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(0, 212, 170, 0.2);
    border-left: 5px solid #00d4aa;
    border-radius: 16px; padding: 24px 32px; display: flex;
    align-items: center; gap: 24px; margin-bottom: 30px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
}
.ws-player-number { font-size: 56px; font-weight: 900; color: #00d4aa; line-height: 1; min-width: 70px; text-shadow: 0 0 15px rgba(0, 212, 170, 0.4); }
.ws-player-name { font-size: 24px; font-weight: 800; color: #fff; margin-bottom: 6px; }
.ws-player-meta { font-size: 14px; color: #8899aa; }

.ws-section-header {
    font-size: 12px; font-weight: 800; text-transform: uppercase;
    letter-spacing: 2px; color: #00d4aa; margin: 30px 0 15px;
    display: flex; align-items: center; gap: 12px;
}
.ws-section-header::after { content: ''; flex: 1; height: 2px; background: linear-gradient(90deg, rgba(0, 212, 170, 0.3), transparent); }

/* ── LOGIN ── */
</style>
""", unsafe_allow_html=True)


def get_logo_base64():
    logo_path = Path("C:/apped/football_analyzer/logo.png")
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
    {"key": "equipo_config",  "icon": "🏆", "label": "Mi Equipo",      "group": "EQUIPO"},
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
    st.session_state["page"] = "partido_nuevo"

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
        {logo_html}
        <div>
            <div class="sidebar-logo-text">ED Analytics</div>
            <div class="sidebar-logo-sub">Scout Platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── EQUIPO ──
    st.markdown('<div class="nav-group-label">EQUIPO</div>', unsafe_allow_html=True)
    if st.button("Mi Equipo", key="nav_equipo", use_container_width=True):
        st.session_state["page"] = "equipo_config"
        st.rerun()
    
    # Nuevo botón Modo Analista bajo Mi Equipo
    if st.button("Modo Analista", key="nav_analista", use_container_width=True):
        st.session_state["page"] = "scout"
        st.rerun()

    # ── PARTIDOS ──
    st.markdown('<div class="nav-group-label">PARTIDOS</div>', unsafe_allow_html=True)
    if st.button("Nuevo Análisis", key="nav_partido_nuevo", use_container_width=True):
        st.session_state["page"] = "partido_nuevo"
        st.rerun()
    if st.button("Clips de Acción", key="nav_partido_clips", use_container_width=True):
        st.session_state["page"] = "partido_clips"
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

    if st.session_state.get("analysis_done"):
        st.markdown("""
        <div style="display:flex;align-items:center;gap:6px;padding:10px 6px 0;">
            <div style="width:7px;height:7px;background:#00d4aa;border-radius:50%;box-shadow:0 0 6px #00d4aa;"></div>
            <span style="font-size:11px;color:#00d4aa;font-weight:600;">Análisis completado</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("↩  Cerrar sesión", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ROUTING
# ══════════════════════════════════════════════════════════════════════════════
page = st.session_state.get("page", "partido_nuevo")

if page == "equipo_config":
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