import streamlit as st

def get_nav_config():
    """Returns the sidebar navigation configuration."""
    return [
        {"key": "home", "icon": "🏠", "label": "Inicio", "group": "PRINCIPAL"},
        {"key": "datos_jugadores", "icon": "👥", "label": "Mi Equipo", "group": "EQUIPO"},
        {"key": "scout", "icon": "🎬", "label": "Modo Analista", "group": "EQUIPO"},
        {"key": "partido_nuevo", "icon": "🎬", "label": "Nuevo Análisis", "group": "PARTIDOS"},
        {"key": "partido_clips", "icon": "✂️", "label": "Clips de Acción", "group": "PARTIDOS"},
        {"key": "calibracion", "icon": "📍", "label": "Calibración", "group": "PARTIDOS"},
        {"key": "datos_colectivos", "icon": "📊", "label": "Datos Colectivos", "group": "DATOS"},
        {"key": "equipo_config", "icon": "⚙️", "label": "Configuración", "group": "SISTEMA"},
    ]

def render_sidebar():
    """Renders the professional sidebar with groups and branding."""
    from modules.utils import get_logo_base64
    
    logo_b64 = get_logo_base64()
    logo_html = (f'<img src="data:image/png;base64,{logo_b64}" style="height:30px;border-radius:6px;">'
                 if logo_b64 else '<span style="font-size:22px;">⚽</span>')

    st.sidebar.markdown(f"""
    <div class="sidebar-logo">
        <div style="margin-top: -10px;">{logo_html}</div>
        <div>
            <div class="sidebar-logo-text">ED Analytics</div>
            <div class="sidebar-logo-sub">Scout Platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav_items = get_nav_config()
    groups = ["PRINCIPAL", "EQUIPO", "PARTIDOS", "DATOS", "SISTEMA"]

    for group in groups:
        st.sidebar.markdown(f'<div class="nav-group-label">{group}</div>', unsafe_allow_html=True)
        for item in [i for i in nav_items if i["group"] == group]:
            if st.sidebar.button(f"{item['icon']} {item['label']}", key=f"nav_{item['key']}", use_container_width=True):
                st.session_state["page"] = item["key"]
                if item["key"] == "scout":
                    st.session_state["scout_step"] = "dashboard"
                st.rerun()

    # Active Match Card
    if st.session_state.get("analysis_config"):
        cfg = st.session_state["analysis_config"]
        st.sidebar.markdown(f"""
        <div class="sidebar-match-card">
            <div class="sidebar-match-title">Partido activo</div>
            <div class="sidebar-match-teams">{cfg.get('team','')} <span style="color:#5a6a7e;font-weight:400;">vs</span> {cfg.get('rival','')}</div>
            <div class="sidebar-match-meta">{cfg.get('match_date','')} · {cfg.get('competition','')}</div>
        </div>
        """, unsafe_allow_html=True)
