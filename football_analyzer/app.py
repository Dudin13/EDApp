import streamlit as st
import base64
from pathlib import Path

st.set_page_config(
    page_title="ED Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_logo_base64():
    logo_path = Path("C:/apped/football_analyzer/logo.png")
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

logo_b64 = get_logo_base64()
logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="height:60px; margin-right:20px; vertical-align:middle;">' if logo_b64 else "⚽"

processing = st.session_state.get("processing", False)
ball_style = "display:block;" if processing else "display:none;"

st.markdown(f"""
<style>
    .main-header {{
        background: linear-gradient(135deg, #1a1a2e 0%, #0d3060 100%);
        padding: 20px 30px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        margin-bottom: 25px;
        border-left: 5px solid #FFD700;
    }}
    .main-header-text h1 {{
        font-size: 2.2em;
        font-weight: 800;
        margin: 0;
        color: #FFD700;
        letter-spacing: 2px;
    }}
    .main-header-text p {{
        font-size: 0.95em;
        margin: 3px 0 0 0;
        color: #cccccc;
    }}
    .ball-anim {{
        position: fixed;
        top: 18px;
        right: 70px;
        z-index: 9999;
        font-size: 2em;
        animation: bounce 0.5s infinite alternate;
        {ball_style}
    }}
    @keyframes bounce {{
        from {{ transform: translateY(0px) rotate(0deg); }}
        to   {{ transform: translateY(-18px) rotate(30deg); }}
    }}
    [data-testid="stSidebar"] {{
        background: #1a1a2e;
    }}
    [data-testid="stSidebar"] .stRadio label {{
        color: white !important;
        font-size: 0.95em;
    }}
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: #FFD700 !important;
    }}
    .stMetric {{
        background: #1a1a2e;
        border-left: 3px solid #FFD700;
        padding: 10px;
        border-radius: 8px;
    }}
</style>

<div class="ball-anim">⚽</div>

<div class="main-header">
    {logo_html}
    <div class="main-header-text">
        <h1>ED Analytics</h1>
        <p>Sistema de análisis automático para seguimiento de cedidos</p>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ED Analytics")
    st.markdown("---")
    page = st.radio("Sección", [
        "🎬 Subir y Analizar Vídeo",
        "👤 Seguimiento de Jugador",
        "📊 Métricas del Partido",
        "🗺️ Mapa Táctico",
        "✂️ Clips por Acción",
        "👥 Análisis Colectivo",
        "⚙️ Configuración"
    ])
    st.markdown("---")
    if st.session_state.get("analysis_config"):
        config = st.session_state["analysis_config"]
        st.markdown(f"**Partido activo:**")
        st.markdown(f"👤 {config.get('player_name','')}")
        st.markdown(f"🏟️ {config.get('team','')} vs {config.get('rival','')}")
        st.markdown(f"📅 {config.get('match_date','')}")

if page == "🎬 Subir y Analizar Vídeo":
    from pages.upload_analyze import render
    render()
elif page == "👤 Seguimiento de Jugador":
    from pages.player_tracking import render
    render()
elif page == "📊 Métricas del Partido":
    from pages.match_metrics import render
    render()
elif page == "🗺️ Mapa Táctico":
    from pages.tactical_map import render
    render()
elif page == "✂️ Clips por Acción":
    from pages.action_clips import render
    render()
elif page == "👥 Análisis Colectivo":
    from pages.collective_analysis import render
    render()
elif page == "⚙️ Configuración":
    from pages.settings import render
    render()