CSS_WYSCOUT = """
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
}
[data-testid="stSidebar"] > div:first-child { 
    padding-top: 0rem !important; 
}
[data-testid="stSidebarUserContent"] {
    padding-top: 0rem !important;
}

.sidebar-logo {
    display: flex; flex-direction: column; align-items: flex-start;
    padding: 0 0 16px 0; margin-bottom: 16px; border-bottom: 1px solid #1e2a3a;
    margin-top: -30px;
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
.stSelectbox div[data-baseweb="select"] span,
.stSelectbox div[data-baseweb="select"] div {
    color: #e8eaed !important;
}
div[data-baseweb="popover"] ul {
    background: #111827 !important;
}
div[data-baseweb="popover"] li, div[data-baseweb="popover"] span, div[data-baseweb="popover"] div {
    color: #e8eaed !important;
}
.stTextInput > div > div > input, .stNumberInput > div > div > input {
    background: rgba(17, 24, 39, 0.6) !important;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(30, 42, 58, 0.8) !important;
    border-radius: 10px !important;
    color: #e8eaed !important;
    padding: 8px 12px !important;
}
[data-testid="stSelectbox"] > div > div, [data-testid="stMultiSelect"] > div > div {
    background: rgba(17, 24, 39, 0.6) !important;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(30, 42, 58, 0.8) !important;
    border-radius: 10px !important;
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
</style>
"""

CSS_IMPECCABLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: #09090b; color: #fafafa; }
.main .block-container { padding: 2.5rem 2rem 2rem; max-width: 1400px; }

/* ── Typography & General Polish ── */
h1, h2, h3, h4, h5, h6 { font-weight: 600 !important; tracking: -0.02em; }
p, span, div { -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }

/* ── Hide default Streamlit Nav ── */
[data-testid="stSidebarNav"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #09090b !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0rem !important; }
[data-testid="stSidebarUserContent"] { padding-top: 0rem !important; }

.sidebar-logo {
    display: flex; flex-direction: column; align-items: flex-start;
    padding: 0 0 16px 0; margin-bottom: 24px; border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    margin-top: -20px;
}
.sidebar-logo-text { font-size: 16px; font-weight: 600; color: #fafafa; letter-spacing: -0.02em; }
.sidebar-logo-sub { font-size: 11px; color: #a1a1aa; font-weight: 500; }

/* Section group header */
.nav-group-label {
    font-size: 10px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.1em; color: #52525b; padding: 20px 8px 8px;
}

/* Nav item buttons */
.nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 10px; border-radius: 6px; cursor: pointer;
    color: #a1a1aa; font-size: 13px; font-weight: 500;
    transition: all 0.2s ease; border: 1px solid transparent; margin-bottom: 2px;
    text-decoration: none;
}
.nav-item:hover { background: rgba(255, 255, 255, 0.04); color: #fafafa; }
.nav-item.active { background: rgba(255, 255, 255, 0.08); color: #fafafa !important; border-color: rgba(255, 255, 255, 0.04); }
.nav-item .nav-icon { font-size: 14px; width: 20px; text-align: center; flex-shrink:0; opacity: 0.8; }
.nav-item .nav-sub { font-size: 11px; color: #71717a; margin-top:2px; }

/* Sub-items indented */
.nav-subitem {
    display: flex; align-items: center; gap: 10px;
    padding: 7px 10px 7px 36px; border-radius: 6px; cursor: pointer;
    color: #71717a; font-size: 12px; font-weight: 500;
    transition: all 0.2s ease; border: 1px solid transparent; margin-bottom: 2px;
}
.nav-subitem:hover { background: rgba(255, 255, 255, 0.03); color: #e4e4e7; }
.nav-subitem.active { color: #fafafa !important; background: rgba(255, 255, 255, 0.05); }

/* Match badge */
.sidebar-match-card {
    background: #09090b; border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px;
    padding: 12px 14px; margin: 12px 4px 0;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}
.sidebar-match-title { font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; color: #52525b; margin-bottom: 6px; font-weight: 600; }
.sidebar-match-teams { font-size: 13px; font-weight: 600; color: #fafafa; margin-bottom: 4px; }
.sidebar-match-meta { font-size: 11px; color: #a1a1aa; }

/* ── METRICS (Glassmorphism minimalist) ── */
[data-testid="metric-container"] {
    background: #09090b !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    padding: 20px 24px !important;
    transition: border-color 0.2s ease;
}
[data-testid="metric-container"]:hover {
    border-color: rgba(255, 255, 255, 0.2) !important;
}
[data-testid="stMetricLabel"] { font-size: 12px !important; color: #a1a1aa !important; font-weight: 500 !important; margin-bottom: 4px !important; }
[data-testid="stMetricValue"] { font-size: 28px !important; font-weight: 600 !important; color: #fafafa !important; letter-spacing: -0.02em; }

/* ── BUTTONS (Impeccable Style Monochrome) ── */
.stButton > button {
    background: #fafafa !important;
    color: #09090b !important; border: 1px solid transparent !important; border-radius: 6px !important;
    font-weight: 500 !important; font-size: 13px !important; padding: 10px 20px !important;
    transition: all 0.2s ease !important;
    letter-spacing: -0.01em;
}
.stButton > button:hover {
    background: #e4e4e7 !important;
}
.stButton > button[kind="secondary"] {
    background: #09090b !important;
    color: #fafafa !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #18181b !important;
    border-color: rgba(255, 255, 255, 0.25) !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] { background: transparent; border-bottom: 1px solid rgba(255, 255, 255, 0.1); gap: 16px; }
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #a1a1aa; font-size: 14px; font-weight: 500;
    padding: 12px 16px; border-bottom: 2px solid transparent; border-radius: 6px 6px 0 0;
}
.stTabs [aria-selected="true"] {
    color: #fafafa !important;
    border-bottom: 2px solid #fafafa !important;
}

/* ── INPUTS ── */
.stSelectbox div[data-baseweb="select"] span,
.stSelectbox div[data-baseweb="select"] div {
    color: #fafafa !important;
}
div[data-baseweb="popover"] ul {
    background: #18181b !important; border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 6px !important;
}
div[data-baseweb="popover"] li, div[data-baseweb="popover"] span, div[data-baseweb="popover"] div {
    color: #fafafa !important;
}
.stTextInput > div > div > input, .stNumberInput > div > div > input {
    background: #09090b !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 6px !important;
    color: #fafafa !important;
    padding: 10px 14px !important;
    font-size: 14px;
}
[data-testid="stSelectbox"] > div > div, [data-testid="stMultiSelect"] > div > div {
    background: #09090b !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 6px !important;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(255, 255, 255, 0.5) !important;
    box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.5) !important;
}

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    background: #09090b !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
}

/* ── REUSABLE COMPONENTS ── */
.ws-player-header {
    background: #09090b;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px; padding: 32px 40px; display: flex;
    align-items: center; gap: 32px; margin-bottom: 32px;
}
.ws-player-number { font-size: 48px; font-weight: 600; color: #52525b; line-height: 1; min-width: 60px; }
.ws-player-name { font-size: 28px; font-weight: 600; color: #fafafa; margin-bottom: 8px; letter-spacing: -0.02em; }
.ws-player-meta { font-size: 14px; color: #a1a1aa; font-weight: 500; }

.ws-section-header {
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.1em; color: #a1a1aa; margin: 32px 0 16px;
    display: flex; align-items: center; gap: 16px;
}
.ws-section-header::after { content: ''; flex: 1; height: 1px; background: rgba(255, 255, 255, 0.1); }

/* ── LOGIN ── */
</style>
"""
