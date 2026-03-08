import streamlit as st
import base64
from pathlib import Path

def render():
    base_dir = Path(__file__).parent.parent
    logo_path = base_dir / "logo.png"
    logo_b64 = ""
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()
            
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="height:120px; margin-bottom:20px; border-radius:16px; box-shadow:0 8px 32px rgba(0,0,0,0.5);">' if logo_b64 else '<div style="font-size:72px; margin-bottom:10px;">⚽</div>'

    st.markdown(f"""
        <style>
        .home-container {{ display: flex; flex-direction: column; align-items: center; justify-content: flex-start; min-height: 80vh; padding-top: 40px; }}
        .club-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            width: 100%;
            max-width: 1000px;
            margin-top: 40px;
        }}
        .club-card {{
            background: rgba(17, 24, 39, 0.6);
            border: 1px solid rgba(0, 212, 170, 0.2);
            border-radius: 16px;
            padding: 24px 16px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }}
        .club-card:hover {{
            transform: translateY(-5px);
            border-color: rgba(0, 212, 170, 0.8);
            box-shadow: 0 8px 25px rgba(0, 212, 170, 0.2);
            background: rgba(17, 24, 39, 0.9);
        }}
        .club-name {{
            color: #fff;
            font-size: 18px;
            font-weight: 700;
            margin-top: 10px;
            margin-bottom: 5px;
        }}
        .club-category {{
            color: #5a6a7e;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        </style>

        <div class="home-container">
            {logo_html}
            <h1 style="color:#fff; font-size:48px; font-weight:900; letter-spacing:2px; margin:0;">ED Analytics</h1>
            <p style="color:#00d4aa; font-size:18px; font-weight:700; text-transform:uppercase; letter-spacing:3px;">Scout Platform</p>
            
            <div style="margin-top:30px; margin-bottom:10px; color:#a2b9ce; font-size:16px; font-weight:600;">Selecciona un equipo para gestionar:</div>
            
            <div class="club-grid">
                <div class="club-card">
                    <div class="club-name">Cádiz CF</div>
                    <div class="club-category">Primer Equipo</div>
                </div>
                <div class="club-card">
                    <div class="club-name">Cádiz Mirandilla</div>
                    <div class="club-category">Filial</div>
                </div>
                <div class="club-card">
                    <div class="club-name">Balón de Cádiz</div>
                    <div class="club-category">Senior Amateur</div>
                </div>
                <div class="club-card">
                    <div class="club-name">Juvenil A</div>
                    <div class="club-category">División de Honor</div>
                </div>
                <div class="club-card">
                    <div class="club-name">Juvenil B</div>
                    <div class="club-category">Liga Nacional</div>
                </div>
                <div class="club-card">
                    <div class="club-name">Cadete A</div>
                    <div class="club-category">División de Honor</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
