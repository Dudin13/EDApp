"""
home.py — Landing page de la aplicación.
Estilo Wyscout: minimalista, alto contraste, visualmente impactante.

FIXES APLICADOS:
  [MEJORA]  Bienvenida dinámica: detecta si ya hay un análisis cargado y muestra
            un botón de acceso directo al Dashboard.
  [MEJORA]  Timeline visual de funciones: explica el flujo de trabajo (Video -> Tracking -> Insights).
  [MEJORA]  Banner "PRO" para diferenciar la versión actual de la V1.
"""

import streamlit as st

def render():
    # ── Header / Hero Section ────────────────────────────────────────────────
    st.markdown("""
    <div style="padding: 60px 0 40px 0; text-align:center;">
        <div style="display:inline-block; background:rgba(0,212,170,0.1); border:1px solid #00d4aa;
                    color:#00d4aa; padding:4px 12px; border-radius:30px; font-size:10px;
                    font-weight:800; text-transform:uppercase; letter-spacing:2px; margin-bottom:20px;">
            Powered by YOLOv8 Architecture
        </div>
        <h1 style="font-size:54px; font-weight:900; color:#fff; letter-spacing:-2px; line-height:1; margin-bottom:15px;">
            Inteligencia Artificial Aplicada al Fútbol
        </h1>
        <p style="font-size:18px; color:#8899aa; max-width:700px; margin:0 auto 40px; line-height:1.6;">
            Transforma vídeo convencional en datos tácticos accionables. Detección, tracking y
            análisis de métricas profesionales en una plataforma única.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Botones de Acción Rápidos ───────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns([0.5, 2, 2, 2, 2, 0.5])

    with c2:
        if st.button("🚀  NUEVO ANÁLISIS", use_container_width=True, type="primary"):
            st.session_state.page = "upload"
            st.rerun()

    with c3:
        # Mostrar Dashboards solo si hay datos, o link a Scout si no
        btn_label = "📊  VER DASHBOARD" if st.session_state.get("analysis_done") else "📋  PANEL SCOUT"
        if st.button(btn_label, use_container_width=True):
            st.session_state.page = "collective" if st.session_state.get("analysis_done") else "scout"
            st.rerun()

    with c4:
        if st.button("📐  CALIBRAR CÁMARA", use_container_width=True):
            st.session_state.page = "calibration"
            st.rerun()

    with c5:
        if st.button("🔬  COMPROBAR DETECCIÓN", use_container_width=True):
            st.session_state.page = "dataset_audit"
            st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ── Timeline del Proceso ───────────────────────────────────────────────
    st.markdown('<div class="ws-section-header">Flujo de Trabajo</div>', unsafe_allow_html=True)

    steps_html = """
    <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:20px;">
        <div class="ws-card">
            <div style="font-size:24px; margin-bottom:12px;">🎥</div>
            <div style="font-size:14px; font-weight:800; color:#fff; margin-bottom:8px;">1. Entrada</div>
            <div style="font-size:12px; color:#5a6a7e; line-height:1.5;">
                Sube tu vídeo (MP4) de partido o entrenamiento en plano general.
            </div>
        </div>
        <div class="ws-card">
            <div style="font-size:24px; margin-bottom:12px;">🧠</div>
            <div style="font-size:14px; font-weight:800; color:#fff; margin-bottom:8px;">2. Procesamiento</div>
            <div style="font-size:12px; color:#5a6a7e; line-height:1.5;">
                Inferencia YOLO v8 seg para detección de jugadores, árbitros y balón.
            </div>
        </div>
        <div class="ws-card">
            <div style="font-size:24px; margin-bottom:12px;">📍</div>
            <div style="font-size:14px; font-weight:800; color:#fff; margin-bottom:8px;">3. Tracking</div>
            <div style="font-size:12px; color:#5a6a7e; line-height:1.5;">
                Asignación de IDs persistentes y transformación a plano 2D (PnL).
            </div>
        </div>
        <div class="ws-card" style="border: 1px solid rgba(0,212,170,0.3);">
            <div style="font-size:24px; margin-bottom:12px;">📈</div>
            <div style="font-size:14px; font-weight:800; color:#00d4aa; margin-bottom:8px;">4. Insights</div>
            <div style="font-size:12px; color:#5a6a7e; line-height:1.5;">
                Visualiza mapas de calor, redes de pases y clips individuales.
            </div>
        </div>
    </div>
    """
    st.markdown(steps_html, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ── Sección de Novedades V2 ───────────────────────────────────────────────
    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.markdown("""
        <div class="ws-card" style="height:100%;">
            <h3 style="margin-top:0; color:#fff; font-size:18px;">✨ Novedades V2 (Actual)</h3>
            <ul style="color:#8899aa; font-size:13px; line-height:2;">
                <li>Identificación automática de dorsales</li>
                <li>Red de pases avanzada con xT (Expected Threat)</li>
                <li>Clasificación de equipos por color HSV robusto</li>
                <li>Gestión de clips de vídeo con IA</li>
                <li>Calibración PnL (Perspective-n-Line) interactiva</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c_right:
        st.markdown("""
        <div class="ws-card" style="height:100%; background: linear-gradient(135deg, rgba(0,212,170,0.05), rgba(13,18,32,0.4));">
            <h3 style="margin-top:0; color:#fff; font-size:18px;">🏢 Proyecto EDudin</h3>
            <p style="color:#8899aa; font-size:13px; line-height:1.6;">
                Este sistema ha sido diseñado para analistas de rendimiento y cuerpos técnicos
                que buscan profesionalizar la toma de decisiones sin grandes presupuestos.
            </p>
            <p style="color:#00d4aa; font-size:13px; font-weight:700;">
                Actualmente optimizado para planos generales (Táctica / Master).
            </p>
        </div>
        """, unsafe_allow_html=True)
