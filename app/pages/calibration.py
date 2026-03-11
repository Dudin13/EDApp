"""
calibration.py — Página de calibración de campo 2D.
Permite definir los 4 puntos de referencia para transformar la perspectiva.

FIXES APLICADOS:
  [CRÍTICO] Reset de calibración: antes no se borraban los puntos de
            st.session_state.calib_points, lo que impedía volver a calibrar.
  [MEJORA]  Visualización del campo 2D: ahora muestra los puntos seleccionados
            en tiempo real para facilitar la precisión.
  [MEJORA]  Integración con PnLCalibrator: guarda los resultados automáticamente
            en el singleton para que detector.py los use de inmediato.
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from modules.calibration_pnl import PnLCalibrator

def render():
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:28px;">
        <div>
            <h2 style="margin:0;font-size:20px;font-weight:700;color:#fff;">Calibración de Campo</h2>
            <p style="margin:2px 0 0;font-size:13px;color:#5a6a7e;">Transformación de perspectiva de cámara a plano 2D</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    calibrator = PnLCalibrator()

    # ── Estado de Calibración ────────────────────────────────────────────────
    if "calib_points" not in st.session_state:
        st.session_state.calib_points = []
    if "calib_img" not in st.session_state:
        st.session_state.calib_img = None

    # ── Carga de Imagen de Referencia ─────────────────────────────────────────
    st.markdown('<div class="ws-section-header">1. Imagen de Referencia</div>', unsafe_allow_html=True)

    source = st.radio("Origen de imagen:", ["Manual (Subir Frame)", "Último vídeo analizado"], horizontal=True)

    if source == "Último vídeo analizado":
        video_path = st.session_state.get("video_path")
        if video_path and Path(video_path).exists():
            if st.button("Extraer frame central"):
                cap = cv2.VideoCapture(video_path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
                ret, frame = cap.read()
                if ret:
                    st.session_state.calib_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cap.release()
        else:
            st.warning("No hay vídeo analizado en la sesión actual.")
    else:
        uploaded_file = st.file_uploader("Sube un frame del campo", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            import PIL.Image
            img = PIL.Image.open(uploaded_file)
            st.session_state.calib_img = np.array(img)

    if st.session_state.calib_img is not None:
        # ── Selección de Puntos ───────────────────────────────────────────────
        st.markdown('<div class="ws-section-header">2. Seleccionar 4 esquinas del área/campo</div>', unsafe_allow_html=True)

        col_inst, col_btn = st.columns([3, 1])
        with col_inst:
            st.info("""
            Haz click sobre la imagen en este orden:
            1. Córner Arriba-Izquierda | 2. Córner Arriba-Derecha |
            3. Córner Abajo-Derecha | 4. Córner Abajo-Izquierda
            """)

        with col_btn:
            if st.button("🔄 Reiniciar Puntos", use_container_width=True):
                st.session_state.calib_points = []
                st.rerun()

        # Canvas para click (usando st.image simplificado o streamlit-drawable-canvas si estuviera instalado)
        # Por simplicidad en esta versión, usamos coordenadas manuales o clicks si el componente está disponible.
        # Aquí simulamos la lógica de visualización:

        img_display = st.session_state.calib_img.copy()
        for i, pt in enumerate(st.session_state.calib_points):
            cv2.circle(img_display, pt, 15, (0, 212, 170), -1)
            cv2.putText(img_display, str(i+1), (pt[0]-10, pt[1]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 4)

        st.image(img_display, use_column_width=True)

        # Simulación de puntos para desarrollo (en producción se usaría un componente de captura)
        if not st.session_state.calib_points:
            st.markdown("""
            <div style="background:rgba(255,255,255,0.02); border:1px dashed rgba(255,255,255,0.1);
                        border-radius:12px; padding:30px; text-align:center; color:#5a6a7e;">
                Interactúa con la imagen para marcar los puntos.
                <br><span style="font-size:10px;">(Nota: Requiere componente streamlit-canvas para clicks directos)</span>
            </div>
            """, unsafe_allow_html=True)

            # Fallback manual para testeo
            with st.expander("Introducir coordenadas manualmente"):
                pts_input = st.text_input("Formato: x1,y1;x2,y2;x3,y3;x4,y4", "")
                if st.button("Aplicar coordenadas"):
                    try:
                        pts = [tuple(map(int, p.split(','))) for p in pts_input.split(';')]
                        if len(pts) == 4:
                            st.session_state.calib_points = pts
                            st.rerun()
                    except:
                        st.error("Formato inválido")

        if len(st.session_state.calib_points) == 4:
            st.success("✅ 4 puntos seleccionados.")
            if st.button("💾 GUARDAR CALIBRACIÓN", type="primary"):
                # Asignar al calibrador
                pts = np.array(st.session_state.calib_points, dtype=np.float32)
                calibrator.set_points(pts)
                st.balloons()
                st.toast("Calibración guardada correctamente")

    # ── Vista Previa 2D ──────────────────────────────────────────────────────
    st.markdown('<div class="ws-section-header">Vista Previa 2D</div>', unsafe_allow_html=True)
    pitch = calibrator.get_warped_pitch(st.session_state.calib_img)
    if pitch is not None:
        st.image(pitch, caption="Transformación Perspectiva (Plano Cenital)", use_column_width=True)
    else:
        st.markdown("""
        <div style="height:200px; background:rgba(0,0,0,0.2); border-radius:12px;
                    display:flex; align-items:center; justify-content:center; color:#3a4a5e;">
            Pendiente de calibración
        </div>
        """, unsafe_allow_html=True)
