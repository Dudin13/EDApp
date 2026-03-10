import streamlit as st
import cv2
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image
import os

def render():
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:28px;">
        <div>
            <h2 style="margin:0;font-size:20px;font-weight:700;color:#fff;">Calibración de Homografía</h2>
            <p style="margin:2px 0 0;font-size:13px;color:#5a6a7e;">Selecciona 4 puntos clave en el vídeo y asígnalos al mapa 2D.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "calib_points" not in st.session_state:
        st.session_state["calib_points"] = []

    video_path = st.session_state.get("video_path", "")
    
    if not video_path or not os.path.exists(video_path):
        st.warning("⚠️ Primero sube o selecciona un vídeo en 'Análisis de Vídeo'.")
        return

    # Extraer el primer frame del vídeo
    @st.cache_data
    def get_first_frame(v_path):
        cap = cv2.VideoCapture(v_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            # Convert BGR to RGB
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    frame_rgb = get_first_frame(video_path)
    
    if frame_rgb is None:
        st.error("❌ No se pudo extraer un frame del vídeo.")
        return

    st.markdown('<div class="ws-section-header">Paso 1: Clickea 4 puntos en el vídeo</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("Haz clic en 4 esquinas/intersecciones claras del campo.")
        # Render image and catch clicks
        value = streamlit_image_coordinates(
            frame_rgb,
            key="calib_img",
            width=800
        )
        
        # Guardar click si es nuevo
        if value is not None:
            point = (value["x"], value["y"])
            # Evitar duplicados inmediatos por re-runs
            if not st.session_state["calib_points"] or point != st.session_state["calib_points"][-1]:
                if len(st.session_state["calib_points"]) < 4:
                    st.session_state["calib_points"].append(point)
                    st.rerun()

    with col2:
        st.markdown('<div class="ws-section-header" style="margin-top:0;">Puntos Seleccionados</div>', unsafe_allow_html=True)
        for i, pt in enumerate(st.session_state["calib_points"]):
            st.code(f"P{i+1}: {pt[0]}, {pt[1]}")
            
        if st.button("🔄 Borrar", use_container_width=True):
            st.session_state["calib_points"] = []
            st.rerun()
            
        if len(st.session_state["calib_points"]) == 4:
            st.success("✅ 4 puntos seleccionados")
            
            # Formatear para copiar y pegar
            pts_str = "[\n"
            for pt in st.session_state["calib_points"]:
                # Ajustar asumiendo original 1280x720 y la imagen mostrada a 800px width
                # ratio = 1280 / 800 = 1.6
                orig_x = int(pt[0] * (1280 / 800))
                orig_y = int(pt[1] * (1280 / 800))
                pts_str += f"    [{orig_x}, {orig_y}],\n"
            pts_str += "]"
            st.text_area("Coordenadas Vídeo (Copia esto):", value=pts_str, height=120)

    st.markdown("---")
    st.markdown('<div class="ws-section-header">Paso 2: Asigna las coordenadas reales 2D</div>', unsafe_allow_html=True)
    st.write("Ahora, escribe las coordenadas físicas (0 a 105, 0 a 68) de esos mismos 4 puntos en orden.")
    
    # Template estándar
    st.code("""
# Ejemplo de configuración para video_processor.py:
config = {
    ...
    "src_pts": [
        [150, 400], # Esquina inferior izq
        [1100, 400], # Esquina inferior der
        [300, 200], # Esquina superior izq
        [900, 200]  # Esquina superior der
    ],
    "dst_pts": [
        [0, 68],  # Bottom Left
        [105, 68], # Bottom Right
        [0, 0],   # Top Left
        [105, 0]   # Top Right
    ]
}
""")
    
    st.info("💡 Una vez tengas `src_pts` y `dst_pts`, añádelos a la configuración del análisis al instanciar VideoProcessor.")
