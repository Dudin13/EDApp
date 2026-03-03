import streamlit as st
import pandas as pd

def render():
    st.header("⚙️ Configuración del sistema")

    tab1, tab2 = st.tabs(["🤖 Modelos IA", "👥 Jugadores cedidos"])

    with tab1:
        st.subheader("Detección de jugadores (YOLO)")
        yolo_model = st.selectbox("Modelo YOLOv8", [
            "yolov8n (más rápido, menos preciso)",
            "yolov8s (equilibrado) ← Recomendado",
            "yolov8m (más preciso, más lento)",
            "yolov8x (máxima precisión, muy lento)"
        ], index=1)
        confidence = st.slider("Umbral de confianza YOLO", 0.1, 0.9, 0.5, 0.05)

        st.markdown("---")
        st.subheader("Clips de vídeo")
        clip_before = st.slider("Segundos antes de la acción", 2, 10, 5)
        clip_after = st.slider("Segundos después de la acción", 2, 10, 5)

        st.markdown("---")
        st.subheader("Credenciales")
        st.text_input("Roboflow API Key", value="rf_ttfHF1xorNTOx8wEqJwqGFvB4Xv2", type="password")
        st.text_input("SoccerNet Email", value="dudin_13@hotmail.com")
        st.text_input("SoccerNet Password", value="s0cc3rn3t", type="password")

        if st.button("💾 Guardar configuración",