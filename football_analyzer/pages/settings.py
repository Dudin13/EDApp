import streamlit as st
import pandas as pd
import json
from pathlib import Path

CONFIG_FILE = Path(__file__).parent.parent / "app_config.json"


def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_config(cfg: dict):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def render():
    st.header("⚙️ Configuración del sistema")

    saved = load_config()

    tab1, tab2 = st.tabs(["🤖 Modelos IA", "👥 Jugadores cedidos"])

    with tab1:
        st.subheader("Motor de detección")
        detection_mode = st.selectbox(
            "Backend de detección",
            ["roboflow", "yolo (local)", "auto"],
            index=["roboflow", "yolo (local)", "auto"].index(
                saved.get("detection_mode", "roboflow")
            ),
            help="'auto' usa YOLOv8 local si existe el modelo entrenado, sino Roboflow"
        )

        # Estado del modelo YOLOv8 local
        yolo_path = Path(__file__).parent.parent / "train_yolo" / "runs" / "detect" / "train" / "weights" / "best.pt"
        if yolo_path.exists():
            size_mb = yolo_path.stat().st_size / 1024 / 1024
            st.success(f"✅ Modelo YOLOv8 local disponible ({size_mb:.1f} MB) — {yolo_path}")

        st.markdown("---")
        st.subheader("Parámetros de análisis")
        col1, col2 = st.columns(2)
        confidence = col1.slider("Umbral de confianza", 20, 80,
                                 saved.get("confidence", 40), 5)
        sample_rate = col2.slider("Intervalo entre frames analizados (segundos)", 1, 10,
                                  saved.get("sample_rate", 3))

        st.markdown("---")
        st.subheader("Clips de vídeo")
        col3, col4 = st.columns(2)
        clip_before = col3.slider("Segundos antes de la acción", 2, 10,
                                  saved.get("clip_before", 5))
        clip_after = col4.slider("Segundos después de la acción", 2, 10,
                                 saved.get("clip_after", 5))

        st.markdown("---")
        st.subheader("Credenciales")

        import os
        try:
            from dotenv import load_dotenv
            load_dotenv(Path(__file__).parent.parent / ".env")
        except Exception:
            pass

        roboflow_key = st.text_input(
            "Roboflow API Key",
            value=saved.get("roboflow_key", os.getenv("ROBOFLOW_API_KEY", "")),
            type="password"
        )
        ffmpeg_path = st.text_input(
            "Ruta a FFmpeg",
            value=saved.get("ffmpeg_path", os.getenv("FFMPEG_PATH", r"C:\ffmpeg\bin\ffmpeg.exe"))
        )

        if st.button("💾 Guardar configuración", type="primary"):
            cfg = {
                "detection_mode": detection_mode.replace(" (local)", ""),
                "confidence": confidence,
                "sample_rate": sample_rate,
                "clip_before": clip_before,
                "clip_after": clip_after,
                "roboflow_key": roboflow_key,
                "ffmpeg_path": ffmpeg_path,
            }
            save_config(cfg)
            # Actualizar session_state con la nueva config
            st.session_state["app_config"] = cfg
            st.success("✅ Configuración guardada correctamente")

    with tab2:
        st.subheader("👥 Registro de jugadores cedidos")
        st.info("Registra aquí los jugadores cedidos para seguimiento automático en todos los partidos.")

        cedidos = saved.get("jugadores_cedidos", [])

        # Mostrar tabla de cedidos actuales
        if cedidos:
            df = pd.DataFrame(cedidos)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.markdown("---")

        st.subheader("➕ Añadir jugador cedido")
        col_a, col_b, col_c, col_d = st.columns(4)
        nombre_c = col_a.text_input("Nombre")
        dorsal_c = col_b.number_input("Dorsal", 1, 99, 1)
        equipo_c = col_c.text_input("Equipo cedido")
        pos_c = col_d.selectbox("Posición", [
            "Portero", "Lateral D", "Lateral I", "Central", "Pivote",
            "Mediocentro", "Interior D", "Interior I", "Mediapunta",
            "Extremo D", "Extremo I", "Delantero Centro"
        ])

        if st.button("➕ Añadir"):
            if nombre_c and equipo_c:
                cedidos.append({
                    "nombre": nombre_c, "dorsal": dorsal_c,
                    "equipo": equipo_c, "posicion": pos_c
                })
                saved["jugadores_cedidos"] = cedidos
                save_config(saved)
                st.success(f"✅ {nombre_c} añadido")
                st.rerun()
            else:
                st.error("Rellena al menos nombre y equipo")