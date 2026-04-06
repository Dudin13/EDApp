import streamlit as st
import pandas as pd
import os
from pathlib import Path
import time
from PIL import Image
import cv2
import numpy as np

from core.config.settings import settings

def main():
    st.markdown("""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                    padding: 24px; border-radius: 16px; border-bottom: 3px solid #00d4aa;
                    margin-bottom: 25px; box-shadow: 0 4px 20px rgba(0,0,0,0.3);">
            <h1>🔬 Auditoría y Corrección de Dataset</h1>
            <p>Supervisa y corrige el etiquetado del dataset de entrenamiento</p>
        </div>
    """, unsafe_allow_html=True)

    # --- LEYENDA DE CLASES ---
    st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(0, 212, 170, 0.2);
                    border-radius: 12px; padding: 16px; margin-bottom: 20px; text-align: center;">
            <strong>LEYENDA DE CLASES:</strong><br>
            0=Team A, 1=Team B, 2=Goalkeeper, 3=Referee, 4=Ball
        </div>
    """, unsafe_allow_html=True)

    # Configuración de dataset
    col_config, col_nav = st.columns([2, 1])

    with col_config:
        dataset_options = {
            "Super Focused 50": settings.BASE_DIR / "data" / "datasets" / "super_focused_50" / "train" / "images",
            "Validación Híbrida": settings.BASE_DIR / "data" / "datasets" / "hybrid_dataset" / "valid" / "images",
            "Entrenamiento Híbrido": settings.BASE_DIR / "data" / "datasets" / "hybrid_dataset" / "train" / "images",
            "Imágenes de Entrenamiento": settings.BASE_DIR / "data" / "datasets" / "imagenes_entrenamiento",
            "Dataset de Muestras": settings.BASE_DIR / "data" / "samples"
        }

        dataset_name = st.selectbox("Seleccionar Dataset:", list(dataset_options.keys()))
        img_dir = dataset_options[dataset_name]

    # Verificar que existe el directorio
    if not img_dir.exists():
        st.error(f"❌ No se encontró el directorio: {img_dir}")
        st.info("💡 Asegúrate de que tienes imágenes en el dataset seleccionado.")
        return

    # Obtener todas las imágenes
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    all_images = []
    for ext in image_extensions:
        all_images.extend(list(img_dir.glob(ext)))

    all_images = sorted(all_images)
    total_images = len(all_images)

    if total_images == 0:
        st.warning(f"⚠️ No se encontraron imágenes en {img_dir}")
        return

    # Navegación entre imágenes
    with col_nav:
        page = st.number_input(
            f"Imagen (1-{total_images})",
            min_value=1,
            max_value=total_images,
            value=st.session_state.get('audit_page', 1)
        )
        st.session_state['audit_page'] = page

    # Botones de navegación
    col_prev, col_next = st.columns(2)
    with col_prev:
        if st.button("⬅️ Anterior", disabled=(page <= 1)):
            st.session_state['audit_page'] = page - 1
            st.rerun()

    with col_next:
        if st.button("Siguiente ➡️", disabled=(page >= total_images)):
            st.session_state['audit_page'] = page + 1
            st.rerun()

    # Obtener imagen actual
    current_img_path = all_images[page - 1]
    st.markdown(f"### 📸 Imagen {page}/{total_images}: `{current_img_path.name}`")

    # Cargar modelo YOLO
    model = None
    model_paths = [
        settings.PLAYER_MODEL_PATH,
        settings.BASE_DIR / "assets" / "weights" / "detect_players_v3.pt",
        settings.BASE_DIR / "yolo11m.pt"
    ]

    for model_path in model_paths:
        if model_path and model_path.exists():
            try:
                from ultralytics import YOLO
                model = YOLO(str(model_path))
                st.success(f"✅ Modelo cargado: {model_path.name}")
                break
            except Exception as e:
                st.warning(f"⚠️ Error cargando {model_path.name}: {e}")
                continue

    if not model:
        st.error("❌ No se pudo cargar ningún modelo YOLO")
        return

    # Layout principal: Izquierda (IA) vs Derecha (Corrección)
    col_ia, col_manual = st.columns(2)

    # --- COLUMNA IZQUIERDA: PREDICCIÓN AUTOMÁTICA ---
    with col_ia:
        st.markdown("### 🤖 Predicción Automática (IA)")

        try:
            # Realizar predicción
            confidence = st.slider("Confianza mínima:", 0.1, 1.0, 0.25, 0.05,
                                 key=f"audit_conf_{current_img_path.name}")

            results = model.predict(
                str(current_img_path),
                conf=confidence,
                verbose=False
            )[0]

            # Mostrar imagen con detecciones
            img_with_dets = results.plot(boxes=True, masks=True)

            # Convertir BGR a RGB para Streamlit
            if isinstance(img_with_dets, np.ndarray):
                img_rgb = cv2.cvtColor(img_with_dets, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, use_container_width=True,
                        caption=f"Predicciones IA (Conf: {confidence})")

            # Mostrar estadísticas
            num_detections = len(results.boxes) if results.boxes is not None else 0
            st.metric("Detecciones", num_detections)

            # Lista de detecciones
            if results.boxes is not None:
                st.markdown("**Detecciones encontradas:**")
                for i, box in enumerate(results.boxes):
                    cls_id = int(box.cls.item())
                    conf_score = box.conf.item()
                    class_names = ["Team A", "Team B", "Goalkeeper", "Referee", "Ball"]
                    class_name = class_names[cls_id] if cls_id < len(class_names) else f"Clase {cls_id}"
                    st.write(f"• {class_name} ({conf_score:.2f})")

        except Exception as e:
            st.error(f"Error en predicción: {e}")
            # Mostrar imagen original si falla la predicción
            try:
                img_original = Image.open(current_img_path)
                st.image(img_original, use_container_width=True, caption="Imagen original")
            except:
                st.error("No se pudo cargar la imagen")

    # --- COLUMNA DERECHA: CORRECCIÓN MANUAL ---
    with col_manual:
        st.markdown("### ✏️ Corrección Manual")

        # Cargar imagen original para el canvas
        try:
            bg_image = Image.open(current_img_path)
            w, h = bg_image.size

            # Ajustar tamaño para el canvas
            display_width = 560
            display_height = int(h * (display_width / w))

            # Canvas para dibujar correcciones
            from streamlit_drawable_canvas import st_canvas

            canvas_result = st_canvas(
                fill_color="rgba(0, 212, 170, 0.3)",  # Color de relleno
                stroke_width=2,
                stroke_color="#00d4aa",  # Color del borde
                background_image=bg_image,
                update_streamlit=True,
                height=display_height,
                width=display_width,
                drawing_mode="rect",  # Solo rectángulos
                key=f"audit_canvas_{current_img_path.name}_{page}",
            )

            # Información sobre el canvas
            if canvas_result.json_data and canvas_result.json_data["objects"]:
                num_manual = len(canvas_result.json_data["objects"])
                st.metric("Rectángulos dibujados", num_manual)

                # Selector de clase para las correcciones manuales
                st.markdown("**Asignar clase a los rectángulos dibujados:**")
                target_class = st.selectbox(
                    "Clase:",
                    options=[0, 1, 2, 3, 4],
                    format_func=lambda x: ["Team A", "Team B", "Goalkeeper", "Referee", "Ball"][x],
                    key=f"audit_class_{current_img_path.name}"
                )

            else:
                st.info("💡 Dibuja rectángulos sobre los jugadores que la IA no detectó correctamente")

        except Exception as e:
            st.error(f"Error cargando canvas: {e}")

    # --- SECCIÓN DE DECISIÓN ---
    st.markdown("---")
    st.markdown("### 🎯 Decisión Final")

    col_decision, col_save = st.columns([2, 1])

    with col_decision:
        decision = st.radio(
            "Cuál versión usar para el entrenamiento:",
            ["Usar predicción automática (IA)", "Usar corrección manual", "Mantener etiquetas existentes"],
            key=f"audit_decision_{current_img_path.name}"
        )

    with col_save:
        if st.button("💾 Guardar Decisión", use_container_width=True, type="primary"):
            save_audit_correction(
                current_img_path,
                decision,
                canvas_result.json_data if canvas_result.json_data else None,
                target_class if 'target_class' in locals() else 0,
                display_width,
                display_height
            )

    # --- ESTADÍSTICAS DE PROGRESO ---
    st.markdown("---")
    st.markdown("### 📊 Estadísticas del Dataset")

    # Mostrar estadísticas del dataset actual
    col_stats1, col_stats2, col_stats3 = st.columns(3)

    with col_stats1:
        st.metric("Total Imágenes", total_images)

    with col_stats2:
        # Contar etiquetas existentes
        labels_dir = img_dir.parent / "labels"
        if labels_dir.exists():
            label_files = list(labels_dir.glob("*.txt"))
            st.metric("Con Etiquetas", len(label_files))
        else:
            st.metric("Con Etiquetas", 0)

    with col_stats3:
        # Calcular porcentaje
        labels_dir = img_dir.parent / "labels"
        if labels_dir.exists():
            label_files = list(labels_dir.glob("*.txt"))
            percentage = (len(label_files) / total_images) * 100 if total_images > 0 else 0
            st.metric("Completado", ".1f")
        else:
            st.metric("Completado", "0.0%")

    st.info("💡 Esta herramienta te permite auditar y corregir todo tu dataset de entrenamiento, mejorando la calidad del modelo YOLO.")


def save_audit_correction(img_path, decision, canvas_data, target_class, display_width, display_height):
    """Guarda la corrección de auditoría como archivo de etiquetas YOLO."""

    # Directorio de etiquetas
    labels_dir = img_path.parent.parent / "labels"
    labels_dir.mkdir(exist_ok=True)

    label_path = labels_dir / f"{img_path.stem}.txt"

    if decision == "Usar predicción automática (IA)":
        st.info("✅ Se mantendrán las predicciones automáticas de la IA")

    elif decision == "Usar corrección manual":
        if not canvas_data or not canvas_data["objects"]:
            st.error("❌ No hay rectángulos dibujados para guardar")
            return

        # Convertir rectángulos del canvas a formato YOLO
        yolo_labels = []
        for obj in canvas_data["objects"]:
            # Normalizar coordenadas
            left = obj["left"] / display_width
            top = obj["top"] / display_height
            width = obj["width"] / display_width
            height = obj["height"] / display_height

            # Centro del rectángulo
            x_center = left + (width / 2)
            y_center = top + (height / 2)

            # Formato YOLO: class x_center y_center width height
            yolo_label = f"{target_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_labels.append(yolo_label)

        # Guardar archivo de etiquetas
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_labels))

        st.success(f"✅ Correcciones guardadas en {label_path.name}")
        st.info(f"📝 Se crearon {len(yolo_labels)} etiquetas para la clase '{['Team A', 'Team B', 'Goalkeeper', 'Referee', 'Ball'][target_class]}'")

    elif decision == "Mantener etiquetas existentes":
        if label_path.exists():
            st.info("✅ Se mantendrán las etiquetas existentes")
        else:
            st.warning("⚠️ No existen etiquetas previas para esta imagen")

    # Pequeño delay para mostrar el mensaje
    time.sleep(1.5)


# Ejecutar solo si se llama directamente
if __name__ == "__main__":
    main()