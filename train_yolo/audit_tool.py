import streamlit as st
import pandas as pd
import os
from pathlib import Path
import time
from PIL import Image

# Configuración de página
st.set_page_config(page_title="Herramienta de Auditoría IA", layout="wide", initial_sidebar_state="expanded")

# Rutas de entrenamiento (Localizadas en 03_ENTRENAMIENTOS ya que este script está dentro)
BASE_TRAIN = Path(__file__).parent
TRAIN_RUN_DIR = BASE_TRAIN / "runs" / "detect" / "EDudin_v1"
RESULTS_CSV = TRAIN_RUN_DIR / "results.csv"

# Estilos CSS (Glassmorphism + Dark Mode)
st.markdown("""
    <style>
    .monitor-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 24px;
        border-radius: 16px;
        border-bottom: 3px solid #00d4aa;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(0, 212, 170, 0.2);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<div class="monitor-header"><h1>🔍 Herramienta de Auditoría y Corrección Independiente</h1><p>Supervisa y corrige el etiquetado del dataset (Standalone version)</p></div>', unsafe_allow_html=True)

    # --- LEYENDA FIJA SEGÚN SKETCH ---
    st.markdown("""
        <div style="background: white; color: black; padding: 10px; border: 4px solid black; text-align: center; font-weight: bold; margin-bottom: 25px; font-family: 'Courier New', monospace; font-size: 1.2rem;">
            LEYENDA FIJA 0=TEAM 1, 1=TEAM 2, 2=GOALKEEPER, 3=REFEREE, 4=BALL, 5=GOALKEEPER 2
        </div>
    """, unsafe_allow_html=True)

    # Configuración de Paginación y Carpeta
    col_set, col_page = st.columns([1, 1])
    with col_set:
        dataset_type = st.radio("Conjunto de datos:", ["Super Focused (50)", "Validación Híbrido", "Entrenamiento Híbrido"], horizontal=True)
    
    if dataset_type == "Super Focused (50)":
        IMG_DIR = BASE_TRAIN / "04_Datasets_Entrenamiento/super_focused_50/train/images"
        CURRENT_RUN_DIR = BASE_TRAIN / "runs" / "detect" / "SuperFocused_v1"
    elif dataset_type == "Validación Híbrido":
        IMG_DIR = BASE_TRAIN / "04_Datasets_Entrenamiento/hybrid_dataset/valid/images"
        CURRENT_RUN_DIR = BASE_TRAIN / "runs" / "detect" / "EDudin_v1"
    else:
        IMG_DIR = BASE_TRAIN / "04_Datasets_Entrenamiento/hybrid_dataset/train/images"
        CURRENT_RUN_DIR = BASE_TRAIN / "runs" / "detect" / "EDudin_v1"
    
    if not IMG_DIR.exists():
        st.error(f"❌ No se encontró la carpeta de imágenes: {IMG_DIR}")
        return

    all_images = sorted(list(IMG_DIR.glob("*.jpg")) + list(IMG_DIR.glob("*.png")))
    total_imgs = len(all_images)
    
    if total_imgs > 0:
        with col_page:
            page = st.number_input(f"Página (1-{max(1, total_imgs)})", min_value=1, max_value=total_imgs, value=1)
        
        img_path = all_images[page-1]
        st.write(f"### 🏟️ Revisión de Jugada: `{img_path.name}`")

        # Cargar modelo
        model = None
        BEST_MODEL = CURRENT_RUN_DIR / "weights" / "best.pt"
        if not BEST_MODEL.exists():
            if dataset_type == "Super Focused (50)":
                BEST_MODEL = BASE_TRAIN / "yolo11m-seg.pt"
            else:
                BEST_MODEL = CURRENT_RUN_DIR / "weights" / "best.pt"
        
        if BEST_MODEL.exists():
            try:
                from ultralytics import YOLO
                model = YOLO(BEST_MODEL)
            except: pass

        from streamlit_drawable_canvas import st_canvas
        
        col1, col2 = st.columns(2)
        
        # --- IZQUIERDA: IMAGEN ENTRENADA ---
        with col1:
            if model:
                results = model.predict(img_path, conf=0.15, verbose=False)[0]
                img_pred = results.plot(boxes=True, masks=True)
                st.image(img_pred, use_container_width=True, caption=f"[PREDICCIÓN IA - Modelo: {CURRENT_RUN_DIR.name}]")
            else:
                st.info("IA no disponible")

        # --- DERECHA: IMAGEN PARA CORREGIR (CANVAS) ---
        with col2:
            st.write("✏️ **Dibuja rectángulos para corregir o añadir jugadores:**")
            bg_image = Image.open(img_path)
            w, h = bg_image.size
            # Ajustar tamaño para que quepa en el dashboard manteniendo aspecto
            display_width = 560
            display_height = int(h * (display_width / w))
            
            canvas_result = st_canvas(
                fill_color="rgba(0, 212, 170, 0.3)",  # Color del relleno
                stroke_width=2,
                stroke_color="#00d4aa",
                background_image=bg_image,
                update_streamlit=True,
                height=display_height,
                width=display_width,
                drawing_mode="rect",
                key=f"canvas_{img_path.name}_{dataset_type}", 
            )

        # --- CHECKBOXES SEGÚN SKETCH ---
        st.markdown("---")
        chk_ok = st.checkbox("✅ MARCA SI ES CORRECTA LA IMAGEN ENTRENADA", key=f"ok_{img_path.name}")
        chk_manual = st.checkbox("❌ MARCA SI LA IMAGEN CORRECTA ES LA CORREGIDA MANUALMENTE.", key=f"man_{img_path.name}")

        if chk_manual:
            if canvas_result.json_data and canvas_result.json_data["objects"]:
                st.info("💡 Se han detectado marcas manuales. Elige la clase para estas marcas:")
                target_cls = st.selectbox("Asignar clase a los nuevos dibujos:", 
                                        [0, 1, 2, 3, 4, 5], 
                                        format_func=lambda x: ["Team 1", "Team 2", "Goalkeeper 1", "Referee", "Ball", "Goalkeeper 2"][x])
                
                if st.button("💾 Guardar Correcciones Manuales", use_container_width=True):
                    lab_dir = IMG_DIR.parent / "labels"
                    lab_path = lab_dir / (img_path.stem + ".txt")
                    lab_dir.mkdir(parents=True, exist_ok=True)
                    new_labels = []
                    for obj in canvas_result.json_data["objects"]:
                        left = obj["left"] / display_width
                        top = obj["top"] / display_height
                        width = obj["width"] / display_width
                        height = obj["height"] / display_height
                        x_center = left + (width / 2)
                        y_center = top + (height / 2)
                        new_labels.append(f"{target_cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                    
                    with open(lab_path, "w") as f:
                        f.write("\n".join(new_labels))
                    
                    st.success(f"¡Etiquetas actualizadas para {img_path.name}!")
                    time.sleep(1)
                    st.rerun()
            else:
                st.warning("🔄 Dibuja rectángulos en la imagen de la derecha para poder guardar la corrección.")
        
        if chk_ok:
            st.success("🌟 ¡Imagen validada! No se requieren cambios.")

    else:
        st.error("No se encontraron imágenes en la carpeta seleccionada.")

    # --- Sección de Acción ---
    st.divider()
    st.subheader("🛠️ Acciones de Gestión")
    if st.button("📂 Abrir Carpeta de Dataset"):
        os.startfile(os.path.abspath(str(BASE_TRAIN / "04_Datasets_Entrenamiento")))

if __name__ == "__main__":
    main()
