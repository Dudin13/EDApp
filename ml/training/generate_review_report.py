import os
import cv2
from pathlib import Path
from ultralytics import YOLO
import base64

def get_image_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def find_image(lbl_stem, search_dirs):
    # Intentar con varias variaciones de nombre
    possible_names = [
        lbl_stem + ".jpg",
        lbl_stem + ".png",
        lbl_stem.replace("_ball", "") + ".jpg", 
        lbl_stem.replace("_ball", "") + ".png",
        "ball_" + lbl_stem.replace("_ball", "") + ".jpg",
        "veo_" + lbl_stem + ".jpg",
        "local_" + lbl_stem + ".jpg"
    ]
    
    for d in search_dirs:
        for name in possible_names:
            img_path = d / name
            if img_path.exists():
                return img_path
    
    # Búsqueda difusa final: si el hash de RoboFlow está en el nombre del archivo
    if ".rf." in lbl_stem:
        rf_hash = lbl_stem.split(".rf.")[1].split("_")[0]
        for d in search_dirs:
            for f in d.glob(f"*{rf_hash}*"):
                if f.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                    return f
                    
    return None

def generate_report(model_path, dataset_path, output_html="review_report.html"):
    print(f"🚀 Generando informe de revisión final...")
    model = YOLO(model_path)
    
    # Directorios donde buscar imágenes
    search_dirs = [
        Path(dataset_path) / "valid" / "images",
        Path(dataset_path) / "train" / "images",
        Path("C:/apped/data/datasets/imagenes_entrenamiento"),
        Path("C:/apped/data/datasets/veo_frames_raw"),
        Path("C:/apped/data/datasets/roboflow_dataset/train/images"),
        Path("C:/apped/data/datasets/roboflow_dataset/valid/images"),
        Path("C:/apped/data/datasets/hybrid_dataset/temp_boxes_only"),
    ]
    
    # Obtener etiquetas de validación
    label_dir = Path(dataset_path) / "valid" / "labels"
    if not label_dir.exists():
        print(f"❌ No se encontró el directorio de etiquetas: {label_dir}")
        return

    label_files = list(label_dir.glob("*.txt"))
    print(f"📋 Encontradas {len(label_files)} etiquetas en validación.")
    
    images_to_process = []
    for lbl in label_files:
        img_path = find_image(lbl.stem, search_dirs)
        if img_path:
            images_to_process.append(img_path)
        if len(images_to_process) >= 50: # Límite de 50 para el reporte
            break

    if not images_to_process:
        print("❌ No se encontraron imágenes para las etiquetas de validación.")
        # Fallback: intentar leer cualquier imagen de imagenes_entrenamiento
        print("💡 Intentando cargar imágenes directas de 'imagenes_entrenamiento' como fallback...")
        fallback_dir = Path("C:/apped/data/datasets/imagenes_entrenamiento")
        if fallback_dir.exists():
            images_to_process = list(fallback_dir.glob("*.jpg"))[:50]

    if not images_to_process:
        print("❌ Error crítico: No se encontraron imágenes para procesar.")
        return

    html_content = """
    <html>
    <head>
        <title>Auditoría Final - ED Analytics</title>
        <style>
            body { font-family: sans-serif; background: #0f172a; color: white; padding: 20px; }
            .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(600px, 1fr)); gap: 20px; }
            .card { background: #1e293b; border-radius: 12px; padding: 15px; border: 1px solid #334155; }
            .img-row { display: flex; gap: 10px; }
            .img-container { flex: 1; text-align: center; }
            img { max-width: 100%; border-radius: 8px; border: 2px solid #00d4aa; }
            h1 { color: #00d4aa; }
            .meta { color: #8899aa; font-size: 0.9em; margin-bottom: 10px; }
            .badge { background: #00d4aa; color: black; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8em; }
            .summary-box { display: flex; gap: 5px; flex-wrap: wrap; margin-bottom: 5px; }
            .badge-team1 { background: #00d4aa; }
            .badge-gk { background: #fbbf24; }
            .badge-ref { background: #ef4444; }
            .badge-ball { background: #ffffff; color: black; }
        </style>
    </head>
    <body>
        <h1>📊 Auditoría de Detección (Fase Final)</h1>
        <p class="meta">Analizando Player, Goalkeeper, Referee y Ball.</p>
        <div class="grid">
    """
    
    # Mapeo de clases (según data.yaml de hybrid_dataset)
    # 0: player, 1: goalkeeper, 2: referee, 3: ball
    class_map = {0: "Player", 1: "GK", 2: "Referee", 3: "Ball"}
    
    for img_path in images_to_process:
        # Predecir con el modelo
        results = model.predict(img_path, imgsz=960)[0] # Usamos imgsz de entrenamiento
        
        # Conteo para el resumen de la card
        counts = {"Player": 0, "GK": 0, "Referee": 0, "Ball": 0}
        for c in results.boxes.cls:
            cls_name = class_map.get(int(c), "Unknown")
            if cls_name in counts: counts[cls_name] += 1
            
        # Imagen con predicción
        img_pred = results.plot(boxes=True, masks=True) 
        
        # Convertir a base64
        b64_pred = get_image_base64(img_pred)
        
        html_content += f"""
        <div class="card">
            <div class="meta">Imagen: {img_path.name} ({img_path.parent.name})</div>
            <div class="summary-box">
                <span class="badge badge-team1">Jugadores: {counts['Player']}</span>
                <span class="badge badge-gk">Portero: {counts['GK']}</span>
                <span class="badge badge-ref">Árbitro: {counts['Referee']}</span>
                <span class="badge badge-ball">Balón: {counts['Ball']}</span>
            </div>
            <div class="img-container">
                <img src="data:image/jpeg;base64,{b64_pred}">
            </div>
            <div style="margin-top: 10px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                   <input type="checkbox"> <span class="meta">Revisar manualmente</span>
                </div>
                <div class="meta">Latencia: {results.speed['inference']:.1f}ms</div>
            </div>
        </div>
        """
        print(f"✅ Procesada {img_path.name}")

    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"✨ Informe generado en: {os.path.abspath(output_html)}")

if __name__ == "__main__":
    BEST_MODEL = "C:/apped/ml/training/runs/detect/players_v3/weights/best.pt"
    DATASET = "C:/apped/data/datasets/hybrid_dataset"
    
    if os.path.exists(BEST_MODEL):
        generate_report(BEST_MODEL, DATASET)
    else:
        print(f"❌ No se encontró el modelo {BEST_MODEL}.")
