import os
import cv2
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import base64

def get_image_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def generate_report(model_path, dataset_path, output_html="review_report.html"):
    print(f"🚀 Generando informe de revisión final...")
    model = YOLO(model_path)
    
    # Rutas
    img_dir = Path(dataset_path) / "valid" / "images"
    if not img_dir.exists():
        img_dir = Path(dataset_path) / "train" / "images" # Fallback a train si valid es pequeño
        
    images = list(img_dir.glob("*.jpg"))[:50] # Revisamos una muestra de 50 para no saturar el HTML
    
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
            .class-player { border: 2px solid #00d4aa; }
            .class-goalkeeper { border: 2px solid #fbbf24; }
            .class-referee { border: 2px solid #f87171; }
            .class-ball { border: 2px solid #60a5fa; }
            .summary-box { display: flex; gap: 5px; flex-wrap: wrap; margin-bottom: 5px; }
            .badge-team1 { background: #00d4aa; }
            .badge-team2 { background: #3b82f6; }
            .badge-gk { background: #fbbf24; }
            .badge-ref { background: #ef4444; }
            .badge-ball { background: #ffffff; color: black; }
        </style>
    </head>
    <body>
        <h1>📊 Auditoría de Siluetas y Detección (Fase Final)</h1>
        <p class="meta">Analizando Portero, Equipos, Balón y Árbitro.</p>
        <div class="grid">
    """
    
    # Mapeo de clases (según data.yaml)
    # 0: player, 1: goalkeeper, 2: referee, 3: ball
    class_map = {0: "Player", 1: "GK", 2: "Referee", 3: "Ball"}
    
    for img_path in images:
        # Predecir con el modelo
        results = model.predict(img_path, imgsz=640)[0]
        
        # Conteo para el resumen de la card
        counts = {"Player": 0, "GK": 0, "Referee": 0, "Ball": 0}
        for c in results.boxes.cls:
            cls_name = class_map.get(int(c), "Unknown")
            if cls_name in counts: counts[cls_name] += 1
            
        # Imagen con predicción (Siluetas + Boxes)
        img_pred = results.plot(boxes=True, masks=True) 
        
        # Convertir a base64 para HTML autocontenido
        b64_pred = get_image_base64(img_pred)
        
        html_content += f"""
        <div class="card">
            <div class="meta">Imagen: {img_path.name}</div>
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
    # Estos paths se ajustarán dinámicamente al terminar el entreno
    BEST_MODEL = "train_yolo/runs/detect/EDudin_v1/weights/best.pt"
    DATASET = "04_Datasets_Entrenamiento/hybrid_dataset"
    
    if os.path.exists(BEST_MODEL):
        generate_report(BEST_MODEL, DATASET)
    else:
        print(f"❌ No se encontró el modelo {BEST_MODEL}. Espera a que termine el entrenamiento.")
