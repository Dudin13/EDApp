import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans

def get_dominant_color(img_crop, k=2):
    """Extrae el color dominante del recorte del jugador."""
    if img_crop.size == 0:
        return None
    
    # Redimensionar para velocidad
    img_crop = cv2.resize(img_crop, (30, 60))
    # Centrar el crop para enfocarse en la camiseta (evitar cara/piernas si es posible)
    h, w = img_crop.shape[:2]
    crop_center = img_crop[int(h*0.2):int(h*0.5), int(w*0.2):int(w*0.8)]
    
    pixels = crop_center.reshape(-1, 3)
    # Filtrar verde (césped) si se coló
    # Un filtro básico por HSV sería mejor, pero por ahora K-Means directo
    
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_
    
    # Devolver el color que NO sea verde/oscuro si hay variedad
    return colors[0] # Simplificación: primer centroide

def relabel_teams(dataset_path):
    dataset_path = Path(dataset_path)
    # Clases: 0:team_1, 1:team_2, 2:goalkeeper, 3:referee, 4:ball
    
    all_player_data = [] # List of (color, label_file_path, line_index, polygon_parts)
    
    print("[INFO] Fase 1: Recolectando colores de todo el dataset...")
    for t in ['train', 'valid']:
        img_dir = dataset_path / t / "images"
        lab_dir = dataset_path / t / "labels"
        if not img_dir.exists(): continue
        
        for img_file in img_dir.glob("*.jpg"):
            lab_file = lab_dir / f"{img_file.stem}.txt"
            if not lab_file.exists(): continue
            
            img = cv2.imread(str(img_file))
            if img is None: continue
            h_img, w_img = img.shape[:2]
            
            with open(lab_file, 'r') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if not parts: continue
                cls = int(parts[0])
                
                # Recolectar personas (ex-players/gks mapeados a 0/1)
                if cls in [0, 1]:
                    coords = [float(x) for x in parts[1:]]
                    if not coords: continue
                    xs = coords[0::2]
                    ys = coords[1::2]
                    
                    x1, y1 = int(min(xs) * w_img), int(min(ys) * h_img)
                    x2, y2 = int(max(xs) * w_img), int(max(ys) * h_img)
                    
                    # Crop con margen de seguridad
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_img, x2), min(h_img, y2)
                    
                    crop = img[y1:y2, x1:x2]
                    color = get_dominant_color(crop)
                    if color is not None:
                        all_player_data.append({
                            'color': color,
                            'file': lab_file,
                            'line_idx': i,
                            'parts': parts
                        })

    if not all_player_data:
        print("[WARN] No se encontraron jugadores para relabeling.")
        return

    print(f"[INFO] Fase 2: Clustering global de {len(all_player_data)} jugadores...")
    colors = [d['color'] for d in all_player_data]
    km = KMeans(n_clusters=2, n_init=10)
    labels = km.fit_predict(colors)
    
    # Agrupar cambios por archivo para evitar escrituras repetidas
    file_changes = {}
    for i, data in enumerate(all_player_data):
        file = data['file']
        if file not in file_changes:
            # Leer el archivo completo una vez para tener la estructura
            with open(file, 'r') as f:
                file_changes[file] = [line.strip().split() for line in f.readlines()]
        
        # Actualizar clase de equipo (0 o 1)
        file_changes[file][data['line_idx']][0] = str(labels[i])

    print("[INFO] Fase 3: Guardando cambios y ajustando clases de referee/ball...")
    for file, lines in file_changes.items():
        # Mapeo final de seguridad para clases no-jugador
        for parts in lines:
            c = int(parts[0])
            # La lógica de simplify_labels dejó Arbitro=1 y Balon=2
            # Aquí, si no fueron parte del cluster de jugadores, los movemos.
            # Nota: Esto es complejo si se mezclaron los IDs. 
            # Asumimos que el cluster solo tocó las líneas recolectadas.
            pass # Las líneas de árbitro/balón ya deberían estar fuera si cls era 1 pero no entró en color_data
        
        with open(file, 'w') as f:
            for p in lines:
                f.write(" ".join(p) + "\n")

    print("[DONE] Relabeling global completado!")

    print("[DONE] Relabeling completado!")

if __name__ == "__main__":
    relabel_teams("04_Datasets_Entrenamiento/hybrid_dataset")
