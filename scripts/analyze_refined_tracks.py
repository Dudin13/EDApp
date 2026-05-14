import os
import cv2
import sys
import numpy as np
from collections import defaultdict

sys.path.append(os.path.abspath("app"))
from modules.detector import detect_frame

def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else "app/videos/test_5min.mp4"
    refined_mot = "output/tracks_refined.txt"
    
    if not os.path.exists(refined_mot):
        print("Error: tracks_refined.txt no encontrado.")
        return
        
    data = np.loadtxt(refined_mot, delimiter=',')
    if data.ndim == 1:
        data = data.reshape(1, -1)
        
    tracks_by_id = defaultdict(list)
    tracks_by_frame = defaultdict(list)
    for line in data:
        f = int(line[0])
        tid = int(line[1])
        tracks_by_id[tid].append(line)
        tracks_by_frame[f].append(line)
        
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * 1) # sample_rate = 1
    
    id_info = []
    for tid, lines in tracks_by_id.items():
        lines.sort(key=lambda x: int(x[0]))
        
        # Obtener moda de la primera linea (ya está propagado por gta_link_refine)
        if len(lines[0]) >= 12:
            cls_id = int(lines[0][10])
            team_id = int(lines[0][11])
        else:
            cls_id = 0
            team_id = -1
            
        equipo_str = "Desconocido"
        if cls_id == 2:
            equipo_str = "Árbitro"
        elif cls_id == 3:
            equipo_str = "Balón"
        elif team_id == 0:
            equipo_str = "Equipo A"
        elif team_id == 1:
            equipo_str = "Equipo B"
            
        id_info.append({
            "id": tid,
            "equipo": equipo_str,
            "frames": f"[{int(lines[0][0])} - {int(lines[-1][0])}]",
            "length": len(lines)
        })
        
    print("="*60)
    print("LISTA DE 20 IDs ÚNICOS (GTA-Link)")
    print("="*60)
    id_info.sort(key=lambda x: -x["length"])
    
    equipo_a_count = sum(1 for info in id_info if info['equipo'] == "Equipo A")
    equipo_b_count = sum(1 for info in id_info if info['equipo'] == "Equipo B")
    
    for info in id_info:
        print(f"ID {info['id']:>2} | {info['equipo']:<10} | Frames: {info['frames']:<10} | Longitud: {info['length']} frames")
        
    avg_players_per_frame = sum(len(lines) for lines in tracks_by_frame.values()) / max(1, len(tracks_by_frame))
    print("-" * 60)
    print(f"Total Equipo A: {equipo_a_count}")
    print(f"Total Equipo B: {equipo_b_count}")
    print(f"Media de jugadores detectados por frame: {avg_players_per_frame:.2f}")
    print("=" * 60)
    # 3. Generar 3 capturas
    frames_to_capture = [5, 75, 140] # MOT frames
    colors = {
        "Equipo A": (0, 0, 255), 
        "Equipo B": (255, 0, 0), 
        "Árbitro": (0, 255, 255), 
        "Balón": (255, 255, 0),
        "Desconocido": (255, 255, 255)
    }
    
    os.makedirs("output/screenshots", exist_ok=True)
    
    for f_mot in frames_to_capture:
        cap.set(cv2.CAP_PROP_POS_FRAMES, (f_mot - 1) * frame_interval)
        ret, frame = cap.read()
        if not ret: continue
        
        for line in tracks_by_frame.get(f_mot, []):
            tid = int(line[1])
            x1, y1, w, h = line[2:6]
            
            info = next((i for i in id_info if i["id"] == tid), None)
            eq = info["equipo"] if info else "Desconocido"
            color = colors.get(eq, (255, 255, 255))
            
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x1+w), int(y1+h)), color, 2)
            cv2.putText(frame, f"ID:{tid}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        out_path = f"output/screenshots/frame_{f_mot}.jpg"
        cv2.imwrite(out_path, frame)
        print(f"Captura guardada en: {out_path}")
        
    cap.release()

if __name__ == "__main__":
    main()
