import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.absolute()))
from app.modules.performance_engine import PerformanceEngine

TRACKS_FILE = "output/tracks_raw.txt"
EVENTS_FILE = "output/events.json"

def main():
    if not Path(TRACKS_FILE).exists() or not Path(EVENTS_FILE).exists():
        print("Faltan archivos en output/")
        return

    track_history = {}
    fps = 5.0 # Assuming 5fps since run_demo_analysis.py uses 0.2 sample rate
    
    # Read tracks
    with open(TRACKS_FILE, "r") as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 12: continue
            
            frame = int(parts[0])
            tid = int(parts[1])
            x = float(parts[2]) + float(parts[4])/2 # center x
            y = float(parts[3]) + float(parts[5])/2 # center y
            
            minute = frame / fps / 60.0
            
            # Naive pitch conversion
            px = max(0, min(105, x / 1280.0 * 105.0))
            py = max(0, min(68, y / 720.0 * 68.0))
            
            if tid not in track_history:
                track_history[tid] = {
                    "history_minute": [],
                    "pitch_x": [],
                    "pitch_y": []
                }
                
            track_history[tid]["history_minute"].append(minute)
            track_history[tid]["pitch_x"].append(px)
            track_history[tid]["pitch_y"].append(py)

    # Calculate stats
    engine = PerformanceEngine()
    stats = engine.process_all_tracks(track_history)

    # Inject into events.json
    with open(EVENTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    data["physical_stats"] = stats
    
    with open(EVENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        
    print(f"Inyectados datos físicos para {len(stats)} tracks en {EVENTS_FILE}")

if __name__ == "__main__":
    main()
