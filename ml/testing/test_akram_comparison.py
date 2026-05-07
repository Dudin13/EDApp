import cv2
import os
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np

def test_models():
    video_path = "data/samples/test_5min.mp4"
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return

    # Models
    models_config = {
        "Baseline": {
            "path": "models/players.pt",
            "classes": [0, 1, 2], # player, goalkeeper, referee
            "ball_class": 3
        },
        "Akram_Players": {
            "path": "models/football-player-detection_akram.pt",
            "classes": [1, 2, 3], # goalkeeper, player, referee
            "ball_class": 0
        },
        "Akram_Ball": {
            "path": "models/football-ball-detection_akram.pt",
            "classes": [],
            "ball_class": 0
        },
        "Akram_Pitch": {
            "path": "models/football-pitch-detection_akram.pt",
            "task": "pose"
        }
    }

    # Load models
    loaded_models = {}
    for name, cfg in models_config.items():
        if os.path.exists(cfg["path"]):
            loaded_models[name] = YOLO(cfg["path"])
            print(f"Loaded {name}")
        else:
            print(f"Warning: {name} model not found at {cfg['path']}")

    cap = cv2.VideoCapture(video_path)
    start_offset = 3000 # Jump into actual gameplay
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_offset)
    
    frame_count = 0
    max_frames = 300
    
    stats = {name: {"total_players": 0, "ball_frames": 0} for name in loaded_models if name != "Akram_Pitch"}
    
    output_dir = "runs/comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    
    snapshot_indices = [1, 150, 290] # Relative to start_offset
    
    pbar = tqdm(total=max_frames, desc="Processing frames")
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Inference for statistical comparisons
        for name, model in loaded_models.items():
            if name == "Akram_Pitch":
                continue
                
            results = model.predict(frame, conf=0.15, verbose=False)[0] # Lower confidence for ball coverage
            
            # Count players (player, gk, ref)
            player_classes = models_config[name]["classes"]
            boxes = results.boxes
            counts = 0
            has_ball = False
            
            ball_cls = models_config[name]["ball_class"]
            
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Use 0.3 for players statistics, lower for ball coverage
                if cls in player_classes and conf >= 0.3:
                    counts += 1
                if cls == ball_cls:
                    has_ball = True
            
            stats[name]["total_players"] += counts
            if has_ball:
                stats[name]["ball_frames"] += 1
            
            # Save snapshots
            if frame_count in snapshot_indices:
                out_img = results.plot()
                cv2.imwrite(f"{output_dir}/{name}_frame_{frame_count}.jpg", out_img)
                
        # Inference for Pitch Model (Pose)
        if "Akram_Pitch" in loaded_models and frame_count in snapshot_indices:
            pitch_results = loaded_models["Akram_Pitch"].predict(frame, conf=0.3, verbose=False)[0]
            pitch_img = pitch_results.plot()
            cv2.imwrite(f"{output_dir}/Akram_Pitch_frame_{frame_count}.jpg", pitch_img)

        frame_count += 1
        pbar.update(1)
        
    cap.release()
    pbar.close()
    
    # Report
    print("\n" + "="*50)
    print(f"COMPARISON REPORT (Frames 0-{frame_count-1})")
    print("="*50)
    print(f"{'Model':<20} | {'Avg Players':<12} | {'Ball Coverage':<12}")
    print("-" * 50)
    
    for name, data in stats.items():
        avg_p = data["total_players"] / frame_count
        ball_cov = (data["ball_frames"] / frame_count) * 100
        print(f"{name:<20} | {avg_p:<12.2f} | {ball_cov:<11.1f}%")
    print("="*50)
    print(f"Snapshots saved to: {output_dir}")

if __name__ == "__main__":
    test_models()
