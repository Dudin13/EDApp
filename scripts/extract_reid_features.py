import os
import cv2
import pickle
import torch
import numpy as np
from tqdm import tqdm
import torch
# Monkeypatch torch.load to handle PyTorch 2.6+ weights_only=True default for old checkpoints
orig_load = torch.load
def hooked_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return orig_load(*args, **kwargs)
torch.load = hooked_load

import torchreid
from PIL import Image

import sys

def main():
    # Paths
    video_path = sys.argv[1] if len(sys.argv) > 1 else "app/videos/test_5min.mp4"
    mot_path = "output/tracks_raw.txt"
    model_path = "models/osnet_sports.pth"
    output_pkl = "output/reid_features.pkl"
    
    if not os.path.exists(mot_path):
        print(f"Error: {mot_path} not found.")
        return

    # Load MOT data
    # format: frame, id, x1, y1, w, h, conf, ...
    try:
        data = np.loadtxt(mot_path, delimiter=',')
    except Exception as e:
        print(f"Error loading MOT data: {e}")
        return
    
    if data.ndim == 1:
        data = data.reshape(1, -1)

    # Group by frame for efficiency
    frames_dict = {}
    for line in data:
        f = int(line[0])
        if f not in frames_dict:
            frames_dict[f] = []
        frames_dict[f].append(line)
        
    # Setup ReID extractor
    print(f"Loading ReID model from {model_path}...")
    extractor = torchreid.utils.FeatureExtractor(
        model_name='osnet_x1_0',
        model_path=model_path,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    sample_rate = 1 # from config
    frame_interval = int(fps * sample_rate)
    
    features_data = {} # (tid, frame) -> feature

    # Group by ID instead of frame to optimize extraction
    tracks_by_id = {}
    for line in data:
        tid = int(line[1])
        if tid not in tracks_by_id:
            tracks_by_id[tid] = []
        tracks_by_id[tid].append(line)
        
    # Select at most 3 frames per tracklet (start, middle, end)
    selected_crops_meta = [] # (f_idx, tid, line)
    for tid, lines in tracks_by_id.items():
        lines.sort(key=lambda x: int(x[0]))
        if len(lines) <= 3:
            selected = lines
        else:
            selected = [lines[0], lines[len(lines)//2], lines[-1]]
            
        for line in selected:
            selected_crops_meta.append((int(line[0]), tid, line))
            
    # Group back by frame to minimize video seeking
    frames_dict = {}
    for f_idx, tid, line in selected_crops_meta:
        if f_idx not in frames_dict:
            frames_dict[f_idx] = []
        frames_dict[f_idx].append(line)
        
    sorted_frames = sorted(frames_dict.keys())
    print(f"Processing {len(sorted_frames)} frames (optimized for {len(tracks_by_id)} tracklets)...")
    
    current_pos = -1
    for f_idx in tqdm(sorted_frames, desc="Extracting features"):
        frame_pos = (f_idx - 1) * frame_interval
        
        if frame_pos > current_pos and (frame_pos - current_pos) < 300:
            for _ in range(int(frame_pos - current_pos - 1)):
                cap.grab()
            ret, frame = cap.read()
            current_pos = frame_pos
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            current_pos = frame_pos
            
        if not ret: continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        crops = []
        meta = []
        for line in frames_dict[f_idx]:
            tid = int(line[1])
            x1, y1, w, h = line[2:6]
            H_f, W_f = frame.shape[:2]
            x_min, y_min = int(max(0, x1)), int(max(0, y1))
            x_max, y_max = int(min(W_f, x1 + w)), int(min(H_f, y1 + h))
            crop = frame_rgb[y_min:y_max, x_min:x_max]
            if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5: continue
            crops.append(np.array(crop))
            meta.append((tid, f_idx))
            
        if not crops: continue
        with torch.no_grad():
            feats = extractor(crops).cpu().numpy()
            
        for i, (tid, frame_num) in enumerate(meta):
            feat = feats[i]
            norm = np.linalg.norm(feat)
            if norm > 1e-6: feat /= norm
            features_data[(tid, frame_num)] = feat
            
    cap.release()
    
    os.makedirs("output", exist_ok=True)
    with open(output_pkl, "wb") as f:
        pickle.dump(features_data, f)
        
    print(f"Features saved. Total tracklets processed: {len(tracks_by_id)}")

if __name__ == "__main__":
    main()
