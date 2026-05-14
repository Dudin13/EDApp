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

def main():
    # Paths
    video_path = "app/videos/test_5min.mp4"
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
    sample_rate = 2 # from config
    frame_interval = int(fps * sample_rate)
    
    features_data = {} # (tid, frame) -> feature

    sorted_frames = sorted(frames_dict.keys())
    
    print(f"Processing {len(sorted_frames)} frames...")
    
    for f_idx in tqdm(sorted_frames, desc="Extracting features"):
        # Calculate video frame position
        # In MOT export: frame_count+1 was used. frame_count started at 0.
        # So f_idx 1 -> frame_pos 0
        # f_idx 2 -> frame_pos frame_interval
        frame_pos = (f_idx - 1) * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if not ret:
            # Try to read the last frame if pos is slightly off
            if frame_pos >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
                 cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT)-1)
                 ret, frame = cap.read()
            
            if not ret:
                print(f"Warning: Could not read frame {f_idx} at pos {frame_pos}")
                continue
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        crops = []
        meta = []
        for line in frames_dict[f_idx]:
            tid = int(line[1])
            x1, y1, w, h = line[2:6]
            
            # Crop player
            # BBox in MOT is x1, y1, w, h
            # Ensure within frame bounds
            H_f, W_f = frame.shape[:2]
            x_min = int(max(0, x1))
            y_min = int(max(0, y1))
            x_max = int(min(W_f, x1 + w))
            y_max = int(min(H_f, y1 + h))
            
            crop = frame_rgb[y_min:y_max, x_min:x_max]
            if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
                continue
            
            # FeatureExtractor expects str or numpy.ndarray
            crop_np = np.array(crop)
            crops.append(crop_np)
            meta.append((tid, f_idx))
            
        if not crops:
            continue
            
        # Extract features
        # torchreid FeatureExtractor handles the preprocessing (resize, normalize)
        with torch.no_grad():
            feats = extractor(crops) # Tensor (N, 512)
        
        feats = feats.cpu().numpy()
        
        for i, (tid, frame_num) in enumerate(meta):
            feat = feats[i]
            # L2 Normalize
            norm = np.linalg.norm(feat)
            if norm > 1e-6:
                feat /= norm
            features_data[(tid, frame_num)] = feat
            
    cap.release()
    
    # Save features
    os.makedirs("output", exist_ok=True)
    with open(output_pkl, "wb") as f:
        pickle.dump(features_data, f)
        
    print(f"Features saved to {output_pkl}. Total detections processed: {len(features_data)}")

if __name__ == "__main__":
    main()
