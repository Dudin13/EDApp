import cv2
import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path to import app modules
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from app.modules.calibration_auto import AutoCalibrator
except ImportError as e:
    print(f"Error importing AutoCalibrator: {e}")
    sys.exit(1)

def main():
    video_path = "data/samples/test_5min.mp4"
    model_path = "models/football-pitch-detection_akram.pt"
    
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Initializing AutoCalibrator with Akram's model: {model_path}")
    calibrator = AutoCalibrator(model_path=model_path)
    
    output_dir = "runs/pitch_integration_test"
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    # Test frames: lineup (start) and game (frame 3000)
    test_frames = [150, 3000, 6000]
    
    for f_idx in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        success, frame = cap.read()
        if not success:
            print(f"Could not read frame {f_idx}")
            continue
            
        print(f"Calibrating frame {f_idx}...")
        is_calibrated = calibrator.calibrate(frame, conf=0.3)
        
        if is_calibrated:
            # 1. Draw keypoints
            annotated_frame = calibrator.draw_keypoints(frame)
            
            # 2. Draw minimap (empty detections just to see field lines)
            minimap = calibrator.draw_pitch_minimap([], width=600, height=400)
            
            # Combine image and minimap for visual check
            # Resize minimap to fit height of frame if needed
            h, w = frame.shape[:2]
            mh, mw = minimap.shape[:2]
            
            # Place minimap in corner
            frame_with_minimap = annotated_frame.copy()
            frame_with_minimap[0:mh, 0:mw] = minimap
            
            out_path = os.path.join(output_dir, f"calibration_f{f_idx}.jpg")
            cv2.imwrite(out_path, frame_with_minimap)
            print(f"Saved results to {out_path}")
        else:
            print(f"Calibration failed for frame {f_idx}")
            # Save raw frame to debug
            cv2.imwrite(os.path.join(output_dir, f"failed_f{f_idx}.jpg"), frame)

    cap.release()
    print("\nCalibration test complete.")
    print(f"Check results in {output_dir}")

if __name__ == "__main__":
    main()
