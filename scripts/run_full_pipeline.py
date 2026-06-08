import subprocess
import sys
import os
import time

def run_step(command, description):
    print("\n" + "="*60)
    print(f"STEP: {description}")
    print("="*60)
    
    python_exe = sys.executable
    full_command = f'"{python_exe}" {command}'
    
    start_time = time.time()
    result = subprocess.run(full_command, shell=True)
    end_time = time.time()
    
    if result.returncode != 0:
        print(f"\nERROR: Step '{description}' failed with exit code {result.returncode}")
        sys.exit(result.returncode)
        
    print(f"\nSUCCESS: Completed in {end_time - start_time:.1f}s")

def main():
    video_path = "app/videos/test_5min.mp4"
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    
    print("EDApp Full Pipeline: UCMCTrack + GTA-Link Refinement")
    print(f"Target Video: {video_path}")
    
    # 1. Raw Analysis (UCMCTrack)
    run_step(f"run_benchmark.py {video_path}", "UCMCTrack Analysis & MOT Export")
    
    # 2. Feature Extraction (ReID)
    run_step(f"scripts/extract_reid_features.py {video_path}", "OSNet ReID Feature Extraction")
    
    # 3. GTA-Link Refinement
    run_step("scripts/gta_link_refine.py", "Global Tracklet Association (GTA-Link)")
    
    # 4. Final Summary
    print("\n" + "#"*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("#"*60)
    print(f"Final Refined Tracks: output/tracks_refined.txt")
    print("Check output/walkthrough_gtalink_poc.md for detailed metrics.")

if __name__ == "__main__":
    main()
