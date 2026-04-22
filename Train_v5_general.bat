@echo off
cd /d %~dp0
echo [V5 GENERAL] Initializing training environment...
call venv_training\Scripts\activate.bat

echo [V5 GENERAL] Preparing dataset split...
python ml\training\prepare_v5_splits.py

echo [V5 GENERAL] Starting YOLO training...
yolo detect train model=models/football-player-detection_akram.pt data=data/datasets/v5_general.yaml epochs=30 imgsz=1024 batch=1 workers=0 project=runs/v5_general name=train

if exist runs\v5_general\train\weights\best.pt (
    echo [V5 GENERAL] Success! Copying weights to models/players_v5.pt
    copy /Y runs\v5_general\train\weights\best.pt models\players_v5.pt
    echo [V5 GENERAL] Extracting metrics...
    python ml\training\scripts\extract_metrics.py runs\v5_general\train\results.csv general
    echo [V5 GENERAL] Chaining to Ball Training...
    call Train_v5_ball.bat
) else (
    echo [ERROR] Training failed or best.pt not found. Sequence stopped.
)

echo [V5 GENERAL] Process complete.
timeout /t 5

