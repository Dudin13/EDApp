@echo off
cd /d %~dp0
echo [V7 GENERAL] Initializing training environment...
call venv_training\Scripts\activate.bat

echo [V7 GENERAL] Preparing dataset split (v7)...
python ml\training\prepare_v7_splits.py

echo [V7 GENERAL] Starting YOLO training...
echo Model base: models/players.pt
echo Config: Epochs=30, imgsz=1024, batch=1, workers=0
yolo detect train model=models/players.pt data=data/datasets/v7_general.yaml epochs=30 imgsz=1024 batch=1 workers=0 project=runs/v7_general name=train %*

if exist runs\v7_general\train\weights\best.pt (
    echo [V7 GENERAL] Success! Copying weights to models/players_v7.pt
    copy /Y runs\v7_general\train\weights\best.pt models\players_v7.pt
    if exist ml\training\scripts\extract_metrics.py (
        python ml\training\scripts\extract_metrics.py runs\v7_general\train\results.csv v7_general
    )
) else if exist runs\detect\runs\v7_general\train\weights\best.pt (
    echo [V7 GENERAL] Success (found in detect/runs)! Copying weights to models/players_v7.pt
    copy /Y runs\detect\runs\v7_general\train\weights\best.pt models\players_v7.pt
    if exist ml\training\scripts\extract_metrics.py (
        python ml\training\scripts\extract_metrics.py runs\detect\runs\v7_general\train\results.csv v7_general
    )
) else (
    echo [ERROR] Training failed or best.pt not found.
)

echo [V7 GENERAL] Process complete.
