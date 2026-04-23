@echo off
cd /d %~dp0
echo [V6 GENERAL] Initializing training environment...
call venv_training\Scripts\activate.bat

echo [V6 GENERAL] Preparing dataset split...
python ml\training\prepare_v6_splits.py

echo [V6 GENERAL] Starting YOLO training...
echo Model: models/players.pt
echo Config: Epochs=30, imgsz=1024, batch=1, workers=0
yolo detect train model=models/players.pt data=data/datasets/v6_general.yaml epochs=30 imgsz=1024 batch=1 workers=0 project=runs/v6_general name=train

if exist runs\v6_general\train\weights\best.pt (
    echo [V6 GENERAL] Success! Copying weights to models/players_v6.pt
    copy /Y runs\v6_general\train\weights\best.pt models\players_v6.pt
    if exist ml\training\scripts\extract_metrics.py (
        python ml\training\scripts\extract_metrics.py runs\v6_general\train\results.csv v6_general
    )
) else if exist runs\detect\runs\v6_general\train\weights\best.pt (
    echo [V6 GENERAL] Success (found in detect/runs)! Copying weights to models/players_v6.pt
    copy /Y runs\detect\runs\v6_general\train\weights\best.pt models\players_v6.pt
    if exist ml\training\scripts\extract_metrics.py (
        python ml\training\scripts\extract_metrics.py runs\detect\runs\v6_general\train\results.csv v6_general
    )
) else (
    echo [ERROR] Training failed or best.pt not found.
)

echo [V6 GENERAL] Process complete.
pause
