@echo off
cd /d %~dp0
echo [V5 BALL] Initializing training environment...
call venv_training\Scripts\activate.bat

echo [V5 BALL] Preparing dataset split...
python ml\training\prepare_v5_splits.py

echo [V5 BALL] Starting YOLO training...
yolo detect train model=models/players.pt data=data/datasets/v5_ball.yaml epochs=30 imgsz=1024 batch=1 workers=0 project=runs/v5_ball name=train

if exist runs\v5_ball\train\weights\best.pt (
    echo [V5 BALL] Success! Copying weights to models/ball_specialist.pt
    copy /Y runs\v5_ball\train\weights\best.pt models\ball_specialist.pt
    echo [V5 BALL] Extracting metrics...
    python ml\training\scripts\extract_metrics.py runs\v5_ball\train\results.csv ball
) else (
    echo [ERROR] Training failed or best.pt not found.
    echo Press any key to proceed to shutdown...
    pause
)

echo [V5 BALL] Sequence complete. All trainings finished.
echo System will shutdown in 60 seconds.
shutdown /s /t 60

