from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional
import os

class Settings(BaseSettings):
    # Base paths
    # Assuming this file is in project_root/core/config/settings.py
    # So .parent.parent.parent is project_root
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    
    # Directories
    DATA_DIR: Path = BASE_DIR / "data"
    ASSETS_DIR: Path = BASE_DIR / "assets"
    WEIGHTS_DIR: Path = ASSETS_DIR / "weights"
    OUTPUT_DIR: Path = BASE_DIR / "app" / "output"
    
    # Model Filenames
    PLAYER_MODEL_NAME: str = "detect_players.pt"
    BALL_MODEL_NAME: str = "detect_ball.pt"
    PITCH_MODEL_NAME: str = "pose_field.pt"
    
    # Computed Model Paths
    @property
    def PLAYER_MODEL_PATH(self) -> str:
        return str(self.WEIGHTS_DIR / self.PLAYER_MODEL_NAME)
        
    @property
    def BALL_MODEL_PATH(self) -> str:
        return str(self.WEIGHTS_DIR / self.BALL_MODEL_NAME)
        
    @property
    def PITCH_MODEL_PATH(self) -> str:
        return str(self.WEIGHTS_DIR / self.PITCH_MODEL_NAME)

    # ML Hyperparameters
    DETECTION_CONF: float = 0.35
    TRACKING_BUFFER: int = 30
    BATCH_SIZE: int = 1
    
    # Hardware
    DEVICE: str = "cuda" # 'cpu', 'cuda', 'mps'

    class Config:
        env_file = ".env"
        case_sensitive = True

# Create instance
settings = Settings()

# Ensure directories exist
for path in [settings.DATA_DIR, settings.ASSETS_DIR, settings.WEIGHTS_DIR, settings.OUTPUT_DIR]:
    path.mkdir(parents=True, exist_ok=True)
