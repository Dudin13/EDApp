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
    MODELS_DIR: Path = BASE_DIR / "models"
    OUTPUT_DIR: Path = BASE_DIR / "app" / "output"
    
    # Model Filenames
    PLAYER_MODEL_NAME: str = "players.pt"
    BALL_MODEL_NAME: str = "ball.pt"
    PITCH_MODEL_NAME: str = "pitch.pt"
    
    # Computed Model Paths
    @property
    def PLAYER_MODEL_PATH(self) -> str:
        return str(self.MODELS_DIR / self.PLAYER_MODEL_NAME)
        
    @property
    def BALL_MODEL_PATH(self) -> str:
        return str(self.MODELS_DIR / self.BALL_MODEL_NAME)
        
    @property
    def PITCH_MODEL_PATH(self) -> str:
        return str(self.MODELS_DIR / self.PITCH_MODEL_NAME)

    # ML Hyperparameters
    DETECTION_CONF: float = 0.35
    TRACKING_BUFFER: int = 30
    BATCH_SIZE: int = 1
    
    # Team Identification Anchors (HSV)
    TEAM_A_COLOR_HSV: tuple = (30, 150, 150)  # Amarillo (Local)
    TEAM_B_COLOR_HSV: tuple = (110, 150, 150) # Azul (Visitante)

    # Hardware
    DEVICE: str = "cuda" # 'cpu', 'cuda', 'mps'

    # External APIs & Tools
    ROBOFLOW_API_KEY: Optional[str] = None
    SOCCERNET_EMAIL: Optional[str] = None
    SOCCERNET_PASSWORD: Optional[str] = None
    FFMPEG_PATH: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

# Create instance
settings = Settings()

# Ensure directories exist
for path in [settings.DATA_DIR, settings.ASSETS_DIR, settings.MODELS_DIR, settings.OUTPUT_DIR]:
    path.mkdir(parents=True, exist_ok=True)
