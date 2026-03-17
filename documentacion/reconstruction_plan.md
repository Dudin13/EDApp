# Professional Reconstruction Plan: EDudin Football Analytics

## 1. Identified Engineering Problems

| Problem Category | Description | Impact |
| :--- | :--- | :--- |
| **Architectural** | No separation between UI and core logic. Logic is scattered across `app/modules`. | Low maintainability; hard to test. |
| **Hardcoding** | Absolute paths like `c:/apped` and `c:/D/EDApp` are used everywhere. | System fails on any other machine. |
| **Code Quality** | Lack of Type Hints, inconsistent logging, and few try-except blocks in critical ML loops. | Fragile production environment; hard to debug. |
| **Incomplete Modules** | `IdentityReader` and `EventSpotterTDEED` are mere skeletons. | Missing core tracking and event features. |
| **Performance** | Optical Flow and YOLO inference are synchronous and sequential. | Slow processing speeds (~1-5 FPS). |
| **Dependency Management** | `requirements.txt` was empty or incomplete. | Difficult deployment and environment replication. |

---

## 2. Professional Project Structure (Modern MVP)

I have designed the following structure to ensure scalability, testability, and clarity.

```
project/
├── app/                        # UI Entry & View Logic
│   ├── main.py                 # Streamlit main entry point
│   ├── pages/                  # Multipage dashboard logic
│   └── components/             # Reusable UI widgets (cards, plots)
│
├── core/                       # System Orchestration
│   ├── config/
│   │   └── settings.py         # Pydantic-Settings & ENV management
│   ├── pipeline/
│   │   └── video_pipeline.py   # Component-based orchestrator
│   ├── logger.py               # Centralized Loguru/Logging setup
│   └── exceptions.py           # Custom Error types
│
├── ai/                         # Machine Learning Modules
│   ├── detector/               # YOLO Discovery & Inference
│   ├── tracker/                # ByteTrack + Motion Compensation
│   ├── classification/         # SigLIP Team Identification
│   ├── identity/               # Dorsal OCR (PARSeq/EasyOCR)
│   └── events/                 # T-DEED & Geometric Spotting
│
├── analytics/                  # Post-processing & Insights
│   ├── spatial.py              # Heatmap & Density logic
│   └── match_stats.py          # Possession/Event metrics
│
├── assets/                     # Weights, Templates, Icons
├── data/                       # Input videos & Output artifacts
├── scripts/                    # setup.py, download_weights.py
├── tests/                      # Pytest unit & integration tests
├── requirements.txt
└── .env.example
```

---

## 3. Configuration System (`core/config/settings.py`)

Eliminating hardcoded paths using `pathlib` and environment variables.

```python
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    WEIGHTS_DIR: Path = BASE_DIR / "assets" / "weights"
    
    # Model Paths
    PLAYER_MODEL: str = "detect_players.pt"
    BALL_MODEL: str = "detect_ball.pt"
    PITCH_MODEL: str = "pose_field.pt"
    
    # ML Hyperparams
    DETECTION_CONF: float = 0.35
    BATCH_SIZE: int = 4
    
    # Device
    DEVICE: str = "cuda" # 'cpu', 'cuda', 'mps'
    
    class Config:
        env_file = ".env"

settings = Settings()
```

---

## 4. Stable Video Processing Pipeline

The new pipeline follows a strictly sequential yet decoupled flow:

1.  **Ingestion**: `cv2.VideoCapture` + Frame Sampling.
2.  **Motion Estimation**: `OpticalFlow` between $F_t$ and $F_{t-1}$.
3.  **Inference**:
    - **Global**: Detect Players/Referees (YOLO).
    - **Zoom**: Detect Ball at 1280px (YOLO).
    - **Pose**: Field Keypoints (YOLO-Pose).
4.  **Temporal Matching**:
    - `ByteTrack` assigns/keeps Track IDs.
    - Camera motion offsets applied to Kalman states.
5.  **Classification**:
    - Torso crops → **SigLIP** Embedding → KMeans/UMAP labeling.
6.  **Analytics**:
    - Pixel → Meters via **Homography**.
    - Velocity filtering for Ball.
    - Geometric event spotting (Tiro, Pase).
7.  **Serialization**: Save results as JSON + Trigger UI update.

---

## 5. Implementations for Skeleton Modules

### IdentityReader (Dorsal OCR)
We will implement a two-step process:
1.  **Dorsal Detection**: Focused YOLOv8n to find the number on the jersey.
2.  **OCR**: Use `EasyOCR` or `PARSeq` on the detected dorsal crop.

### EventSpotter (T-DEED Integration)
Shift from naive distance rules to a **Temporal Feature Window**:
- Extract features from 30-frame windows.
- Run a lightweight MLP or Transformer to classify the temporal sequence (Goal, Foul, Corner).

---

## 6. Stability & Performance Improvements

- **GPU Manager**: Shared model instance in VRAM to avoid reloading.
- **Batching**: Process 4-8 frames in a single YOLO forward pass if memory allows.
- **Async Sink**: Saving statistics and frames to disk on a separate thread to avoid blocking the main processing loop.
- **Logging**: Using `loguru` for structured, color-coded logs that rotate automatically.
