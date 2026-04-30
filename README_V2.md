# EDApp v2 — Advanced Football Match Analysis

**EDApp v2** is a re-engineered, production-ready football video analysis platform. It transitions from basic tracking to professional-grade performance analytics by integrating SOTA computer vision and rule-based intelligence.

---

## 🚀 Key Features

- **Professional Metrics**: Real-world distance (km), top speed (km/h), and sprint detection.
- **Tactical Event Detection**: Automatic detection of passes, possession changes, and ball recoveries.
- **Hybrid Team Classification**: Unsupervised player clustering using SigLIP embeddings and adaptive HSV tracking.
- **Automatic Pitch Calibration**: Uses deep learning to detect pitch keypoints and compute homography for meter-accurate mapping.
- **Ball Trajectory Refinement**: Advanced interpolation to handle fast-moving ball sequences.

---

## 🛠️ Installation

### 1. Requirements
- Python 3.10+
- NVIDIA GPU (RTX 20-series or higher recommended)
- CUDA 12.1+

### 2. Setup Environment
```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install deep learning weights (ensure models are in /models or /assets/weights)
# Models required: players.pt, ball_specialist.pt, pitch.pt
```

---

## 🏃 How to Run

### Start the Analysis Platform
```powershell
streamlit run app/app.py
```

### Workflow
1. **Upload**: Select a match video (VEO or Broadcast).
2. **Calibrate**: The system automatically detects pitch keypoints.
3. **Identify**: Mark players manually (optional) or let the SigLIP classifier group them.
4. **Analyze**: Run the SOTA pipeline to extract metrics and events.
5. **Report**: View the heatmaps, tactical maps, and physical stats in the dashboard.

---

## 🧪 Testing & Audit

To verify the system integrity and metric accuracy without launching the UI, use the Audit Script:

```powershell
python scripts/audit_analysis.py
```

This script will:
1. Load the SOTA multi-model detector.
2. Process a sample clip through the full v2 pipeline.
3. Print a quality report including detected players, events, and physical stats.

---

## 📂 Project Structure (V2)

```
C:\D\EDApp\
├── app/
│   ├── modules/
│   │   ├── detector.py          # Multi-model YOLO logic
│   │   ├── tracker.py           # ByteTrack + Motion Compensation
│   │   ├── coordinates.py       # Px to Meters transformation (New)
│   │   ├── performance_engine.py # Physical stats calculator (New)
│   │   ├── event_engine.py      # Rule-based event detector (New)
│   │   └── team_classifier.py   # SigLIP + HSV clustering
│   └── pages/                   # Streamlit UI Dashboards
├── core/
│   └── pipeline/
│       └── video_pipeline.py    # Main Orchestrator (Refactored)
├── models/                      # YOLO Weights (.pt)
├── scripts/
│   └── audit_analysis.py        # System verification (New)
└── requirements.txt
```

---

*Developed for Professional Football Analytics*
