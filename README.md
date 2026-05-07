# ⚽ EDApp v2 — Football Match Analysis Platform

**EDApp v2** converts raw match footage into tactical and physical insights using a 3-model computer vision pipeline optimized for VEO panoramic and standard broadcast cameras.

**GitHub:** [https://github.com/Dudin13/EDApp](https://github.com/Dudin13/EDApp)

---

## 🤖 Active Models

| Model | File | Task | Input resolution |
|-------|------|------|-----------------|
| Player Detector | `detect_players.pt` | Players, goalkeepers, referees | 640 px |
| Ball Detector | `detect_ball.pt` | Ball only — panoramic-optimized | 1280 px |
| Pitch Keypoints | `pose_field.pt` | 32 FIFA keypoints for auto-calibration | variable |

Place `.pt` files in `C:\D\New folder` (default search path) or `<repo>\models\`.

**Fallback chain:** `best_football_seg.pt` (legacy) → `yolov8n.pt` (COCO) → Roboflow API.

---

## ⚙️ Pipeline

```
Video input
  ↓  Frame sampling          configurable interval (default 2 s)
  ↓  Camera motion           Lucas-Kanade optical flow → (dx, dy) correction
  ↓  Detection               PlayerModel @640px + BallModel @1280px + PitchModel (optional)
  ↓  Team classification     TeamClassifier: KMeans HSV (fast) / SigLIP+UMAP+KMeans (SOTA)
  ↓  Tracking                ByteTrack + 8-frame team-vote stability window
  ↓  Coordinate transform    Pixels → pitch metres via homography (auto or manual)
  ↓  Event detection         EventSpotterTDEED — possession-based (Peral et al. 2025)
  ↓  Performance metrics     Distance (km), top speed (km/h), sprint count (FIFA: 25.2 km/h)
Results → JSON dict + Streamlit dashboard
```

**Team classification modes:**
- `kmeans` (default): KMeans on HSV torso colors — fast, no GPU required.
- `siglip`: `google/siglip-base-patch16-224` embeddings + UMAP + KMeans — SOTA, robust to similar colors and VEO lighting.

---

## 🛠️ Quick Setup (Windows)

1. Copy `.pt` model files to `C:\D\New folder`.
2. Double-click **`install.bat`** — installs all dependencies automatically.
3. Double-click **`run_app.bat`** — opens the app in your browser.

---

## 📖 User Guide (Coaches & Scouts)

### 1. Launch
Open **`run_app.bat`**. The EDApp dashboard opens in your browser.

### 2. Upload Video
Click **"Upload Video"** in the sidebar. Accepts `.mp4` files — VEO panoramic and standard broadcast both supported.

### 3. Calibration
The AI detects pitch lines and corners automatically (pose_field.pt). No manual steps required for most videos.

### 4. Start Analysis
Click **"Start Analysis"**. A progress bar shows detection → tracking → events.

### 5. Explore Results
- **Physical Stats** — fastest player, distance covered per player
- **Event Log** — passes, turnovers, shots with timestamps
- **Heatmaps** — player positioning maps on the pitch

---

## 📂 Module Reference

| Module | Role |
|--------|------|
| `modules/detector.py` | 3-model detection cascade + SigLIP/KMeans team classifier |
| `modules/tracker.py` | ByteTrack wrapper with team voting (`ProfessionalTracker`) |
| `modules/team_classifier.py` | KMeans HSV team color classifier with adaptive re-calibration |
| `modules/event_spotter_tdeed.py` | Possession-based event detection + geometry rules |
| `modules/performance_engine.py` | Physical metrics: distance, speed, sprints |
| `modules/calibration_auto.py` | Auto pitch calibration via YOLO pose keypoints |
| `modules/calibration_pnl.py` | Manual PnP pitch calibration |
| `modules/camera_motion.py` | Lucas-Kanade optical flow for camera motion compensation |
| `modules/video_processor.py` | Pipeline orchestrator |
| `core/pipeline/video_pipeline.py` | Entry point and config management |
| `core/config/settings.py` | Pydantic settings (paths, thresholds, device) |

---

## 🗺️ Roadmap

- [ ] Jersey number OCR (`identity_reader.py` scaffold ready — needs PARSeq/YOLO weights)
- [ ] Automatic highlight clip generation (`auto_clip_generator.py` scaffold ready)
- [ ] Export to Wyscout / StatsBomb JSON format
- [ ] Formation detection from average player positions
- [ ] GPU-accelerated batch inference for full 90-min matches

---

*Powered by YOLO · ByteTrack · SigLIP · Streamlit*
