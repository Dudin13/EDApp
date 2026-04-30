# ⚽ EDApp v2 — Advanced Football Match Analysis

**EDApp v2** is a professional-grade football video analysis platform. It uses cutting-edge AI to transform standard match footage into deep tactical and physical insights.

**GitHub Repository:** [https://github.com/Dudin13/EDApp](https://github.com/Dudin13/EDApp)

---

## 🌟 What's New in v2?

- **Professional Metrics**: Converts video pixels into real-world units. Get actual **Speed (km/h)** and **Distance (km)**.
- **Event Intelligence**: Automatically detects **Passes**, **Ball Recoveries**, and **Possession changes**.
- **Auto-Calibration**: Detects pitch markings automatically to map the field accurately.
- **High-Performance Tracking**: Uses an optimized pipeline to track players even in complex broadcast or VEO panoramic views.

---

## 🛠️ Quick Setup (For Windows)

1. **Download the Models**: Ensure your `.pt` files are in `C:\D\New folder` (as per the current configuration).
2. **Install**: Double-click **`install.bat`**. This will set up everything automatically.
3. **Run**: Double-click **`run_app.bat`**.

---

## 📖 Simple User Guide (Coaches & Scouts)

Follow these 5 simple steps to analyze your match:

### 1. Launch the App
Open **`run_app.bat`**. A window will open in your web browser showing the EDApp Dashboard.

### 2. Upload Video
On the sidebar, click **"Upload Video"**. Select your match file (`.mp4` format). The system supports both VEO (panoramic) and standard broadcast videos.

### 3. Automatic Calibration
Once the video is uploaded, the AI will "look" at the pitch. It identifies the lines and corners automatically to understand the dimensions of the field. You don't need to do anything!

### 4. Start Analysis
Click the **"Start Analysis"** button. You will see a progress bar.
- **Detection**: The AI finds all players and the ball.
- **Intelligence**: The system calculates how fast players are running and detects passes.

### 5. Explore Results
Once finished, you can explore several tabs:
- **Physical Stats**: See who was the fastest player and how many kilometers they covered.
- **Event Log**: A list of all detected passes and turnovers with timestamps.
- **Heatmaps**: Visual maps of where players spent most of their time on the pitch.

---

## 📂 Technical Project Structure

- **`app/modules/`**: Contains the "brain" of the system (Detector, Tracker, Performance Engine, Event Engine).
- **`core/pipeline/`**: The orchestrator that manages the flow from video to data.
- **`scripts/audit_analysis.py`**: A developer tool to verify system accuracy without using the UI.
- **`.gitignore`**: Configured to protect large models and sensitive data.

---

*Powered by Computer Vision & Professional Football Logic*
