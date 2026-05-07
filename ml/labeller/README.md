# EDUDIN Pro Labeller — Setup Guide

## Requirements
- Python 3.10 or higher
- Git (optional)

## Installation
1. Download or clone this repository
2. Open a terminal in this folder
3. Run: pip install -r requirements.txt

## Adding Images to Label
1. Copy your .jpg images into:
   data/datasets/veo_frames_raw/images/
2. The labeller will load them automatically

## Running the Labeller
- Double-click Labeller.bat (Windows)
- Or run: python labeller_app.py
- Open your browser at: http://localhost:5000

## How to Label
- Click "IA PREDICT (YOLO)" to auto-detect players
- Right-click a box to delete it
- Left-click + drag to move around
- Scroll wheel to zoom in/out
- Fix wrong boxes in the right panel
- Click "GUARDAR MANUAL" to save
- Green counter shows your progress

## Sending your work back
- Zip the labels/ folder and send it back
- Labels are saved as .txt files (YOLO format)
