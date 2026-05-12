# TruthLens Pro v3

Real-time exam surveillance system using YOLOv8 pose estimation and object detection.

## Features
- Real-time student tracking
- Violation detection (Bad Posture, Copying, Phone Use, Restless)
- Live video stream via browser
- Flask REST API

## Requirements
- Python 3.9+
- Webcam or Raspberry Pi Camera

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python detector.py
```

Then open your browser at: `http://localhost:5050`

## Files
- `detector.py` — Main backend (Flask + YOLOv8)
- `app.html` — Frontend dashboard
- `yolov8n.pt` — YOLOv8 object detection model (auto-downloaded)
- `yolov8n-pose.pt` — YOLOv8 pose estimation model (auto-downloaded)

## Notes
- Models `.pt` files are excluded from the repo (see `.gitignore`)
- They will be downloaded automatically on first run
