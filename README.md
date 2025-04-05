# rugby-analysis-yolov8

Rugby Player Tracking System (v1)

Advanced player tracking system for rugby videos with team assignment, speed tracking, and distance measurement.

Features

    🏃‍♂️ Track individual players with unique IDs

    👕 Automatically assign players to teams based on jersey colors

    📏 Calculate real-world distances in meters

    🚀 Measure player speed in m/s

    📊 Monitor total distance covered by each player

```python
# Clone the repository
git clone https://github.com/raynsj/rugby-analysis-yolov8.git
cd rugby-analysis-yolov8

# Install dependencies
pip install -r requirements.txt
```

Usage

Place your rugby video in the input_videos/ folder

Run the analysis:

```python
main.py
```

# Project Structure

```txt
rugby-tracking/
├── input_videos/      # Place input videos here
├── output_videos/     # Results are saved here
├── models/            # Pre-trained YOLO models
├── trackers/          # Tracking modules
├── utils/             # Utility functions
├── main.py/           # Main.py file
├── stubs/             # Pickle files stored here
└── README.md          # This file

```

# How It Works

The system uses computer vision and machine learning to:

    1. Detect players using YOLO object detection

    2. Track player movement across frames

    3. Cluster jersey colors to assign team membership

    4. Calculate motion vectors using optical flow

    5. Transform pixel measurements to real-world coordinates
