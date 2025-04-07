# rugby-analysis-yolov8

Rugby Player Tracking System (v1.1)

Advanced player tracking system for rugby videos with team assignment, speed tracking, and distance measurement.

# Features

    🏃‍♂️ Track individual players with unique IDs

    👕 Automatically assign players to teams based on jersey colors

    📏 Calculate real-world distances in meters

    🚀 Measure player speed in m/s

    📊 Monitor total distance covered by each player

    📏 Player stats will be saved in a .csv file
    

```python
# Clone the repository
git clone https://github.com/raynsj/rugby-analysis-yolov8.git
cd rugby-analysis-yolov8

# Install dependencies
pip install -r requirements.txt
```

# Usage

Place your rugby video in the input_videos/ folder

Track a Specific Player

To focus on a specific player and save their stats to CSV:

```python
# In main.py, modify the player ID value:
player_id = 4  # Change this to your target player's ID

```

# Run the analysis:

```python
python main.py
```

# How it works

This system uses computer vision and machine learning to track rugby players:

    Object Detection: Uses YOLO to detect players in each video frame

    Player Tracking: Maintains player identities across frames

    Team Assignment: Uses color clustering to assign players to teams

    Motion Analysis: Calculates player movement between frames

    Distance Calculation: Measures total distance covered by each player

    Speed Calculation: Computes player speeds based on frame-to-frame movement


# Project Structure

```txt
rugby-analysis-yolov8/
├── input_videos/       # Place input videos here
├── output_videos/      # Processed videos are saved here
├── trackers/           # Core tracking modules
│   ├── tracker.py      # Main tracking functionality
│   ├── team_assignment.py  # Team identification
│   ├── optical_flow.py     # Motion analysis
│   ├── speed_distance.py   # Performance metrics
│   └── perspective_transform.py  # Real-world measurements
├── utils/              # Utility functions
├── stubs/              # pickle files for speed
├── main.py             # Main application
└── requirements.txt    # Dependencies

```

# Performance Notes

Video analysis is computationally intensive and optimization has not been complete yet. For a 7-second video, expect processing to take several minutes depending on your hardware.

# Disclaimers

This is by no means a final product. This is currently a passion project initiated and done completely by myself and I will continue to hone my skills and push updates constantly.

Stay tuned for more!

