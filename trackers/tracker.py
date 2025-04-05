# trackers/tracker.py
from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import numpy as np
import sys

sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width
from trackers.team_assignment import extract_player_colors, assign_teams, get_team_colors
from trackers.optical_flow import calculate_optical_flow
from trackers.perspective_transform import PerspectiveTransformer
from trackers.speed_distance import calculate_player_velocity, update_player_distances

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.perspective_transformer = PerspectiveTransformer()
        self.team_colors = None

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch

        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "Player": [],
            "ref": [],
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["Player"].append({})
            tracks["ref"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['Player']:
                    tracks['Player'][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['ref']:
                    tracks['ref'][frame_num][track_id] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    
    def process_video(self, frames, tracks):
        """Process video frames to calculate player metrics"""
        print("Calculating optical flow...")
        flow_vectors = calculate_optical_flow(frames)
        
        print("Setting up perspective transformation...")
        # For simplicity, we'll use predefined field corners
        h, w = frames[0].shape[:2]
        field_corners = [
            (w * 0.1, h * 0.2),  # Top left
            (w * 0.9, h * 0.2),  # Top right
            (w * 0.1, h * 0.9),  # Bottom left
            (w * 0.9, h * 0.9)   # Bottom right
        ]
        self.perspective_transformer.set_field_corners(frames[0], field_corners)
        
        print("Extracting team information...")
        # Use the middle frame for team assignment
        mid_frame = len(frames) // 2
        player_colors = extract_player_colors(frames[mid_frame], tracks, mid_frame)
        team_assignments = assign_teams(player_colors)
        self.team_colors = get_team_colors(player_colors, team_assignments)
        
        # Assign teams to all frames
        for frame_num in range(len(frames)):
            for player_id in tracks['Player'][frame_num].keys():
                if player_id in team_assignments:
                    tracks['Player'][frame_num][player_id]['team'] = team_assignments[player_id]
                else:
                    # Assign team based on colors in current frame

                    # Inside process_video method
                    if tracks['Player'] and frame_num < len(tracks['Player']):
                    # Make sure tracks['Player'][frame_num] exists and is not empty
                        if tracks['Player'][frame_num]:
                            frame_colors = extract_player_colors(frames[frame_num], {
                                'Player': {0: tracks['Player'][frame_num]}
                            }, 0)
                        else:
                            frame_colors = {}
                    else:
    # Handle the case when there are no tracks for this frame
                        frame_colors = {}

                        
                        if player_id in frame_colors:
                            # Find closest team color
                            min_dist = float('inf')
                            assigned_team = 0
                            
                            for team_id, team_color in self.team_colors.items():
                                dist = np.linalg.norm(frame_colors[player_id] - team_color)
                                if dist < min_dist:
                                    min_dist = dist
                                    assigned_team = team_id
                            
                            tracks['Player'][frame_num][player_id]['team'] = assigned_team
        
        print("Calculating player velocities and distances...")
        for frame_num in range(len(frames) - 1):
            # Calculate velocities for this frame
            velocities = calculate_player_velocity(
                tracks, flow_vectors, self.perspective_transformer, 
                frame_num, frame_rate=30.0
            )
            
            # Update player velocities
            for player_id, velocity in velocities.items():
                if player_id in tracks['Player'][frame_num]:
                    tracks['Player'][frame_num][player_id]['velocity'] = velocity
            
            # Update cumulative distances
            tracks = update_player_distances(tracks, velocities, frame_num)
        
        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        return frame

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks['Player'][frame_num]
            referee_dict = tracks['ref'][frame_num]

            # Team colors for visualization
            team_colors = [
                (255, 0, 0),   # Team 0: Red
                (0, 0, 255)    # Team 1: Blue
            ]

            # Drawing Players
            for track_id, player in player_dict.items():
                bbox = player["bbox"]
                
                # Get team color
                team_id = player.get('team', 0)  # Default to team 0 if not assigned
                team_color = team_colors[team_id % len(team_colors)]
                
                # Draw ellipse with team color
                frame = self.draw_ellipse(frame, bbox, team_color, track_id)
                
                # Get player velocity and distance
                velocity = player.get('velocity', 0.0)
                distance = player.get('distance', 0.0)
                
                # Display player info
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                
                # Display team, ID, velocity, and distance
                info_text = f"ID:{track_id} T:{team_id} {velocity:.1f}m/s {distance:.0f}m"
                cv2.putText(
                    frame, info_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, team_color, 2
                )

            # Drawing Referees
            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255), track_id)

            output_video_frames.append(frame)
            
        return output_video_frames
