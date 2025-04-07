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
                            frame = self.draw_ellipse(frame, player["bbox"], team_color, track_id)
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
        
        # Define modern color schemes for teams
        team_colors = [
            {"primary": (59, 89, 152),   # Team 0: Blue
            "secondary": (223, 227, 238),
            "text": (255, 255, 255)},
            {"primary": (176, 58, 46),   # Team 1: Red
            "secondary": (241, 221, 219),
            "text": (255, 255, 255)},
            {"primary": (46, 134, 87),   # Team 2: Green (if needed)
            "secondary": (221, 236, 230),
            "text": (255, 255, 255)}
        ]
        
        # Referee color scheme
        ref_color = {"primary": (255, 204, 0), "secondary": (30, 30, 30), "text": (0, 0, 0)}
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks['Player'][frame_num]
            referee_dict = tracks['ref'][frame_num]

            # Drawing Players with modern style
            for track_id, player in player_dict.items():
                bbox = player["bbox"]
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                
                # Get team id or default to 0
                team_id = player.get('team', 0) % len(team_colors)
                colors = team_colors[team_id]
                
                
                # Get player velocity and distance
                velocity = player.get('velocity', 0.0)
                distance = player.get('distance', 0.0)
                
                # Create center point
                x_center, _ = get_center_of_bbox(bbox)
                width = get_bbox_width(bbox)
                
                # Draw shadow effect
                cv2.ellipse(
                    frame,
                    center=(x_center+2, y2+2),
                    axes=(int(width), int(0.35*width)),
                    angle=0.0,
                    startAngle=-45,
                    endAngle=235,
                    color=(30, 30, 30),
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
                
                # Draw team-colored ellipse
                cv2.ellipse(
                    frame,
                    center=(x_center, y2),
                    axes=(int(width), int(0.35*width)),
                    angle=0.0,
                    startAngle=-45,
                    endAngle=235,
                    color=colors["primary"],
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
                
                # Create info box with metrics
                # Instead of using bullet points or special symbols like •
                info_text = f"ID:{track_id} T:{team_id} {velocity:.1f}m/s {distance:.0f}m"

                
                # Calculate text size
                (text_width, text_height), _ = cv2.getTextSize(
                    info_text, font, font_scale, font_thickness
                )
                
                # Draw semi-transparent background for text
                alpha = 0.7
                overlay = frame.copy()
                cv2.rectangle(
                    overlay, 
                    (x1, y1-text_height-10), 
                    (x1+text_width+10, y1), 
                    colors["secondary"], 
                    -1
                )
                cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
                
                # Draw team indicator circle
                cv2.circle(
                    frame, 
                    (x1+8, y1-text_height//2-5), 
                    5, 
                    colors["primary"], 
                    -1
                )
                
                # Draw text with shadow effect
                cv2.putText(
                    frame, 
                    info_text, 
                    (x1+18, y1-5), 
                    font, 
                    font_scale, 
                    (30, 30, 30), 
                    font_thickness+1, 
                    cv2.LINE_AA
                )
                cv2.putText(
                    frame, 
                    info_text, 
                    (x1+18, y1-5), 
                    font, 
                    font_scale, 
                    colors["primary"], 
                    font_thickness, 
                    cv2.LINE_AA
                )

            # Drawing Referees with distinct style
            for track_id, referee in referee_dict.items():
                bbox = referee["bbox"]
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                
                # Create referee indicator
                x_center, _ = get_center_of_bbox(bbox)
                width = get_bbox_width(bbox)
                
                # Draw shadow
                cv2.ellipse(
                    frame,
                    center=(x_center+2, y2+2),
                    axes=(int(width), int(0.35*width)),
                    angle=0.0,
                    startAngle=-45,
                    endAngle=235,
                    color=(30, 30, 30),
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
                
                # Draw striped pattern for referee
                for i in range(0, 360, 40):
                    start_angle = i
                    end_angle = i + 20
                    color = ref_color["secondary"] if (i // 40) % 2 == 0 else ref_color["primary"]
                    
                    cv2.ellipse(
                        frame,
                        center=(x_center, y2),
                        axes=(int(width), int(0.35*width)),
                        angle=0.0,
                        startAngle=start_angle,
                        endAngle=end_angle,
                        color=color,
                        thickness=2,
                        lineType=cv2.LINE_AA
                    )
                
                # Add referee label
                ref_text = f"REFEREE #{track_id}"
                
                # Calculate text size
                (text_width, text_height), _ = cv2.getTextSize(
                    ref_text, font, font_scale, font_thickness
                )
                
                # Draw semi-transparent background
                alpha = 0.7
                overlay = frame.copy()
                cv2.rectangle(
                    overlay, 
                    (x1, y1-text_height-10), 
                    (x1+text_width+10, y1), 
                    (240, 240, 200), 
                    -1
                )
                cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
                
                # Draw text
                cv2.putText(
                    frame, 
                    ref_text, 
                    (x1+5, y1-5), 
                    font, 
                    font_scale, 
                    (0, 0, 0), 
                    font_thickness, 
                    cv2.LINE_AA
                )

            output_video_frames.append(frame)
            
        return output_video_frames

