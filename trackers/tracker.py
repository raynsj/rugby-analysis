# trackers/tracker.py
from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import numpy as np
import sys
from coremltools.models import MLModel 



sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width
from trackers.team_assignment import extract_player_colors, assign_teams, get_team_colors
from trackers.optical_flow import calculate_optical_flow
from trackers.perspective_transform import PerspectiveTransformer
from trackers.speed_distance import calculate_player_velocity, update_player_distances

class Tracker:
    def __init__(self, model_path, scale_factor=0.5):
        """
        Initialize tracker with model and scale factor

        Args:
            model_path: Path to CoreML model
            scale_factor: Scale factor for frame resizing (0.5 = half size)
        """
        self.model = MLModel(model_path)  # Load Core ML model
        self.tracker = sv.ByteTrack()
        self.perspective_transformer = PerspectiveTransformer()
        self.team_colors = None
        self.scale_factor = scale_factor
        print(f"Using scale factor: {scale_factor} for  processing")

    def downscale_frame(self, frame):
        # Downscale a frame for processing
        return cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
    
    def downscale_frames(self, frames):
        # Downscale a list of frames for processing
        return [self.downscale_frame(frame) for frame in frames]

    def upscale_bbox(self, bbox):
        # Upscale a bounding box from small to original frame size
        return [coord / self.scale_factor for coord in bbox]

    

    def detect_frames(self, frames):

        small_frames = self.downscale_frames(frames)

        batch_size = 20
        detections = []
        for i in range(0, len(small_frames), batch_size):
            batch = small_frames[i:i+batch_size]
            detections_batch = self.model.predict(batch, conf=0.1)
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
                # Get bbox in small frame coordinates
                small_box = frame_detection[0].tolist()

                # Scale bbox to original frame size
                bbox = self.upscale_bbox(small_box)

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
    
    def process_video(self, frames, tracks, frame_skip=3):
        """Process video frames to calculate player metrics"""

        small_frames = self.downscale_frames(frames)


        print("Calculating optical flow...")
        flow_vectors = calculate_optical_flow(small_frames, frame_skip)
        
        print("Setting up perspective transformation...")
        # For simplicity, we'll use predefined field corners
        h, w = frames[0].shape[:2]
        field_corners = [
            (w * 0.1, h * 0.2),  # Top left
            (w * 0.9, h * 0.2),  # Top right
            (w * 0.1, h * 0.9),  # Bottom left
            (w * 0.9, h * 0.9)   # Bottom right
        ]

        # Set up perspective trasnformer with small frame
        self.perspective_transformer.set_field_corners(small_frames[0], field_corners)
        
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
        # Scale factor is automatically considered in velocity calculations
        # because we are using the upscaled bounding boxes in tracks

        for frame_num in range(len(frames) - 1):
            # Calculate velocities for this frame
            # flow_vectors already accounts for scaling
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
            {"primary": (59, 89, 152), "secondary": (223, 227, 238)},    # Blue
            {"primary": (176, 58, 46), "secondary": (241, 221, 219)},    # Red
            {"primary": (46, 134, 87), "secondary": (221, 236, 230)}     # Green
        ]
            
        # Referee color
        ref_color = {"primary": (255, 204, 0), "secondary": (30, 30, 30)}
            
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        font_thickness = 1
            
        # 1. Render at lower resolution for speed
        scale = 0.5  # Render at half resolution
            
        for frame_num, frame in enumerate(video_frames):
            # Downscale for faster rendering
            small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            h, w = small_frame.shape[:2]
                
            # 2. Create a single annotation layer
            annotation_layer = np.zeros((h, w, 3), dtype=np.uint8)
                
            player_dict = tracks['Player'][frame_num]
            referee_dict = tracks['ref'][frame_num]

            # 3. Batch drawing - first do all shapes, then all text
            for track_id, player in player_dict.items():
                # Scale down original bbox
                bbox = [int(coord * scale) for coord in player["bbox"]]
                x1, y1, x2, y2 = bbox
                    
                team_id = player.get('team', 0) % len(team_colors)
                colors = team_colors[team_id]
                    
                # Create center point
                x_center = int((x1 + x2) / 2)
                width = int(x2 - x1)
                    
                # Just draw team-colored ellipse (skip shadow)
                cv2.ellipse(
                    annotation_layer,
                    center=(x_center, y2),
                    axes=(width, int(0.35*width)),
                    angle=0.0,
                    startAngle=-45,
                    endAngle=235,
                    color=(255, 255, 255),            # colors["primary"],
                    thickness=2,
                    lineType=cv2.LINE_8  # Faster than LINE_AA
                )
                    
                # Draw text background all at once
                velocity = player.get('velocity', 0.0)
                distance = player.get('distance', 0.0)
                info_text = f"ID:{track_id} {velocity:.1f}m/s {distance:.0f}m"
                    
                (text_width, text_height), _ = cv2.getTextSize(
                    info_text, font, font_scale, font_thickness
                )
                    
                # Draw text background directly on annotation layer
                cv2.rectangle(
                    annotation_layer, 
                    (x1, y1-text_height-5), 
                    (x1+text_width+5, y1), 
                    colors["secondary"], 
                    -1
                )
                    
                # Store text info for batch processing
                cv2.putText(
                    annotation_layer, 
                    info_text, 
                    (x1+5, y1-5), 
                    font, 
                    font_scale, 
                    (255, 0, 0),              # colors["primary"], 
                    font_thickness, 
                    cv2.LINE_8
                )

            # 4. Process all referees together
            for track_id, referee in referee_dict.items():
                bbox = [int(coord * scale) for coord in referee["bbox"]]
                x1, y1, x2, y2 = bbox
                    
                x_center = int((x1 + x2) / 2)
                width = int(x2 - x1)
                    
                # Simplify: just draw a single distinctive ellipse
                cv2.ellipse(
                    annotation_layer,
                    center=(x_center, y2),
                    axes=(width, int(0.35*width)),
                    angle=0.0,
                    startAngle=-45,
                    endAngle=235,
                    color=ref_color["primary"],
                    thickness=2,
                    lineType=cv2.LINE_8
                )
                    
                # Basic referee label
                ref_text = f"REF:{track_id}"
                cv2.rectangle(
                    annotation_layer, 
                    (x1, y1-15), 
                    (x1+70, y1), 
                    (240, 240, 200), 
                    -1
                )
                    
                cv2.putText(
                    annotation_layer, 
                    ref_text, 
                    (x1+5, y1-5), 
                    font, 
                    font_scale, 
                    (0, 0, 0), 
                    font_thickness, 
                    cv2.LINE_8
                )

            # 5. Blend annotation layer with frame just once
            alpha = 0.7
            small_result = cv2.addWeighted(small_frame, 1.0, annotation_layer, alpha, 0)
                
            # 6. Upscale back to original resolution
            result = cv2.resize(small_result, (frame.shape[1], frame.shape[0]))
                
            output_video_frames.append(result)
            
            # 7. Hardware acceleration for video encoding
            # This happens in save_video, make sure to use hardware-accelerated codec
        return output_video_frames


