# trackers/speed_distance.py
import cv2
import numpy as np
from trackers.optical_flow import estimate_camera_motion, compensate_camera_motion

def calculate_player_velocity(tracks, flow_vectors, perspective_transformer, frame_num, frame_rate=30.0):
    """Calculate velocity for each player using optical flow and perspective transformation"""
    if frame_num >= len(flow_vectors):
        return {}
    
    velocities = {}
    flow = flow_vectors[frame_num]
    
    # Calculate camera motion
    camera_motion = estimate_camera_motion(flow)
    
    # Compensate for camera motion
    compensated_flow = compensate_camera_motion(flow, camera_motion)
    
    for track_id, player in tracks['Player'][frame_num].items():
        bbox = player["bbox"]
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Get center point of bottom of bbox (feet position)
        foot_point = (int((x1 + x2) / 2), int(y2))
        
        # Check if point is within flow bounds
        h, w = flow.shape[:2]
        if 0 <= foot_point[0] < w and 0 <= foot_point[1] < h:
            # Get flow vector at foot position
            flow_x = compensated_flow[foot_point[1], foot_point[0], 0]
            flow_y = compensated_flow[foot_point[1], foot_point[0], 1]
            
            # Transform points to real-world coordinates
            p1 = perspective_transformer.transform_point(foot_point)
            p2 = perspective_transformer.transform_point((foot_point[0] + flow_x, foot_point[1] + flow_y))
            
            # Calculate displacement in meters
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            # Calculate velocity in m/s
            displacement = np.sqrt(dx**2 + dy**2)
            velocity = displacement * frame_rate  # Convert to m/s
            
            velocities[track_id] = velocity
    
    return velocities

def update_player_distances(tracks, velocities, frame_num, frame_rate=30.0):
    """Update cumulative distance for each player"""
    # Time between frames in seconds
    dt = 1.0 / frame_rate
    
    # Initialize distances for first frame
    if frame_num == 0:
        for track_id in tracks['Player'][frame_num].keys():
            tracks['Player'][frame_num][track_id]['distance'] = 0.0
    
    # For subsequent frames, update distances from previous frame
    elif frame_num > 0:
        prev_frame = tracks['Player'][frame_num - 1]
        curr_frame = tracks['Player'][frame_num]
        
        for track_id, player in curr_frame.items():
            # Get previous distance or initialize to 0
            prev_distance = 0.0
            if track_id in prev_frame:
                prev_distance = prev_frame[track_id].get('distance', 0.0)
            
            # Calculate new distance
            additional_distance = 0.0
            if track_id in velocities:
                additional_distance = velocities[track_id] * dt
            
            # Update player distance
            curr_frame[track_id]['distance'] = prev_distance + additional_distance
    
    return tracks


def get_player_final_distance(tracks, player_id):
    for frame_data in reversed(tracks['Player']):
        if player_id in frame_data and 'distance' in frame_data[player_id]:
            return frame_data[player_id]['distance']

    # Player not found
    return None