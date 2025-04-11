# trackers/team_assignment.py
import cv2
import numpy as np
from sklearn.cluster import KMeans

def extract_player_colors(frame, tracks, frame_num):
    """Extract colors from player jerseys using their bounding boxes"""
    player_colors = {}

    if 'Player' in tracks and frame_num in tracks['Player']:
        for track_id, player in tracks['Player'][frame_num].items():
            bbox = player["bbox"]
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Extract the upper part of the player (jersey)
            jersey_region = frame[y1:int((y1+y2)/2), x1:x2]
            
            if jersey_region.size > 0:
                # Reshape for color extraction
                pixels = jersey_region.reshape(-1, 3)
                if len(pixels) > 0:
                    # Calculate dominant color
                    dominant_color = pixels.mean(axis=0)
                    player_colors[track_id] = dominant_color
        
    return player_colors

def assign_teams(player_colors, n_teams=2):
    """Assign players to teams based on their dominant jersey colors"""
    if len(player_colors) < n_teams:
        return {}
    
    # Extract colors as a feature matrix
    track_ids = list(player_colors.keys())
    colors = np.array([player_colors[tid] for tid in track_ids])
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_teams, random_state=42)
    labels = kmeans.fit_predict(colors)
    
    # Assign team labels to players
    team_assignments = {track_id: int(label) for track_id, label in zip(track_ids, labels)}
    
    return team_assignments

def get_team_colors(player_colors, team_assignments):
    """Get representative colors for each team"""
    team_colors = {}

    if not team_assignments:
        return team_colors
    
    for team_id in range(max(team_assignments.values()) + 1):
        team_members = [tid for tid, team in team_assignments.items() if team == team_id]
        team_member_colors = [player_colors[tid] for tid in team_members if tid in player_colors]
        
        if team_member_colors:
            team_colors[team_id] = np.mean(team_member_colors, axis=0).astype(int)
        else:
            team_colors[team_id] = np.array([0, 0, 0])
    
    return team_colors
