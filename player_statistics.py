import pandas as pd

def analyze_consecutive_players(processed_tracks, min_seconds=2, fps=30):
    """
    Find all players who appeared for at least min_seconds consecutive seconds
    and calculate their highest distance covered.
    
    Args:
        processed_tracks: The tracking data
        min_seconds: Minimum consecutive seconds required
        fps: Frames per second of the video
    
    Returns:
        DataFrame with player statistics
    """
    # Define minimum consecutive frames
    min_consecutive_frames = min_seconds * fps
    
    # Track consecutive frame appearances
    player_consecutive_frames = {}
    player_current_streaks = {}

    for frame_idx, frame_data in enumerate(processed_tracks['Player']):
        for player_id in frame_data.keys():
            # Initialize if new player
            if player_id not in player_current_streaks:
                player_current_streaks[player_id] = 1
                player_consecutive_frames[player_id] = 1
            else:
                # Check if this is the next consecutive frame
                if frame_idx > 0 and player_id in processed_tracks['Player'][frame_idx-1]:
                    player_current_streaks[player_id] += 1
                else:
                    player_current_streaks[player_id] = 1
                
                # Update max consecutive frames if current streak is higher
                if player_current_streaks[player_id] > player_consecutive_frames[player_id]:
                    player_consecutive_frames[player_id] = player_current_streaks[player_id]

    # Filter players who appeared for at least the minimum consecutive frames
    qualified_players = [player_id for player_id, frames in player_consecutive_frames.items() 
                        if frames >= min_consecutive_frames]

    print(f"Found {len(qualified_players)} players who appeared for at least {min_seconds} consecutive seconds")

    # Find highest distance for each qualified player
    all_stats = []

    for player_id in qualified_players:
        # Get all distance values for this player
        player_distances = [frame[player_id]['distance'] for frame in processed_tracks['Player']
                          if player_id in frame and 'distance' in frame[player_id]]
        
        # Find highest distance
        if player_distances:
            highest_distance = max(player_distances)
            all_stats.append({
                'Player ID': player_id,
                'Highest Distance Covered (meters)': highest_distance
            })
            

    # Create DataFrame with all player stats
    if all_stats:
        return pd.DataFrame(all_stats)
    else:
        return pd.DataFrame(columns=['Player ID', 'Total Distance Covered (meters)'])

def save_stats_to_csv(stats_df, output_file='all_player_statistics.csv'):
    """Save player statistics to CSV file"""
    stats_df.to_csv(output_file, index=False)
    print(f"Statistics for {len(stats_df)} players saved to {output_file}")
