# main.py
from utils import read_video, save_video
from trackers import Tracker
import pandas as pd
from trackers.speed_distance import get_player_final_distance

def main():

    # Install required packages
    import subprocess
    subprocess.check_call(["pip", "install", "opencv-python-headless", "numpy", "scikit-learn"])
    
    # read video
    video_frames = read_video('input_videos/input_video_cuhk.mp4')

    # Initialize tracker
    tracker = Tracker('modelsv0.0.4/best(4).pt')

    # Get object tracks
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stub_cuhk_1.pkl')
    
    # Process video to get player metrics
    processed_tracks = tracker.process_video(video_frames, tracks)

    # Draw output
    output_video_frames = tracker.draw_annotations(video_frames, processed_tracks)
    
    # save video
    save_video(output_video_frames, 'output_videos/output_video_cuhk_1.mp4')


    # Extract final distance for player ID 4
    player_id = 4
    all_distances = [frame[player_id]['distance'] for frame in processed_tracks['Player'] if player_id in frame and 'distance' in frame[player_id]]


    # Highest distance covered
    highest_distance = max(all_distances) if all_distances else None

    # Save to CSV
    if highest_distance is not None:
        print(f"Player {player_id} highest distance: {highest_distance} meters")
        
        # Create DataFrame and save to CSV
        data = {'Player ID': [player_id], 'Highest Distance Covered (meters)': [highest_distance]}
        df = pd.DataFrame(data)
        df.to_csv('player_statistics.csv', index=False)
        print(f"Statistics saved to player_statistics.csv")
    else:
        print(f"Player {player_id} not found in any frames")


if __name__ == "__main__":
    main()
