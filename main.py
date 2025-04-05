# main.py
from utils import read_video, save_video
from trackers import Tracker

def main():
    # Install required packages
    import subprocess
    subprocess.check_call(["pip", "install", "opencv-python-headless", "numpy", "scikit-learn"])
    
    # read video
    video_frames = read_video('input_videos/input_video.mp4')

    # Initialize tracker
    tracker = Tracker('models/model_name.pt')

    # Get object tracks
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stub.pkl')
    
    # Process video to get player metrics
    processed_tracks = tracker.process_video(video_frames, tracks)

    # Draw output
    output_video_frames = tracker.draw_annotations(video_frames, processed_tracks)
    
    # save video
    save_video(output_video_frames, 'output_videos/output_video.mp4')


if __name__ == "__main__":
    main()
