from utils import read_video, save_video
from trackers import Tracker
from player_statistics import analyze_consecutive_players, save_stats_to_csv
from performance_tracker import PerformanceTracker
import subprocess

def main():
    # Initialize performance tracker
    perf_tracker = PerformanceTracker()

    try:
        # Start tracking the installation process
        perf_tracker.start_section('installation_time')
        subprocess.check_call(["pip", "install", "opencv-python-headless", "numpy", "scikit-learn"])
        perf_tracker.end_section('installation_time')

        # Read video (track I/O time)
        perf_tracker.start_section('video_io_time')
        video_frames = read_video('input_videos/input_video.mp4')
        perf_tracker.end_section('video_io_time')

        # Initialize tracker (track model loading time)
        perf_tracker.start_section('model_loading_time')
        tracker = Tracker('models/model.mlpackage/Data/com.apple.CoreML/model.mlmodel', scale_factor=0.5)
        perf_tracker.end_section('model_loading_time')

        # Object tracking (track detection time)
        perf_tracker.start_section('detection_time')
        tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stub.pkl')
        perf_tracker.end_section('detection_time')

        # Process video WITH OPTIMIZED OPTICAL FLOW
        perf_tracker.start_section('processing_time')
        processed_tracks = tracker.process_video(
            video_frames, 
            tracks,
            frame_skip=3  # 👈 Add this parameter for frame skipping
        )
        perf_tracker.end_section('processing_time')

        # Render output video (track rendering time)
        perf_tracker.start_section('rendering_time')
        output_video_frames = tracker.draw_annotations(video_frames, processed_tracks)
        save_video(output_video_frames, 'output_videos/output_video_cuhk_1.mp4')
        perf_tracker.end_section('rendering_time')

    finally:
        # Record all metrics to CSV
        perf_tracker.record_metrics()

if __name__ == "__main__":
    main()
