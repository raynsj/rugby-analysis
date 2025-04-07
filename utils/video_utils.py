import cv2
from pathlib import Path

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path):
    if not output_video_frames:
        raise ValueError("No frames to save")
    
    # Get video dimensions from first frame
    height, width = output_video_frames[0].shape[:2]
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_video_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use H264 codec for better compatibility
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    
    # Create video writer
    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        24,  # frame rate
        (width, height)
    )
    
    # Write frames
    for frame in output_video_frames:
        # Ensure frame dimensions match
        if frame.shape[:2] != (height, width):
            raise ValueError(f"Frame dimensions don't match: {frame.shape[:2]} != {(height, width)}")
        out.write(frame)
    
    out.release()
    print(f"Video saved to: {output_video_path}")
