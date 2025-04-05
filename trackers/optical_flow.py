# trackers/optical_flow.py
import cv2
import numpy as np

def calculate_optical_flow(frames):
    """Calculate optical flow between consecutive frames"""
    flow_vectors = []
    
    for i in range(len(frames) - 1):
        prev_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        next_frame = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
        
        # Use Farneback method for optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame, next_frame, None, 
            pyr_scale=0.5, levels=3, winsize=15, 
            iterations=3, poly_n=5, poly_sigma=1.2, 
            flags=0
        )
        
        flow_vectors.append(flow)
    
    return flow_vectors

def estimate_camera_motion(flow, mask=None):
    """Estimate global camera motion from optical flow"""
    h, w = flow.shape[:2]
    
    if mask is None:
        mask = np.ones((h, w), dtype=np.uint8)
    
    # Calculate mean flow in x and y directions
    mean_flow_x = np.mean(flow[..., 0][mask > 0])
    mean_flow_y = np.mean(flow[..., 1][mask > 0])
    
    return mean_flow_x, mean_flow_y

def compensate_camera_motion(flow, camera_motion):
    """Compensate for camera motion in the optical flow"""
    cam_motion_x, cam_motion_y = camera_motion
    
    compensated_flow = flow.copy()
    compensated_flow[..., 0] -= cam_motion_x
    compensated_flow[..., 1] -= cam_motion_y
    
    return compensated_flow
