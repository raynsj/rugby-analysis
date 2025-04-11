# trackers/optical_flow.py
import cv2
import numpy as np

def calculate_optical_flow(frames, frame_skip=3):
    """Calculate sparse optical flow using Lucas-Kanade every nth frame"""
    flow_vectors = []
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Process every nth frame pair (0-1, 3-4, 6-7, etc.)
    for i in range(0, len(frames) - 1, frame_skip):
        prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY)
        
        # Detect key features to track
        features = cv2.goodFeaturesToTrack(prev_gray, 
                                         maxCorners=100,
                                         qualityLevel=0.3,
                                         minDistance=7,
                                         blockSize=7)
        
        if features is not None:
            # Calculate optical flow
            new_features, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, features, None, **lk_params)
            
            # Filter only successful tracks
            good_old = features[status == 1]
            good_new = new_features[status == 1]
            
            # Calculate displacement vectors
            displacements = good_new - good_old
            flow_vectors.append(displacements)
        else:
            flow_vectors.append(np.array([]))  # Empty array if no features
    
    return flow_vectors

def estimate_camera_motion(flow):
    """Estimate global camera motion from sparse flow vectors"""
    if len(flow) == 0:
        return 0.0, 0.0
    mean_dx = np.mean(flow[:, 0])
    mean_dy = np.mean(flow[:, 1])
    return mean_dx, mean_dy

def compensate_camera_motion(flow, camera_motion):
    """Compensate for camera motion in sparse flow vectors"""
    if len(flow) == 0:
        return flow
    compensated = flow.copy()
    compensated[:, 0] -= camera_motion[0]
    compensated[:, 1] -= camera_motion[1]
    return compensated
