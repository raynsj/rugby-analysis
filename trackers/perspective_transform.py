# trackers/perspective_transform.py
import cv2
import numpy as np

class PerspectiveTransformer:
    def __init__(self, field_width_meters=105, field_height_meters=68):
        """Initialize with real-world field dimensions"""
        self.field_width_meters = field_width_meters
        self.field_height_meters = field_height_meters
        self.transform_matrix = None
        self.inv_transform_matrix = None
        self.pixels_per_meter = None
    
    def set_field_corners(self, frame, corners):
        """Set the corners of the field in the frame"""
        h, w = frame.shape[:2]
        
        # Source points (corners in the image)
        src_points = np.array(corners, dtype=np.float32)
        
        # Destination points (corners in a rectangle)
        dst_width = 1000  # arbitrary width for transformation
        dst_height = int(dst_width * self.field_height_meters / self.field_width_meters)
        
        dst_points = np.array([
            [0, 0],
            [dst_width, 0],
            [0, dst_height],
            [dst_width, dst_height]
        ], dtype=np.float32)
        
        # Calculate perspective transform matrix
        self.transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        self.inv_transform_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
        
        # Calculate pixels per meter
        self.pixels_per_meter = dst_width / self.field_width_meters
    
    def transform_point(self, point):
        """Transform a point from image coordinates to field coordinates"""
        if self.transform_matrix is None:
            raise ValueError("Transform matrix not set, call set_field_corners first")
        
        # Reshape point for transformation
        point = np.array([point], dtype=np.float32).reshape(-1, 1, 2)
        
        # Apply perspective transform
        transformed_point = cv2.perspectiveTransform(point, self.transform_matrix)
        
        # Convert to meters
        x_meters = transformed_point[0][0][0] / self.pixels_per_meter
        y_meters = transformed_point[0][0][1] / self.pixels_per_meter
        
        return x_meters, y_meters
