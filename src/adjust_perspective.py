import cv2
import numpy as np

def adjust_perspective(frame, lines):
    """Dynamically adjusts the field perspective based on detected lines."""
    if lines is not None and len(lines) >= 4:
        sorted_lines = sorted(lines, key=lambda x: x[0][1])  # Sort by Y coordinate
        src_points = np.float32([
            sorted_lines[0][0][:2], 
            sorted_lines[1][0][:2], 
            sorted_lines[2][0][:2], 
            sorted_lines[3][0][:2]
        ])
        dst_points = np.float32([[100, 400], [500, 400], [100, 600], [500, 600]])
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        return cv2.warpPerspective(frame, perspective_matrix, (frame.shape[1], frame.shape[0]))
    return frame  # Return original frame if no lines detected
