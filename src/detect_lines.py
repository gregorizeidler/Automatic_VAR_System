import cv2
import numpy as np

def detect_lines(frame):
    """Detecta as linhas do campo usando Canny e HoughLinesP."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
    
    field_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            field_lines.append((x1, y1, x2, y2))
    
    return field_lines
