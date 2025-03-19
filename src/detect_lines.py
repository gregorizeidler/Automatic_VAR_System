import cv2
import numpy as np

def detect_lines(frame):
    """ Detects field lines using edge detection and Hough Transform. """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)
    return lines
