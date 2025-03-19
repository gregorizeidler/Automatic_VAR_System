import cv2
import numpy as np

def draw_var_line(frame, width, tracked_players, field_lines):
    """ Adjusts the VAR line to compensate for field inclination. """
    if len(tracked_players) > 1:
        tracked_players.sort(key=lambda p: p[1], reverse=True)
        penultimate_player = tracked_players[1]

        # Adjust the line based on field inclination
        if field_lines is not None and len(field_lines) >= 2:
            line1_y = field_lines[0][0][1]
            line2_y = field_lines[1][0][1]
            inclination = (line2_y - line1_y) / width
            corrected_y = penultimate_player[1] - inclination * width / 2
        else:
            corrected_y = penultimate_player[1]

        # Draw the corrected VAR line
        cv2.line(frame, (0, int(corrected_y)), (width, int(corrected_y)), (0, 0, 255), 3)
