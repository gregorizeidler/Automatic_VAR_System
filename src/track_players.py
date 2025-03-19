from deep_sort_realtime.deepsort_tracker import DeepSort

tracker = DeepSort(max_age=10)

def track_players(player_boxes, frame):
    """ Applies tracking to detected players. """
    detections = [(x1, y1, x2 - x1, y2 - y1, 1.0) for x1, y1, x2, y2 in player_boxes]
    return tracker.update_tracks(detections, frame=frame)
