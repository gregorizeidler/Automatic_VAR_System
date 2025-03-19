from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("models/yolov8.pt")

def detect_players(frame):
    """ Detects players in the frame and returns bounding boxes. """
    results = model(frame)
    boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2))
    return boxes
