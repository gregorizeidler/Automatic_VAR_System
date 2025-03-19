import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Carrega o modelo YOLO pré-treinado
model = YOLO("yolov8n.pt")  # Usará o modelo da raiz ou podemos alterá-lo para "models/yolov8.pt"

def detect_players(frame):
    """Detecta jogadores no frame usando YOLO."""
    results = model(frame)
    players = []
    
    for result in results:
        for box in result.boxes:
            if box.cls == 0:  # Filtra apenas pessoas (classe 0 do COCO)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                players.append((x1, y1, x2, y2))
    
    return players
