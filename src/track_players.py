import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

# Cache para armazenar o tracker entre chamadas
global_tracker = None

def track_players(frame, detections, tracker=None):
    """
    Rastreia jogadores em um frame usando DeepSort.
    
    Args:
        frame: O frame atual.
        detections: Lista de detecções no formato [(bbox, score, class_id), ...].
        tracker: O objeto tracker existente para reutilização (opcional).
    
    Returns:
        Um objeto TrackPlayers com os jogadores rastreados.
    """
    # Inicializar o rastreador se não fornecido
    if tracker is None:
        tracker = DeepSort(
            max_age=30,
            n_init=3,
            nn_budget=100,
            use_cuda=True
        )
    
    # Formatar as detecções para o DeepSort
    formatted_detections = []
    for det in detections:
        bbox = det[0]  # [x1, y1, x2, y2]
        score = det[1]
        class_id = det[2]
        
        # Converter para o formato esperado pelo DeepSort: [x, y, w, h, confidence]
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        formatted_detections.append(([x1, y1, w, h], score, class_id))
    
    # Rastrear os objetos
    tracks = tracker.update_tracks(formatted_detections, frame=frame)
    
    # Filtrar apenas os tracks ativos
    active_tracks = [track for track in tracks if track.is_confirmed() and track.time_since_update <= 5]
    
    return TrackPlayers(active_tracks, tracker)

class TrackPlayers:
    def __init__(self, tracks, tracker):
        self.tracks = tracks
        self.tracker = tracker
    
    def get_track_by_id(self, id):
        """Obtém um track pelo ID."""
        for track in self.tracks:
            if track.track_id == id:
                return track
        return None
    
    def get_all_tracks(self):
        """Retorna todos os tracks."""
        return self.tracks
    
    def get_track_positions(self):
        """Retorna as posições (centros) de todos os tracks."""
        positions = []
        for track in self.tracks:
            if hasattr(track, 'to_tlwh'):
                bbox = track.to_tlwh()
                x, y, w, h = bbox
                center_x = x + w/2
                foot_y = y + h  # Posição dos pés (parte inferior)
                positions.append((center_x, foot_y))
        return positions
    
    def get_track_boxes(self):
        """Retorna as caixas delimitadoras de todos os tracks."""
        boxes = []
        for track in self.tracks:
            if hasattr(track, 'to_tlbr'):
                bbox = track.to_tlbr()  # [x1, y1, x2, y2]
                boxes.append(bbox)
        return boxes
