import cv2
import numpy as np
from detect_players import detect_players
from track_players import track_players
from detect_lines import detect_lines
from adjust_perspective import adjust_perspective
from refine_field_lines import refine_field_lines
from draw_var_line import draw_var_line

# Carregar vídeo
video_path = "data/match_video.mp4"
cap = cv2.VideoCapture(video_path)

# Obter informações do vídeo
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Criar vídeo de saída
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_video.mp4", fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Fim do vídeo

    # Refinar a detecção das linhas do campo
    field_lines = refine_field_lines(frame)

    # Ajustar a perspectiva dinamicamente
    frame = adjust_perspective(frame, field_lines)

    # Detectar jogadores
    player_positions = detect_players(frame)

    # Aplicar tracking com DeepSORT
    tracked_players = track_players(player_positions, frame)

    # Ajustar e desenhar a linha do VAR
    draw_var_line(frame, width, tracked_players, field_lines)

    # Exibir frame processado
    cv2.imshow("VAR Automático", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
