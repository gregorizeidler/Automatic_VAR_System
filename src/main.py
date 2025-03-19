import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizando dispositivo: {device}")

# Configurações
CONFIDENCE_THRESHOLD = 0.35  # Limiar de confiança para detecção
LINE_COLOR = (0, 0, 255)     # Cor da linha de impedimento (BGR: vermelho)
OFFSIDE_COLOR = (0, 0, 255)  # Cor para jogadores em possível impedimento (vermelho)
ONSIDE_COLOR = (0, 255, 0)   # Cor para jogadores em posição regular (verde)
DEFENDER_COLOR = (255, 165, 0)  # Cor para o último defensor (laranja)
ATTACKER_COLOR = (255, 0, 255)  # Cor para o atacante (magenta)

# Variáveis globais - simplificadas
paused = False
current_frame_num = 0
frame_offset = 0
analysis_frame = None  # Frame atual para análise
analysis_detections = []
field_points = []  # Para definir a linha de campo (perspectiva)
defender_point = None  # Posição do último defensor
attacker_point = None  # Posição do atacante
offside_line = None  # Linha de impedimento calculada
attacking_direction = 'right'  # Direção padrão do ataque
analysis_mode = False  # Se estamos em modo de análise
selection_step = 0  # 0: nada, 1: primeiro ponto, 2: segundo ponto, 3: defensor, 4: atacante, 5: finalizado

# Funções auxiliares simplificadas

def calculate_offside_line(field_line, defender):
    """Calcula a linha de impedimento PARALELA à linha de campo, passando pelo defensor."""
    # Obter pontos da linha de campo
    x1, y1 = field_line[0]
    x2, y2 = field_line[1]
    
    print(f"Pontos da linha de campo: ({x1},{y1}) -> ({x2},{y2})")
    print(f"Ponto do defensor: ({defender[0]},{defender[1]})")
    
    # Coordenadas do defensor
    dx, dy = defender
    
    # MUDANÇA IMPORTANTE: a linha de impedimento deve ser PARALELA à linha de campo,
    # não perpendicular. Então, calculamos o vetor diretor da linha de campo
    field_vector = (x2 - x1, y2 - y1)
    
    # Normalizar o vetor diretor
    field_length = np.sqrt(field_vector[0]**2 + field_vector[1]**2)
    if field_length == 0:
        field_unit = (1, 0)  # Fallback para vetores muito pequenos
    else:
        field_unit = (field_vector[0] / field_length, field_vector[1] / field_length)
    
    print(f"Vetor diretor normalizado da linha de campo: {field_unit}")
    
    # Calcular pontos para a linha de impedimento usando o mesmo vetor diretor
    # mas passando pelo defensor e estendendo em ambas as direções
    distance = 10000  # Distância grande para garantir que a linha cruze toda a tela
    
    # Criar pontos em ambas as direções a partir do defensor
    p1 = (int(dx + field_unit[0] * distance), int(dy + field_unit[1] * distance))
    p2 = (int(dx - field_unit[0] * distance), int(dy - field_unit[1] * distance))
    
    print(f"Linha de impedimento calculada (não recortada): {p1} -> {p2}")
    return [p1, p2]

def calculate_attacker_line(field_line, attacker):
    """Calcula a linha do atacante PARALELA à linha de campo, passando pelo atacante."""
    # Obter pontos da linha de campo
    x1, y1 = field_line[0]
    x2, y2 = field_line[1]
    
    print(f"Pontos da linha de campo: ({x1},{y1}) -> ({x2},{y2})")
    print(f"Ponto do atacante: ({attacker[0]},{attacker[1]})")
    
    # Coordenadas do atacante
    ax, ay = attacker
    
    # A linha do atacante deve ser PARALELA à linha de campo
    field_vector = (x2 - x1, y2 - y1)
    
    # Normalizar o vetor diretor
    field_length = np.sqrt(field_vector[0]**2 + field_vector[1]**2)
    if field_length == 0:
        field_unit = (1, 0)  # Fallback para vetores muito pequenos
    else:
        field_unit = (field_vector[0] / field_length, field_vector[1] / field_length)
    
    # Calcular pontos para a linha do atacante usando o mesmo vetor diretor
    distance = 10000  # Distância grande para garantir que a linha cruze toda a tela
    
    # Criar pontos em ambas as direções a partir do atacante
    p1 = (int(ax + field_unit[0] * distance), int(ay + field_unit[1] * distance))
    p2 = (int(ax - field_unit[0] * distance), int(ay - field_unit[1] * distance))
    
    return [p1, p2]

def check_offside(defender_point, attacker_point, attacking_direction):
    """Verifica se o atacante está em posição de impedimento com base na direção de ataque."""
    if attacking_direction == 'right':
        # Atacando para a direita, impedimento se atacante estiver mais à direita que o defensor
        return attacker_point[0] > defender_point[0]
    else:  # attacking_direction == 'left'
        # Atacando para a esquerda, impedimento se atacante estiver mais à esquerda que o defensor
        return attacker_point[0] < defender_point[0]

def clip_line_to_screen(line, frame_width, frame_height):
    """Recorta uma linha para os limites da tela."""
    p1, p2 = line
    x1, y1 = p1
    x2, y2 = p2
    
    # Calcular os coeficientes da equação da linha: ax + by + c = 0
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    
    # Array para armazenar as interseções válidas
    intersections = []
    
    # Verificar interseção com borda esquerda (x = 0)
    if b != 0:  # Evita divisão por zero
        y_left = -c / b  # Quando x = 0, y = -c/b
        if 0 <= y_left < frame_height:
            intersections.append((0, int(y_left)))
    
    # Verificar interseção com borda direita (x = frame_width-1)
    if b != 0:  # Evita divisão por zero
        y_right = -(a * (frame_width-1) + c) / b
        if 0 <= y_right < frame_height:
            intersections.append((frame_width-1, int(y_right)))
    
    # Verificar interseção com borda superior (y = 0)
    if a != 0:  # Evita divisão por zero
        x_top = -c / a  # Quando y = 0, x = -c/a
        if 0 <= x_top < frame_width:
            intersections.append((int(x_top), 0))
    
    # Verificar interseção com borda inferior (y = frame_height-1)
    if a != 0:  # Evita divisão por zero
        x_bottom = -(b * (frame_height-1) + c) / a
        if 0 <= x_bottom < frame_width:
            intersections.append((int(x_bottom), frame_height-1))
    
    # Se temos pelo menos 2 interseções válidas, usar essas
    if len(intersections) >= 2:
        # Pegar as duas primeiras interseções (elas já são ordenadas por construção)
        return [intersections[0], intersections[1]]
    
    # Se não temos interseções suficientes, limitar pontos originais às bordas
    clipped_line = [
        (max(0, min(frame_width-1, x1)), max(0, min(frame_height-1, y1))),
        (max(0, min(frame_width-1, x2)), max(0, min(frame_height-1, y2)))
    ]
    return clipped_line

def draw_simple_frame(frame):
    """Desenha um frame de análise simplificado."""
    h, w = frame.shape[:2]
    output = frame.copy()
    
    # 1. Desenhar pontos da linha de campo
    for i, point in enumerate(field_points):
        cv2.circle(output, point, 10, (255, 255, 0), -1)
        cv2.putText(output, f"Campo {i+1}", (point[0]+10, point[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # 2. Desenhar linha de campo se temos dois pontos
    if len(field_points) >= 2:
        # Desenhar linha amarela entre os pontos selecionados
        cv2.line(output, field_points[0], field_points[1], (255, 255, 0), 3)
        
        # Estender a linha de campo para preencher a largura (em ciano)
        x1, y1 = field_points[0]
        x2, y2 = field_points[1]
        
        if x2 - x1 != 0:  # Evitar divisão por zero
            # Calcular a equação da linha y = mx + b
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            
            # Calcular pontos nas bordas da tela
            extended_line = clip_line_to_screen([(0, int(b)), (w, int(m * w + b))], w, h)
            
            # Desenhar linha estendida em ciano
            cv2.line(output, extended_line[0], extended_line[1], (0, 255, 255), 2)
    
    # 3. Desenhar o defensor
    if defender_point:
        cv2.circle(output, defender_point, 10, DEFENDER_COLOR, -1)
        cv2.putText(output, "DEFENSOR", (defender_point[0]+10, defender_point[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, DEFENDER_COLOR, 2)
    
    # 4. Desenhar o atacante
    if attacker_point:
        cv2.circle(output, attacker_point, 10, ATTACKER_COLOR, -1)
        cv2.putText(output, "ATACANTE", (attacker_point[0]+10, attacker_point[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, ATTACKER_COLOR, 2)
    
    # 5. Desenhar a linha de impedimento (defensor)
    if offside_line and len(offside_line) >= 2 and defender_point:
        # Recortar a linha para caber na tela
        clipped_line = clip_line_to_screen(offside_line, w, h)
        p1, p2 = clipped_line
        
        # Desenhar linha branca mais grossa (borda)
        cv2.line(output, p1, p2, (255, 255, 255), 7)
        # Desenhar linha vermelha por cima (mais grossa)
        cv2.line(output, p1, p2, LINE_COLOR, 5)
        
        # Texto da linha de impedimento
        mid_x = int((p1[0] + p2[0]) / 2)
        mid_y = int((p1[1] + p2[1]) / 2)
        
        # Adicionar fundo para o texto
        text = "LINHA DO DEFENSOR"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        
        # Garantir que o texto esteja visível mesmo quando a linha está perto da borda
        text_x = max(text_size[0]//2 + 5, min(mid_x, w - text_size[0]//2 - 5))
        text_y = max(25, min(mid_y, h - 10))
        
        cv2.rectangle(output, 
                     (text_x - text_size[0]//2 - 5, text_y - 25),
                     (text_x + text_size[0]//2 + 5, text_y + 5),
                     (0, 0, 0), -1)
        
        # Adicionar texto
        cv2.putText(output, text, (text_x - text_size[0]//2, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    
    # 6. Desenhar a linha do atacante
    if attacker_point and len(field_points) >= 2:
        # Calcular linha do atacante
        attacker_line = calculate_attacker_line(field_points, attacker_point)
        
        # Recortar a linha para caber na tela
        clipped_attacker_line = clip_line_to_screen(attacker_line, w, h)
        p1, p2 = clipped_attacker_line
        
        # Desenhar linha branca mais grossa (borda)
        cv2.line(output, p1, p2, (255, 255, 255), 7)
        # Desenhar linha azul por cima (mais grossa)
        cv2.line(output, p1, p2, (255, 0, 0), 5)  # Azul
        
        # Texto da linha do atacante
        mid_x = int((p1[0] + p2[0]) / 2)
        mid_y = int((p1[1] + p2[1]) / 2) + 30  # Um pouco abaixo para não sobrepor
        
        # Adicionar fundo para o texto
        text = "LINHA DO ATACANTE"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        
        # Garantir que o texto esteja visível
        text_x = max(text_size[0]//2 + 5, min(mid_x, w - text_size[0]//2 - 5))
        text_y = max(60, min(mid_y, h - 10))
        
        cv2.rectangle(output, 
                     (text_x - text_size[0]//2 - 5, text_y - 25),
                     (text_x + text_size[0]//2 + 5, text_y + 5),
                     (0, 0, 0), -1)
        
        # Adicionar texto
        cv2.putText(output, text, (text_x - text_size[0]//2, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    
    # 7. Verificar e mostrar status de impedimento se temos atacante e defensor
    if attacker_point and defender_point:
        # Verificar impedimento
        is_offside = check_offside(defender_point, attacker_point, attacking_direction)
        
        # Definir texto e cor com base no resultado
        status_text = "IMPEDIDO!" if is_offside else "POSIÇÃO LEGAL"
        status_color = (0, 0, 255) if is_offside else (0, 255, 0)  # Vermelho para impedido, verde para legal
        
        # Posicionar na parte superior da tela
        status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        status_x = w // 2 - status_size[0] // 2
        status_y = 100
        
        # Desenhar fundo para status com destaque
        cv2.rectangle(output, 
                     (status_x - 20, status_y - 40),
                     (status_x + status_size[0] + 20, status_y + 10),
                     (0, 0, 0), -1)
        
        # Desenhar texto do status
        cv2.putText(output, status_text, (status_x, status_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3, cv2.LINE_AA)
        
        # Marcar jogadores (se temos detecções)
        for det in analysis_detections:
            box = det[0]
            x1, y1, x2, y2 = [int(v) for v in box]
            
            # Posição dos pés (base da caixa)
            foot_point = (int((x1 + x2) / 2), y2)
            
            # Código para determinar impedimento
            # Verificar qual lado da linha o jogador está
            a = p2[1] - p1[1]  # Coeficiente A da equação Ax + By + C = 0
            b_coef = p1[0] - p2[0]  # Coeficiente B
            c = p2[0] * p1[1] - p1[0] * p2[1]  # Coeficiente C
            
            # Calcular de que lado da linha o jogador está
            side = a * foot_point[0] + b_coef * foot_point[1] + c
            
            # Determinar cor baseada na direção de ataque e posição
            if (defender_point and abs(foot_point[0] - defender_point[0]) < 15 and 
                abs(foot_point[1] - defender_point[1]) < 15):
                color = DEFENDER_COLOR
            elif (attacker_point and abs(foot_point[0] - attacker_point[0]) < 15 and 
                  abs(foot_point[1] - attacker_point[1]) < 15):
                color = ATTACKER_COLOR
            else:
                if attacking_direction == 'right':
                    color = OFFSIDE_COLOR if side < 0 else ONSIDE_COLOR
                else:  # attacking_direction == 'left'
                    color = OFFSIDE_COLOR if side > 0 else ONSIDE_COLOR
            
            # Desenhar retângulo e círculo
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.circle(output, foot_point, 12, (0, 0, 0), -1)
            cv2.circle(output, foot_point, 10, color, -1)
    
    # 8. Adicionar instruções
    instructions_box = np.zeros((180, 400, 3), dtype=np.uint8)
    output[10:190, 10:410] = instructions_box
    
    instructions = [
        "Espaço: Pausar/Continuar",
        "S: Selecionar frame para análise",
        "A/D: Mudar direção (esq/dir)",
        "R: Resetar análise",
        "Q: Sair"
    ]
    
    for i, text in enumerate(instructions):
        cv2.putText(output, text, (20, 40 + i*30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    
    # 9. Adicionar status
    if analysis_mode:
        status_messages = [
            "SELECIONE PONTO 1 DA LINHA",
            "SELECIONE PONTO 2 DA LINHA",
            "SELECIONE O DEFENSOR",
            "SELECIONE O ATACANTE",
            "ANÁLISE COMPLETA"
        ]
        
        step_text = f"PASSO {selection_step}/4: "
        if selection_step <= 4:
            step_text += status_messages[selection_step-1]
        else:
            step_text = "ANÁLISE COMPLETA"
        
        cv2.rectangle(output, (10, h-50), (w-10, h-10), (0, 0, 0), -1)
        cv2.putText(output, step_text, (20, h-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Mostrar direção de ataque
        direction_text = f"DIREÇÃO: {'←' if attacking_direction == 'left' else '→'}"
        cv2.putText(output, direction_text, (w-300, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    
    return output

def mouse_click_handler(event, x, y, flags, param):
    """Manipulador de eventos do mouse simplificado."""
    global field_points, defender_point, attacker_point, offside_line, selection_step
    
    if event == cv2.EVENT_LBUTTONDOWN and analysis_mode:
        print(f"Clique em ({x}, {y}), passo atual: {selection_step}")
        
        if selection_step == 1:  # Primeiro ponto da linha de campo
            field_points.append((x, y))
            selection_step = 2
            print(f"Ponto 1 definido: ({x}, {y})")
            
        elif selection_step == 2:  # Segundo ponto da linha de campo
            field_points.append((x, y))
            selection_step = 3
            print(f"Ponto 2 definido: ({x}, {y})")
            
        elif selection_step == 3:  # Defensor
            defender_point = (x, y)
            selection_step = 4
            print(f"Defensor definido: ({x}, {y})")
            
        elif selection_step == 4:  # Atacante
            attacker_point = (x, y)
            selection_step = 5
            print(f"Atacante definido: ({x}, {y})")
            
            # Calcular linha de impedimento
            if len(field_points) >= 2 and defender_point:
                try:
                    print("\nCalculando linha de impedimento...")
                    offside_line = calculate_offside_line(field_points, defender_point)
                    print(f"Linha de impedimento calculada: {offside_line}")
                except Exception as e:
                    print(f"Erro ao calcular linha de impedimento: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Após qualquer clique, redraw o frame
        if analysis_frame is not None:
            display_frame = draw_simple_frame(analysis_frame)
            cv2.imshow("VAR Analysis System", display_frame)
            cv2.waitKey(1)  # Força atualização da janela

def main():
    global paused, current_frame_num, frame_offset, analysis_frame, analysis_detections
    global field_points, defender_point, attacker_point, offside_line
    global attacking_direction, analysis_mode, selection_step
    
    # Carregar modelo YOLO
    try:
        model = YOLO("yolov8n.pt")
        model.to(device)
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return
    
    # Abrir o vídeo
    video_path = "data/match_video.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return
    
    # Informações do vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Vídeo carregado: {width}x{height} a {fps} fps, {total_frames} frames")
    
    # Configurar janela e callback
    window_name = "VAR Analysis System"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_click_handler)
    
    # Variáveis para o loop principal
    last_processed_frame = -1
    running = True
    
    # Loop principal
    while running:
        # Determinar qual frame processar
        if paused:
            frame_num = current_frame_num + frame_offset
        else:
            frame_num = current_frame_num
            current_frame_num += 1
        
        # Verificar limites
        frame_num = max(0, min(frame_num, total_frames - 1))
        
        # Processar apenas se for um novo frame ou estamos em modo análise
        if last_processed_frame != frame_num or analysis_mode:
            # Obter o frame do vídeo
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                print("Erro ao ler o frame")
                break
            
            last_processed_frame = frame_num
            
            # Usar YOLO para detectar pessoas
            results = model(frame, classes=[0], conf=CONFIDENCE_THRESHOLD)
            
            # Preparar detecções
            detections = []
            if len(results) > 0 and hasattr(results[0], 'boxes'):
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    conf = boxes.conf[i].cpu().numpy()
                    cls = boxes.cls[i].cpu().numpy()
                    detections.append((box, conf, cls))
            
            # Se pressionamos S e estamos em modo de análise, use este frame
            if analysis_mode and analysis_frame is None:
                analysis_frame = frame.copy()
                analysis_detections = detections.copy()
                print(f"Frame capturado para análise com {len(analysis_detections)} detecções")
            
            # Preparar o frame para exibição
            if analysis_mode and analysis_frame is not None:
                # Usar o frame de análise com todas as marcações
                display_frame = draw_simple_frame(analysis_frame)
            else:
                # Mostrar o frame atual com detecções básicas
                display_frame = frame.copy()
                for det in detections:
                    box = det[0]
                    x1, y1, x2, y2 = [int(v) for v in box]
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Adicionar informações básicas
                if not analysis_mode:
                    cv2.putText(display_frame, "Pressione 'S' para iniciar análise", 
                                (20, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Adicionar número do frame
            frame_text = f"Frame: {frame_num+1}/{total_frames}" + (" (PAUSADO)" if paused else "")
            cv2.putText(display_frame, frame_text, (width-350, height-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Mostrar o frame
            cv2.imshow(window_name, display_frame)
        
        # Verificar teclas pressionadas
        key = cv2.waitKey(0 if paused else 30) & 0xFF
        
        if key == ord(' '):  # Espaço - Pausar/continuar
            paused = not paused
            print(f"Vídeo {'pausado' if paused else 'continuando'}")
            
        elif key == ord('s'):  # S - Selecionar frame para análise
            if not analysis_mode:
                analysis_mode = True
                selection_step = 1
                field_points = []
                defender_point = None
                attacker_point = None
                offside_line = None
                # Frame será capturado no próximo ciclo
                analysis_frame = None
                print("Análise iniciada! Selecione o primeiro ponto da linha de campo")
            
        elif key == ord('a'):  # A - Direção de ataque para esquerda
            attacking_direction = 'left'
            print("Direção de ataque: para a ESQUERDA")
            if analysis_frame is not None:
                display_frame = draw_simple_frame(analysis_frame)
                cv2.imshow(window_name, display_frame)
            
        elif key == ord('d'):  # D - Direção de ataque para direita
            attacking_direction = 'right'
            print("Direção de ataque: para a DIREITA")
            if analysis_frame is not None:
                display_frame = draw_simple_frame(analysis_frame)
                cv2.imshow(window_name, display_frame)
            
        elif key == ord('r'):  # R - Resetar análise
            if analysis_mode:
                selection_step = 1
                field_points = []
                defender_point = None
                attacker_point = None
                offside_line = None
                print("Análise resetada! Selecione o primeiro ponto da linha de campo")
                if analysis_frame is not None:
                    display_frame = draw_simple_frame(analysis_frame)
                    cv2.imshow(window_name, display_frame)
            
        elif key == ord('q'):  # Q - Sair
            if analysis_frame is not None and offside_line is not None:
                try:
                    # Gerar imagem final de análise
                    final_frame = draw_simple_frame(analysis_frame)
                    cv2.imwrite("offside_analysis.jpg", final_frame)
                    print("Análise salva em offside_analysis.jpg")
                except Exception as e:
                    print(f"Erro ao salvar análise: {e}")
            running = False
            
        elif key == ord('.') and paused:  # . - Avançar um frame
            frame_offset += 1
            
        elif key == ord(',') and paused:  # , - Voltar um frame
            frame_offset -= 1
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
