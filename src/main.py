import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizando dispositivo: {device}")

# Configurações
CONFIDENCE_THRESHOLD = 0.35  # Limiar de confianca para detecção
LINE_COLOR = (0, 0, 255)     # Cor da linha de impedimento (BGR: vermelho)
OFFSIDE_COLOR = (0, 0, 255)  # Cor para jogadores em possivel impedimento (vermelho)
ONSIDE_COLOR = (0, 255, 0)   # Cor para jogadores em posicao regular (verde)
DEFENDER_COLOR = (255, 165, 0)  # Cor para o ultimo defensor (laranja)
ATTACKER_COLOR = (255, 0, 255)  # Cor para o atacante (magenta)

# Variáveis globais - simplificadas
paused = False
current_frame_num = 0
frame_offset = 0
analysis_frame = None  # Frame atual para analise
analysis_detections = []
field_points = []      # Linha lateral (opcional)
area_points = []       # Linha da área (opcional)
goal_line_points = []  # Renomeado na lógica, mas mantido como variável por compatibilidade
calibration_complete = False
defender_point = None
attacker_point = None
attacking_direction = 'right'  # Direção de ataque (left/right)
analysis_mode = False
selection_step = 0
playback_speed_normal = 1.0
show_instructions = True

# Zoom e pan
zoom_level = 1.0
zoom_max = 5.0
zoom_min = 1.0
zoom_step = 0.2
pan_x = 0
pan_y = 0
is_panning = False
pan_start_x = 0
pan_start_y = 0

# Reprodução
playback_mode = False
playback_frames = []
playback_index = 0
playback_speed = 1

# Calibração de perspectiva
field_width = 105
field_height = 68
penalty_area_width = 16.5
penalty_area_height = 40.3
goal_width = 7.32
homography_matrix = None

# Ajuste fino de perspectiva
perspective_correction = 45
calibration_settings = {}

# Slow motion
slow_motion_mode = False
slow_motion_factor = 0.25

# Ponto de fuga automático
auto_vanishing_point = None
use_auto_vanishing = True

############################################
#                  FUNÇÕES                 #
############################################

def calculate_homography_from_field_lines():
    """
    Calcula a matriz de homografia de forma mais robusta, usando todos os pontos disponíveis
    para mapear entre as coordenadas da imagem e as coordenadas reais do campo.
    """
    global homography_matrix, calibration_complete
    
    required_points = 2  # Mínimo de linhas para uma calibração básica

    if len(field_points) < required_points:
        print("Erro: Não há pontos suficientes para calcular a homografia (mínimo: linha lateral).")
        return None
    
    try:
        src_points = []
        dst_points = []
        
        # 1. Adicionar pontos da linha lateral (base para qualquer calibração)
        if len(field_points) >= 2:
            src_points.append(field_points[0])
            src_points.append(field_points[1])
            # Mapeamento para o modelo do campo (em metros)
            dst_points.append((0, 0))
            dst_points.append((field_width, 0))
        
        # 2. Adicionar pontos da linha da área (enriquece a calibração)
        if len(area_points) >= 2:
            src_points.append(area_points[0])
            src_points.append(area_points[1])
            # A linha da área fica a uma distância conhecida da linha lateral
            dst_points.append((0, penalty_area_height/2))
            dst_points.append((penalty_area_width, penalty_area_height/2))
        
        # 3. Adicionar pontos da linha de gol (especialmente importantes para impedimento)
        if len(goal_line_points) >= 2:
            src_points.append(goal_line_points[0])
            src_points.append(goal_line_points[1])
            # Linha de gol (depende da direção de ataque)
            if attacking_direction == 'right':
                dst_points.append((0, field_height))
                dst_points.append((goal_width, field_height))
            else:
                dst_points.append((field_width - goal_width, field_height))
                dst_points.append((field_width, field_height))
        
        # Converter para numpy arrays - formato esperado por findHomography
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)
        
        # Calcular a homografia usando RANSAC para maior robustez
        H, status = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
        
        if H is not None:
            homography_matrix = H
            calibration_complete = True
            
            # Verificação de qualidade da homografia
            error_sum = 0
            for i in range(len(src_points)):
                src_pt = np.array([[[src_points[i][0], src_points[i][1]]]], dtype=np.float32)
                dst_pt_expected = dst_points[i]
                
                # Projetar usando a homografia e comparar com o esperado
                dst_pt_calc = cv2.perspectiveTransform(src_pt, H)[0][0]
                error = np.sqrt((dst_pt_calc[0] - dst_pt_expected[0])**2 + 
                                (dst_pt_calc[1] - dst_pt_expected[1])**2)
                error_sum += error
            
            avg_error = error_sum / len(src_points)
            print(f"Matriz de homografia calculada! Erro médio: {avg_error:.2f}m")
            
            if avg_error > 2.0:  # Erro grande pode indicar pontos mal selecionados
                print("ATENÇÃO: Erro de calibração alto. Considere rever os pontos marcados.")
                
            return homography_matrix
        else:
            print("Erro: Não foi possível calcular a homografia com os pontos fornecidos.")
            homography_matrix = None
            calibration_complete = False
            return None
            
    except Exception as e:
        print(f"Erro ao calcular homografia: {e}")
        import traceback
        traceback.print_exc()
        calibration_complete = False
        homography_matrix = None
        return None

def calculate_offside_line_fallback(defender):
    """Linha inclinada simples se não houver linha de gol."""
    dx, dy = defender
    tilt = -0.6 if attacking_direction == 'right' else 0.6
    p1 = (int(dx + tilt * 1500), int(dy + 1500))
    p2 = (int(dx - tilt * 1500), int(dy - 1500))
    return [p1, p2]

def calculate_attacker_line_fallback(attacker):
    """Linha inclinada simples se não houver linha de gol."""
    ax, ay = attacker
    tilt = -0.6 if attacking_direction == 'right' else 0.6
    p1 = (int(ax + tilt * 1500), int(ay + 1500))
    p2 = (int(ax - tilt * 1500), int(ay - 1500))
    return [p1, p2]

def image_to_field(px, py, H_inv):
    """
    Transforma um ponto na imagem (px, py) para coordenadas no campo (X, Y)
    usando a homografia inversa.
    """
    try:
        # Formatar o ponto conforme esperado pelo OpenCV (array 3D)
        pt_img = np.array([[[px, py]]], dtype=np.float32)  # shape (1,1,2)
        
        # Aplicar a transformação de perspectiva inversa
        pt_field = cv2.perspectiveTransform(pt_img, H_inv)  # shape (1,1,2)
        
        # Extrair as coordenadas resultantes
        X = float(pt_field[0, 0, 0])
        Y = float(pt_field[0, 0, 1])
        
        return (X, Y)
        
    except Exception as e:
        print(f"Erro ao transformar ponto para o campo: {e}")
        import traceback
        traceback.print_exc()
        return None

def field_to_image(X, Y, H):
    """
    Transforma um ponto do campo (X, Y) para coordenadas na imagem (px, py)
    usando a homografia direta.
    """
    try:
        # Formatar o ponto conforme esperado pelo OpenCV (array 3D)
        pt_field = np.array([[[X, Y]]], dtype=np.float32)
        
        # Aplicar a transformação de perspectiva
        pt_img = cv2.perspectiveTransform(pt_field, H)  # shape (1,1,2)
        
        # Extrair as coordenadas resultantes
        x_img = int(pt_img[0, 0, 0])
        y_img = int(pt_img[0, 0, 1])
        
        return (x_img, y_img)
        
    except Exception as e:
        print(f"Erro ao transformar ponto para a imagem: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_offside_line(defender):
    """
    Calcula a linha de impedimento com EXATAMENTE o mesmo ângulo da linha de referência.
    Usa a direção da linha de referência para criar uma linha que passa pelo defensor.
    """
    global goal_line_points
    
    # Se temos a linha de referência definida
    if len(goal_line_points) >= 2:
        try:
            # Obter os pontos da linha de referência (coordenadas originais)
            g1, g2 = goal_line_points[0], goal_line_points[1]
            gx1, gy1 = g1
            gx2, gy2 = g2
            
            # A linha de impedimento deve passar pelo pé do defensor (coordenadas originais)
            dx, dy = defender
            
            # Calcular a inclinação da linha de referência
            if abs(gx2 - gx1) < 1e-6:  # Evitar divisão por zero se linha for vertical
                # Linha de referência é vertical
                # Retornar uma linha vertical infinita passando pelo defensor
                return [(dx, -1000000), (dx, 1000000)]
                
            # Calcular a inclinação (slope) da linha de referência
            slope = (gy2 - gy1) / (gx2 - gx1)
            
            # Calcular o ponto de interseção b onde a linha cruza o eixo y: y = slope*x + b
            # Para a linha de referência: gy1 = slope*gx1 + b_goal, então b_goal = gy1 - slope*gx1
            # Para a linha do defensor: dy = slope*dx + b_def, então b_def = dy - slope*dx
            b_def = dy - slope * dx
            
            # Agora temos a equação da linha y = slope*x + b_def
            # Escolher dois pontos suficientemente distantes nessa linha
            # para garantir que ela seja visível em qualquer zoom
            x_min = dx - 1000000  # Um valor grande negativo
            y_min = slope * x_min + b_def
            
            x_max = dx + 1000000  # Um valor grande positivo
            y_max = slope * x_max + b_def
            
            # Retornar os dois pontos que definem a linha de impedimento
            p1 = (int(x_min), int(y_min))
            p2 = (int(x_max), int(y_max))
            
            # Log simplificado para não sobrecarregar o console
            print(f"Linha de impedimento paralela à linha de referência calculada")
            return [p1, p2]
            
        except Exception as e:
            print(f"Erro ao calcular linha paralela à linha de referência: {e}")
            import traceback
            traceback.print_exc()
    
    # Se não temos linha de referência ou houve erro, usar método alternativo
    print("Usando método alternativo para linha de impedimento")
    return calculate_offside_line_fallback(defender)

def calculate_attacker_line(attacker):
    """
    Calcula a linha do atacante com EXATAMENTE o mesmo ângulo da linha de referência.
    Usa a direção da linha de referência para criar uma linha que passa pelo atacante.
    """
    global goal_line_points
    
    # Se temos a linha de referência definida
    if len(goal_line_points) >= 2:
        try:
            # Obter os pontos da linha de referência (coordenadas originais)
            g1, g2 = goal_line_points[0], goal_line_points[1]
            gx1, gy1 = g1
            gx2, gy2 = g2
            
            # A linha do atacante deve passar pelo pé do atacante (coordenadas originais)
            ax, ay = attacker
            
            # Calcular a inclinação da linha de referência
            if abs(gx2 - gx1) < 1e-6:  # Evitar divisão por zero se linha for vertical
                # Linha de referência é vertical
                # Retornar uma linha vertical infinita passando pelo atacante
                return [(ax, -1000000), (ax, 1000000)]
                
            # Calcular a inclinação (slope) da linha de referência
            slope = (gy2 - gy1) / (gx2 - gx1)
            
            # Calcular o ponto de interseção b onde a linha cruza o eixo y: y = slope*x + b
            # Para a linha do atacante: ay = slope*ax + b_att, então b_att = ay - slope*ax
            b_att = ay - slope * ax
            
            # Agora temos a equação da linha y = slope*x + b_att
            # Escolher dois pontos suficientemente distantes nessa linha
            # para garantir que ela seja visível em qualquer zoom
            x_min = ax - 1000000  # Um valor grande negativo
            y_min = slope * x_min + b_att
            
            x_max = ax + 1000000  # Um valor grande positivo
            y_max = slope * x_max + b_att
            
            # Retornar os dois pontos que definem a linha do atacante
            p1 = (int(x_min), int(y_min))
            p2 = (int(x_max), int(y_max))
            
            # Log simplificado para não sobrecarregar o console
            print(f"Linha do atacante paralela à linha de referência calculada")
            return [p1, p2]
            
        except Exception as e:
            print(f"Erro ao calcular linha do atacante paralela à linha de referência: {e}")
            import traceback
            traceback.print_exc()
    
    # Se não temos linha de referência ou houve erro, usar método alternativo
    print("Usando método alternativo para linha do atacante")
    return calculate_attacker_line_fallback(attacker)

def check_offside(defender_point, attacker_point, attacking_direction):
    """
    Verifica impedimento com alta precisão usando coordenadas reais do campo.
    Método preferido: comparar as posições X dos jogadores no CAMPO, não na imagem.
    """
    global homography_matrix, calibration_complete, goal_line_points
    
    # Método principal: usar a linha de referência para projeção correta
    if len(goal_line_points) >= 2:
        try:
            # Usar o vetor da linha de referência
            g1, g2 = goal_line_points[0], goal_line_points[1]
            gx1, gy1 = g1
            gx2, gy2 = g2
            
            # Calcular vetor perpendicular à linha de referência (vetor que aponta para dentro do campo)
            if attacking_direction == 'right':
                normal_x = -(gy2 - gy1)
                normal_y = gx2 - gx1
            else:
                normal_x = gy2 - gy1
                normal_y = -(gx2 - gx1)
            
            # Normalizar o vetor
            length = np.sqrt(normal_x**2 + normal_y**2)
            if length < 1e-6:
                print("Vetor de linha de referência quase nulo. Tentando outro método.")
                raise ValueError("Vetor quase nulo")
                
            normal_x /= length
            normal_y /= length
            
            # Projetar jogadores nessa direção
            defender_proj = defender_point[0]*normal_x + defender_point[1]*normal_y
            attacker_proj = attacker_point[0]*normal_x + attacker_point[1]*normal_y
            
            # Comparar as projeções
            diff_proj = attacker_proj - defender_proj
            is_offside = diff_proj > 0
            
            print(f"Verificação de impedimento (projeção pela linha de referência):")
            print(f"Projeção defensor: {defender_proj:.2f}")
            print(f"Projeção atacante: {attacker_proj:.2f}")
            print(f"Diferença: {diff_proj:.2f} pixels, {'IMPEDIDO' if is_offside else 'POSIÇÃO LEGAL'}")
            
            return is_offside
        except Exception as e:
            print(f"Erro ao usar projeção pela linha de referência: {e}")
    
    # Se não conseguimos usar a linha de referência, tentamos os outros métodos
    # Método alternativo 1: transformar para coordenadas do campo
    if homography_matrix is not None and calibration_complete:
        try:
            # Calcular a homografia inversa (imagem → campo)
            H_inv = np.linalg.inv(homography_matrix)
            
            # Transformar os pontos dos jogadores para coordenadas do campo
            def_field = image_to_field(defender_point[0], defender_point[1], H_inv)
            att_field = image_to_field(attacker_point[0], attacker_point[1], H_inv)
            
            if def_field is not None and att_field is not None:
                X_def = def_field[0]
                X_att = att_field[0]
                Y_def = def_field[1]
                Y_att = att_field[1]
                
                # Verificar se os pontos transformados fazem sentido
                # (evita erros em homografias mal calculadas)
                if 0 <= X_def <= field_width and 0 <= X_att <= field_width:
                    # No campo: impedimento é quando X_atacante > X_defensor (ataque para direita)
                    # ou X_atacante < X_defensor (ataque para esquerda)
                    if attacking_direction == 'right':
                        is_offside = X_att > X_def
                    else:
                        is_offside = X_att < X_def
                    
                    diff_meters = abs(X_att - X_def)
                    print(f"Verificação de impedimento (campo real):")
                    print(f"Defensor: X={X_def:.2f}m, Y={Y_def:.2f}m")
                    print(f"Atacante: X={X_att:.2f}m, Y={Y_att:.2f}m")
                    print(f"Diferença: {diff_meters:.2f}m, {'IMPEDIDO' if is_offside else 'POSIÇÃO LEGAL'}")
                    
                    return is_offside
            
            # Se chegou aqui, algo falhou na transformação - tentar métodos alternativos
            print("Transformação para o campo falhou. Tentando método alternativo.")
        except Exception as e:
            print(f"Erro ao verificar impedimento usando coordenadas do campo: {e}")
    
    # Método alternativo 1: usar linha de referência para calcular vetor perpendicular
    if len(goal_line_points) >= 2:
        try:
            gx1, gy1 = goal_line_points[0]
            gx2, gy2 = goal_line_points[1]
            
            # Calcular vetor perpendicular à linha de referência (vetor que aponta para dentro do campo)
            if attacking_direction == 'right':
                normal_x = -(gy2 - gy1)
                normal_y = gx2 - gx1
            else:
                normal_x = gy2 - gy1
                normal_y = -(gx2 - gx1)
            
            # Normalizar o vetor
            length = np.sqrt(normal_x**2 + normal_y**2)
            if length < 1e-6:
                print("Vetor de linha de referência quase nulo. Tentando outro método.")
                raise ValueError("Vetor quase nulo")
                
            normal_x /= length
            normal_y /= length
            
            # Projetar jogadores nessa direção
            defender_proj = defender_point[0]*normal_x + defender_point[1]*normal_y
            attacker_proj = attacker_point[0]*normal_x + attacker_point[1]*normal_y
            
            # Comparar as projeções
            diff_proj = attacker_proj - defender_proj
            is_offside = diff_proj > 0
            
            print(f"Verificação de impedimento (projeção pela linha de referência):")
            print(f"Projeção defensor: {defender_proj:.2f}")
            print(f"Projeção atacante: {attacker_proj:.2f}")
            print(f"Diferença: {diff_proj:.2f} pixels, {'IMPEDIDO' if is_offside else 'POSIÇÃO LEGAL'}")
            
            return is_offside
        except Exception as e:
            print(f"Erro ao usar projeção pela linha de referência: {e}")
    
    # Método alternativo 2: usar linha da área (mesma lógica do método anterior)
    if len(area_points) >= 2:
        try:
            ax1, ay1 = area_points[0]
            ax2, ay2 = area_points[1]
            
            # Vetor perpendicular à linha da área
            if attacking_direction == 'right':
                normal_x = -(ay2 - ay1)
                normal_y = ax2 - ax1
            else:
                normal_x = ay2 - ay1
                normal_y = -(ax2 - ax1)
            
            # Normalizar
            length = np.sqrt(normal_x**2 + normal_y**2)
            if length < 1e-6:
                print("Vetor de linha da área quase nulo. Tentando outro método.")
                raise ValueError("Vetor quase nulo")
                
            normal_x /= length
            normal_y /= length
            
            # Projetar
            defender_proj = defender_point[0]*normal_x + defender_point[1]*normal_y
            attacker_proj = attacker_point[0]*normal_x + attacker_point[1]*normal_y
            
            diff_proj = attacker_proj - defender_proj
            is_offside = diff_proj > 0
            
            print(f"Verificação de impedimento (projeção pela linha da área):")
            print(f"Diferença: {diff_proj:.2f} pixels, {'IMPEDIDO' if is_offside else 'POSIÇÃO LEGAL'}")
            
            return is_offside
        except Exception as e:
            print(f"Erro ao usar projeção pela linha da área: {e}")
    
    # Método "last-resort": linha lateral
    if len(field_points) >= 2:
        try:
            fx1, fy1 = field_points[0]
            fx2, fy2 = field_points[1]
            
            # Vetor perpendicular à linha lateral
            angle_rad = np.arctan2(fy2 - fy1, fx2 - fx1) + np.pi/2
            perp_x = np.cos(angle_rad)
            perp_y = np.sin(angle_rad)
            
            # Projetar
            defender_proj = defender_point[0]*perp_x + defender_point[1]*perp_y
            attacker_proj = attacker_point[0]*perp_x + attacker_point[1]*perp_y
            
            diff_proj = attacker_proj - defender_proj
            
            # Dependendo da direção de ataque, a interpretação muda
            if attacking_direction == 'right':
                is_offside = diff_proj > 0
            else:
                is_offside = diff_proj < 0
            
            print(f"Verificação de impedimento (projeção pela linha lateral):")
            print(f"Diferença: {diff_proj:.2f} pixels, {'IMPEDIDO' if is_offside else 'POSIÇÃO LEGAL'}")
            
            return is_offside
        except Exception as e:
            print(f"Erro ao usar projeção pela linha lateral: {e}")
    
    # Se chegou aqui, não pudemos fazer nenhuma verificação confiável
    print("AVISO: Não foi possível determinar impedimento. Retorno padrão: NÃO IMPEDIDO")
    return False

def clip_line_to_screen(line, frame_width, frame_height):
    """
    Recorta uma linha para caber na tela, mesmo quando os pontos estão muito distantes.
    Calcula as interseções da linha com os limites da tela.
    """
    try:
        p1, p2 = line
        x1, y1 = p1
        x2, y2 = p2
        
        # Converter os pontos para float para evitar overflow nas operações
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        
        # Calcular os coeficientes da equação da linha: ax + by + c = 0
        a = y2 - y1
        b = x1 - x2
        c = x2*y1 - x1*y2
        
        # Se a e b forem ambos muito próximos de zero, a linha é um ponto
        if abs(a) < 1e-6 and abs(b) < 1e-6:
            return [(int(max(0, min(frame_width-1, x1))), int(max(0, min(frame_height-1, y1)))),
                    (int(max(0, min(frame_width-1, x2))), int(max(0, min(frame_height-1, y2))))]
        
        intersections = []
        
        # Borda esquerda x=0
        if abs(b) > 1e-6:  # Evita divisão por zero
            y_left = -c / b
            if 0 <= y_left < frame_height:
                intersections.append((0, int(y_left)))
        
        # Borda direita x=frame_width-1
        if abs(b) > 1e-6:  # Evita divisão por zero
            y_right = -(a*(frame_width-1) + c) / b
            if 0 <= y_right < frame_height:
                intersections.append((frame_width-1, int(y_right)))
        
        # Borda superior y=0
        if abs(a) > 1e-6:  # Evita divisão por zero
            x_top = -c / a
            if 0 <= x_top < frame_width:
                intersections.append((int(x_top), 0))
        
        # Borda inferior y=frame_height-1
        if abs(a) > 1e-6:  # Evita divisão por zero
            x_bottom = -(b*(frame_height-1) + c) / a
            if 0 <= x_bottom < frame_width:
                intersections.append((int(x_bottom), frame_height-1))
        
        # Se temos pelo menos duas interseções, usamos elas
        if len(intersections) >= 2:
            # Limitamos a apenas duas interseções (as primeiras encontradas)
            return [intersections[0], intersections[1]]
        
        # Se chegamos aqui, a linha não cruza a tela ou temos apenas uma interseção
        # Recortamos os pontos originais para dentro da tela
        clipped_line = [
            (max(0, min(frame_width-1, int(x1))), max(0, min(frame_height-1, int(y1)))),
            (max(0, min(frame_width-1, int(x2))), max(0, min(frame_height-1, int(y2))))
        ]
        return clipped_line
    
    except Exception as e:
        print(f"Erro ao recortar linha: {e}")
        # Fallback - retornar os pontos limitados às bordas da tela
        try:
            return [
                (max(0, min(frame_width-1, int(line[0][0]))), max(0, min(frame_height-1, int(line[0][1])))),
                (max(0, min(frame_width-1, int(line[1][0]))), max(0, min(frame_height-1, int(line[1][1]))))
            ]
        except:
            # Se tudo falhar, retornar linha diagonal na tela
            return [(0, 0), (frame_width-1, frame_height-1)]

def draw_simple_frame(frame):
    """Desenha o frame de análise (linhas, pontos etc.)."""
    h, w = frame.shape[:2]
    original_frame = frame.copy()
    
    # Aplicar zoom/pan
    if zoom_level > 1.0:
        zoomed_w = int(w / zoom_level)
        zoomed_h = int(h / zoom_level)
        center_x = w // 2
        center_y = h // 2
        start_x = max(0, min(w - zoomed_w, center_x - zoomed_w // 2 - int(pan_x / zoom_level)))
        start_y = max(0, min(h - zoomed_h, center_y - zoomed_h // 2 - int(pan_y / zoom_level)))
        roi = original_frame[start_y:start_y+zoomed_h, start_x:start_x+zoomed_w]
        frame = cv2.resize(roi, (w, h), interpolation=cv2.INTER_LINEAR)
    
    output = frame.copy()
    
    # MENSAGENS ESPECÍFICAS PARA ETAPAS DE SELEÇÃO DE JOGADORES
    if selection_step == 7 or selection_step == 8:
        # Adicionar uma caixa de instrução específica para zoom na parte superior da tela
        zoom_box = np.zeros((130, 650, 3), dtype=np.uint8)
        output[10:140, w//2-325:w//2+325] = zoom_box
        
        if selection_step == 7:
            title = "SELECIONE O DEFENSOR (use zoom para maior precisão)"
        else:
            title = "SELECIONE O ATACANTE (use zoom para maior precisão)"
            
        cv2.putText(output, title, (w//2-300, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(output, "ZOOM: Use teclas +/- OU i/o para zoom in/out", (w//2-300, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(output, "PAN: Clique e arraste para mover a imagem com zoom", (w//2-300, 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(output, "Pressione Z para resetar zoom e pan", (w//2-300, 125), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Desenhar pontos e linhas (lateral, área, gol)
    # -- Lateral
    for i, pt in enumerate(field_points):
        px = int(pt[0]*zoom_level + pan_x)
        py = int(pt[1]*zoom_level + pan_y)
        cv2.circle(output, (px, py), 10, (255,255,0), -1)
        cv2.putText(output, f"Lateral {i+1}", (px+10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    if len(field_points) >= 2:
        adj_p1 = (int(field_points[0][0]*zoom_level + pan_x), int(field_points[0][1]*zoom_level + pan_y))
        adj_p2 = (int(field_points[1][0]*zoom_level + pan_x), int(field_points[1][1]*zoom_level + pan_y))
        cv2.line(output, adj_p1, adj_p2, (255,255,0), 3)
    
    # -- Área
    for i, pt in enumerate(area_points):
        px = int(pt[0]*zoom_level + pan_x)
        py = int(pt[1]*zoom_level + pan_y)
        cv2.circle(output, (px, py), 10, (0,255,255), -1)
        cv2.putText(output, f"Area {i+1}", (px+10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    if len(area_points) >= 2:
        adj_p1 = (int(area_points[0][0]*zoom_level + pan_x), int(area_points[0][1]*zoom_level + pan_y))
        adj_p2 = (int(area_points[1][0]*zoom_level + pan_x), int(area_points[1][1]*zoom_level + pan_y))
        cv2.line(output, adj_p1, adj_p2, (0,255,255), 3)
    
    # -- Linha de Referência
    for i, pt in enumerate(goal_line_points):
        px = int(pt[0]*zoom_level + pan_x)
        py = int(pt[1]*zoom_level + pan_y)
        cv2.circle(output, (px, py), 10, (255,0,255), -1)
        cv2.putText(output, f"Ref {i+1}", (px+10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)
    if len(goal_line_points) >= 2:
        adj_p1 = (int(goal_line_points[0][0]*zoom_level + pan_x), int(goal_line_points[0][1]*zoom_level + pan_y))
        adj_p2 = (int(goal_line_points[1][0]*zoom_level + pan_x), int(goal_line_points[1][1]*zoom_level + pan_y))
        cv2.line(output, adj_p1, adj_p2, (255,0,255), 3)
        
        # Destacar a linha de referência quando está sendo selecionada
        if selection_step == 5 or selection_step == 6:
            # Desenhar uma linha estendida para mostrar a direção das linhas de impedimento
            if len(goal_line_points) >= 1:
                g1x, g1y = goal_line_points[0]
                if selection_step == 6:  # Temos dois pontos
                    g2x, g2y = goal_line_points[1]
                    
                    # Calcular a inclinação da linha
                    if abs(g2x - g1x) > 1e-6:  # Não vertical
                        slope = (g2y - g1y) / (g2x - g1x)
                        b = g1y - slope * g1x
                        
                        # Desenhar uma linha estendida com essa inclinação
                        x_left = 0
                        y_left = int(slope * x_left + b)
                        x_right = w
                        y_right = int(slope * x_right + b)
                        
                        cv2.line(output, (x_left, y_left), (x_right, y_right), (0, 255, 255), 2, cv2.LINE_DASH)
                        cv2.putText(output, "Previsão da Direção das Linhas", (w//2-200, h//2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    
    # Desenhar defensor
    if defender_point:
        # Aplicar zoom/pan às coordenadas do defensor para exibição
        dx = int(defender_point[0]*zoom_level + pan_x)
        dy = int(defender_point[1]*zoom_level + pan_y)
        cv2.circle(output, (dx,dy), 10, DEFENDER_COLOR, -1)
        cv2.putText(output, "DEFENSOR", (dx+10, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, DEFENDER_COLOR, 2)
    
    # Desenhar atacante
    if attacker_point:
        # Aplicar zoom/pan às coordenadas do atacante para exibição
        ax = int(attacker_point[0]*zoom_level + pan_x)
        ay = int(attacker_point[1]*zoom_level + pan_y)
        cv2.circle(output, (ax,ay), 10, ATTACKER_COLOR, -1)
        cv2.putText(output, "ATACANTE", (ax+10, ay), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ATTACKER_COLOR, 2)
    
    # Destacar jogadores detectados quando estiver nos passos de seleção do defensor ou atacante
    if selection_step == 7 or selection_step == 8:
        for i, det in enumerate(analysis_detections):
            box = det[0]
            x1, y1, x2, y2 = [int(v) for v in box]
            
            # Ajustar coordenadas para zoom e pan
            x1_adj = int(x1*zoom_level + pan_x)
            y1_adj = int(y1*zoom_level + pan_y)
            x2_adj = int(x2*zoom_level + pan_x)
            y2_adj = int(y2*zoom_level + pan_y)
            
            # Calcular o ponto do pé (importante para impedimento)
            foot_point = (int((x1_adj+x2_adj)/2), y2_adj)
            
            # Desenhar caixa delimitadora mais visível
            cv2.rectangle(output, (x1_adj,y1_adj), (x2_adj,y2_adj), (0,255,255), 2)
            
            # Marcar o ponto do pé com um círculo grande para facilitar a seleção
            cv2.circle(output, foot_point, 5, (0,255,255), -1)
            
            # Adicionar número ao jogador para facilitar referência - CORRIGIDO: uso de índice diretamente
            jogador_id = i + 1
            cv2.putText(output, f"#{jogador_id}", (x1_adj, y1_adj-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
    
    # Desenhar linha de impedimento (defensor)
    if defender_point:
        # Calcular linha de impedimento usando as coordenadas originais
        current_offside_line = calculate_offside_line(defender_point)
        
        # Ajustar cada ponto da linha para o zoom e pan atuais
        p1_orig, p2_orig = current_offside_line
        
        # Converter para coordenadas relativas ao frame
        p1_rel = (p1_orig[0], p1_orig[1])
        p2_rel = (p2_orig[0], p2_orig[1])
        
        # Aplicar transformação de zoom e pan
        p1 = (int(p1_rel[0]*zoom_level + pan_x), int(p1_rel[1]*zoom_level + pan_y))
        p2 = (int(p2_rel[0]*zoom_level + pan_x), int(p2_rel[1]*zoom_level + pan_y))
        
        # Recortar para caber na tela
        clipped = clip_line_to_screen([p1, p2], w, h)
        p1, p2 = clipped
        
        # Desenhar a linha
        cv2.line(output, p1, p2, (255,255,255), 7)  # Borda branca
        cv2.line(output, p1, p2, LINE_COLOR, 5)    # Vermelho
        
        # Texto "LINHA DO DEFENSOR"
        mid_x = (p1[0]+p2[0])//2
        mid_y = (p1[1]+p2[1])//2
        text = "LINHA DO DEFENSOR"
        tsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        tx = max(tsize[0]//2+5, min(mid_x, w-tsize[0]//2-5))
        ty = max(25, min(mid_y, h-10))
        cv2.rectangle(output, (tx - tsize[0]//2 -5, ty -25), (tx + tsize[0]//2 +5, ty+5), (0,0,0), -1)
        cv2.putText(output, text, (tx - tsize[0]//2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    
    # Desenhar linha do atacante (sempre paralela ao gol)
    if attacker_point:
        # Calcular linha do atacante usando as coordenadas originais
        current_attacker_line = calculate_attacker_line(attacker_point)
        
        # Ajustar cada ponto da linha para o zoom e pan atuais
        p1_orig, p2_orig = current_attacker_line
        
        # Converter para coordenadas relativas ao frame
        p1_rel = (p1_orig[0], p1_orig[1])
        p2_rel = (p2_orig[0], p2_orig[1])
        
        # Aplicar transformação de zoom e pan
        p1 = (int(p1_rel[0]*zoom_level + pan_x), int(p1_rel[1]*zoom_level + pan_y))
        p2 = (int(p2_rel[0]*zoom_level + pan_x), int(p2_rel[1]*zoom_level + pan_y))
        
        # Recortar para caber na tela
        clipped_att = clip_line_to_screen([p1, p2], w, h)
        p1, p2 = clipped_att
        
        # Desenhar a linha
        cv2.line(output, p1, p2, (255,255,255), 7)  # Borda branca
        cv2.line(output, p1, p2, (255,0,0), 5)     # Azul
        
        # Texto "LINHA DO ATACANTE"
        mid_x = (p1[0]+p2[0])//2
        mid_y = (p1[1]+p2[1])//2 + 30
        text = "LINHA DO ATACANTE"
        tsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        tx = max(tsize[0]//2+5, min(mid_x, w-tsize[0]//2-5))
        ty = max(60, min(mid_y, h-10))
        cv2.rectangle(output, (tx - tsize[0]//2 -5, ty -25), (tx + tsize[0]//2 +5, ty+5), (0,0,0), -1)
        cv2.putText(output, text, (tx - tsize[0]//2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    
    # Verificar impedimento
    if attacker_point and defender_point:
        is_offside = check_offside(defender_point, attacker_point, attacking_direction)
        status_text = "IMPEDIDO!" if is_offside else "POSICAO LEGAL"
        status_color = (0,0,255) if is_offside else (0,255,0)
        
        status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        sx = w//2 - status_size[0]//2
        sy = 100
        cv2.rectangle(output, (sx-20, sy-40), (sx+status_size[0]+20, sy+10), (0,0,0), -1)
        cv2.putText(output, status_text, (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)
        
        # Marcar jogadores detectados
        for det in analysis_detections:
            box = det[0]
            x1, y1, x2, y2 = [int(v) for v in box]
            foot_point = (int((x1+x2)/2), y2)
            
            # Ajustar para zoom e pan
            x1_adj = int(x1*zoom_level + pan_x)
            y1_adj = int(y1*zoom_level + pan_y)
            x2_adj = int(x2*zoom_level + pan_x)
            y2_adj = int(y2*zoom_level + pan_y)
            foot_point_adj = (int((x1_adj+x2_adj)/2), y2_adj)
            
            # Checar de que lado da offside_line ele está
            a = p2[1] - p1[1]
            b = p1[0] - p2[0]
            c = p2[0]*p1[1] - p1[0]*p2[1]
            side = a*foot_point[0] + b*foot_point[1] + c
            
            if defender_point and abs(foot_point[0]-defender_point[0])<15 and abs(foot_point[1]-defender_point[1])<15:
                color = DEFENDER_COLOR
            elif attacker_point and abs(foot_point[0]-attacker_point[0])<15 and abs(foot_point[1]-attacker_point[1])<15:
                color = ATTACKER_COLOR
            else:
                if attacking_direction == 'right':
                    color = OFFSIDE_COLOR if side<0 else ONSIDE_COLOR
                else:
                    color = OFFSIDE_COLOR if side>0 else ONSIDE_COLOR
            
            cv2.rectangle(output, (x1_adj,y1_adj), (x2_adj,y2_adj), (0,0,0), 4)
            cv2.rectangle(output, (x1_adj,y1_adj), (x2_adj,y2_adj), color, 2)
            cv2.circle(output, foot_point_adj, 12, (0,0,0), -1)
            cv2.circle(output, foot_point_adj, 10, color, -1)
    
    # Instruções
    if show_instructions:
        instructions_box = np.zeros((320, 450, 3), dtype=np.uint8)
        output[10:330, 10:460] = instructions_box
        if analysis_mode:
            if selection_step <= 6:
                instructions = [
                    "Espaco: Pausar/Continuar",
                    "S: Selecionar frame p/ analise",
                    "A/D: Direcao esq/dir",
                    "R: Reset analise",
                    "N: Pular calibracao atual",
                    "+/-: Velocidade video",
                    "H: Ocultar instrucoes",
                    "Q: Sair",
                    "",
                    "CALIBRACAO:",
                    "1-2: Linha lateral (amarelo)",
                    "3-4: Linha da area (ciano)",
                    "5-6: Linha de referência (magenta)",
                    "7: Defensor, 8: Atacante"
                ]
            else:
                instructions = [
                    "Espaco: Pausar/Continuar",
                    "P: Iniciar/parar reproducao",
                    "+/-: Velocidade video",
                    "R: Reset analise",
                    "Z: Reset zoom/pan",
                    "H: Ocultar instrucoes",
                    "Mouse: Pan/zoom (roda)",
                    "[/]: Ajuste perspectiva",
                    "F: Ponto de fuga AUTO/MANUAL",
                    "C/L: Salvar/Carregar cal.",
                    "E: Exportar imagem",
                    "V: Exportar video",
                    "Q: Sair"
                ]
        else:
            instructions = [
                "Espaco: Pausar/Continuar",
                "S: Iniciar analise",
                ",/.: Voltar/Avancar frame",
                "+/-: Velocidade video",
                "H: Ocultar instrucoes",
                "Q: Sair"
            ]
        
        for i, txt in enumerate(instructions):
            cv2.putText(output, txt, (20, 40 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
    else:
        cv2.putText(output, "H: Mostrar instrucoes", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
    
    # Zoom info
    zoom_text = f"Zoom: {zoom_level:.1f}x"
    cv2.putText(output, zoom_text, (w-150, h-80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    
    # Indicador visual de zoom mais proeminente
    if analysis_mode:
        # Adicionar uma barra de zoom visível
        max_zoom = 5.0
        bar_width = int(200 * (zoom_level / max_zoom))
        bar_height = 15
        
        # Fundo da barra (cinza)
        cv2.rectangle(output, (w-220, h-60), (w-20, h-45), (80,80,80), -1)
        
        # Barra de zoom (amarelo)
        cv2.rectangle(output, (w-220, h-60), (w-220+bar_width, h-45), (0,255,255), -1)
        
        # Texto informativo
        cv2.putText(output, "Zoom: +/- ou i/o", (w-220, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(output, "Pan: clique e arraste", (w-220, h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    
    # Status
    if analysis_mode:
        if selection_step == 1:
            status_text = "SELECIONE PONTO 1 DA LINHA LATERAL"
        elif selection_step == 2:
            status_text = "SELECIONE PONTO 2 DA LINHA LATERAL"
        elif selection_step == 3:
            status_text = "SELECIONE PONTO 1 DA LINHA DA AREA (ou 'N' p/ pular)"
        elif selection_step == 4:
            status_text = "SELECIONE PONTO 2 DA LINHA DA AREA (ou 'N' p/ pular)"
        elif selection_step == 5:
            status_text = "SELECIONE PONTO 1 DA LINHA DE REFERÊNCIA (QUALQUER LINHA VISÍVEL)"
        elif selection_step == 6:
            status_text = "SELECIONE PONTO 2 DA LINHA DE REFERÊNCIA (MESMA LINHA DO PONTO 1)"
        elif selection_step == 7:
            status_text = "SELECIONE O DEFENSOR (use zoom/pan para maior precisão)"
        elif selection_step == 8:
            status_text = "SELECIONE O ATACANTE (use zoom/pan para maior precisão)"
        else:
            status_text = "ANALISE COMPLETA"
        
        cv2.rectangle(output, (10,h-50), (w-10,h-10), (0,0,0), -1)
        cv2.putText(output, status_text, (20,h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        
        # Adicionar instruções específicas para seleção da linha de referência
        if selection_step == 5 or selection_step == 6:
            cv2.putText(output, "MARQUE PRECISAMENTE NOS EXTREMOS DA LINHA MAIS VISÍVEL DISPONÍVEL", 
                       (20, h-80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)
            cv2.putText(output, "As linhas de impedimento seguirão EXATAMENTE este ângulo", 
                       (20, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)
        
        direction_text = f"DIRECAO: {'ESQUERDA' if attacking_direction=='left' else 'DIREITA'}"
        cv2.putText(output, direction_text, (w-300,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        
        if calibration_complete:
            cv2.putText(output, "CALIBRACAO COMPLETA", (w-300,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        elif len(field_points) >= 2:
            cv2.putText(output, "LINHA LATERAL OK", (w-300,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
            if len(area_points) >= 2:
                cv2.putText(output, "LINHA DA AREA OK", (w-300,90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            if len(goal_line_points) >= 2:
                cv2.putText(output, "LINHA DE REFERÊNCIA OK", (w-300,120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 2)
    
    return output

def mouse_click_handler(event, x, y, flags, param):
    """
    Handler do mouse para selecionar pontos (lateral, area, gol, defensor, atacante)
    e também para pan/zoom.
    """
    global field_points, area_points, goal_line_points
    global defender_point, attacker_point, selection_step
    global zoom_level, pan_x, pan_y, is_panning, pan_start_x, pan_start_y
    global homography_matrix, calibration_complete, analysis_mode
    global analysis_frame
    
    # Corrigir coordenadas para o zoom e pan
    original_x = int((x - pan_x)/zoom_level)
    original_y = int((y - pan_y)/zoom_level)
    
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse clicado em ({x},{y}) -> Original: ({original_x},{original_y})")
        
        # Verificar se estamos no modo de análise
        if analysis_mode:
            # Fluxo de seleção dos pontos
            if selection_step == 1:  # Ponto 1 da linha lateral
                field_points = [(original_x, original_y)]
                selection_step = 2
                print("Ponto 1 da linha lateral marcado. Marque o ponto 2.")
            elif selection_step == 2:  # Ponto 2 da linha lateral
                field_points.append((original_x, original_y))
                selection_step = 3
                print("Ponto 2 da linha lateral marcado. Agora marque o ponto 1 da área.")
            elif selection_step == 3:  # Ponto 1 da área
                area_points = [(original_x, original_y)]
                selection_step = 4
                print("Ponto 1 da área marcado. Marque o ponto 2 da área.")
            elif selection_step == 4:  # Ponto 2 da área
                area_points.append((original_x, original_y))
                selection_step = 5
                print("Ponto 2 da área marcado. Agora marque o ponto 1 da linha de referência.")
                # Tentar calcular homografia preliminar
                if len(field_points) >= 2 and len(area_points) >= 2:
                    calculate_homography_from_field_lines()
            elif selection_step == 5:  # Ponto 1 da linha de referência
                # IMPORTANTE: Aqui definimos o ponto 1 da linha de referência
                # Isto é crítico para o cálculo correto das linhas de impedimento
                goal_line_points = [(original_x, original_y)]
                selection_step = 6
                print("Ponto 1 da linha de referência marcado. Marque o ponto 2 (na mesma linha).")
                print("IMPORTANTE: Marque pontos PRECISOS na mesma linha - uma linha visível qualquer que esteja disponível.")
                print("DICA: Use a linha do gol, da área pequena, da área grande ou qualquer outra linha visível.")
            elif selection_step == 6:  # Ponto 2 da linha de referência
                # IMPORTANTE: Aqui definimos o ponto 2 da linha de referência
                # Os dois pontos juntos definem a direção da linha de impedimento
                goal_line_points.append((original_x, original_y))
                selection_step = 7
                # Recalcular homografia final com todos os pontos
                calculate_homography_from_field_lines()
                print("Ponto 2 da linha de referência marcado. Agora selecione o defensor.")
                print("\n=== INSTRUÇÕES DE ZOOM ===")
                print("Para dar ZOOM: Use as teclas + e - ou i e o")
                print("Para MOVER a imagem (PAN): Clique e arraste com o mouse")
                print("Para RESETAR zoom e pan: Pressione Z")
                print("===========================\n")
                # Resetar zoom e pan para facilitar a visualização inicial
                zoom_level = 1.0
                pan_x = 0
                pan_y = 0
                print("Você pode usar o zoom (teclas +/- ou i/o) e o pan (arrastar) para selecionar os jogadores com mais precisão.")
                print("Use as teclas + e - para zoom, ou role a roda do mouse.")
            elif selection_step == 7:  # Defensor
                # Nas etapas 7 e 8, permitimos usar zoom para maior precisão
                # Aqui é crítico usar as coordenadas originais (sem o zoom aplicado)
                defender_point = (original_x, original_y)
                selection_step = 8
                print("Defensor marcado. Agora selecione o atacante.")
                print(f"Coordenadas originais do defensor: {defender_point}")
                print("Continue usando zoom/pan se necessário para maior precisão.")
            elif selection_step == 8:  # Atacante
                # Aqui é crítico usar as coordenadas originais (sem o zoom aplicado)
                attacker_point = (original_x, original_y)
                print(f"Coordenadas originais do atacante: {attacker_point}")
                selection_step = 9
                print("Atacante marcado. Análise completa!")
            else:
                # Se já passamos da etapa 8, iniciar pan em qualquer clique
                is_panning = True
                pan_start_x = x
                pan_start_y = y
        else:
            # Se não está em modo de análise, iniciar pan
            is_panning = True
            pan_start_x = x
            pan_start_y = y
        
        # Atualizar a visualização
        if analysis_frame is not None:
            display_frame = draw_simple_frame(analysis_frame)
            cv2.imshow("VAR Analysis System", display_frame)
            cv2.waitKey(1)
    
    elif event == cv2.EVENT_LBUTTONUP:
        # Finalizar pan quando o botão é solto
        is_panning = False
    
    elif event == cv2.EVENT_MOUSEMOVE and is_panning:
        # Atualizar pan enquanto o mouse se move com botão pressionado
        pan_x += (x - pan_start_x)
        pan_y += (y - pan_start_y)
        pan_start_x = x
        pan_start_y = y
        
        # Atualizar a visualização durante o pan
        if analysis_frame is not None:
            display_frame = draw_simple_frame(analysis_frame)
            cv2.imshow("VAR Analysis System", display_frame)
            cv2.waitKey(1)
    
    # Suporte para evento de roda de mouse (MOUSEWHEEL) em Windows e macOS
    elif event == cv2.EVENT_MOUSEWHEEL:
        # Windows: flags > 0 para zoom in, < 0 para zoom out
        if flags > 0:
            zoom_level = min(zoom_max, zoom_level + zoom_step)
            print(f"Zoom in: {zoom_level:.1f}x")
        else:
            zoom_level = max(zoom_min, zoom_level - zoom_step)
            print(f"Zoom out: {zoom_level:.1f}x")
        
        # Atualizar a visualização com novo zoom
        if analysis_frame is not None:
            display_frame = draw_simple_frame(analysis_frame)
            cv2.imshow("VAR Analysis System", display_frame)
            cv2.waitKey(1)
            
        print(f"Zoom ajustado para {zoom_level:.1f}x. Use pan (arrastar) para posicionar a imagem.")
    
    # Suporte para eventos de trackpad no macOS
    elif event == 10 or event == cv2.EVENT_MOUSEHWHEEL:  # macOS usa EVENT_MOUSEHWHEEL (10)
        # macOS: flags negativo para zoom in, positivo para zoom out (inverso do Windows)
        direction = -1 if flags < 0 else 1  # Inverter direção no macOS
        
        zoom_level = min(zoom_max, max(zoom_min, zoom_level + direction * zoom_step))
        print(f"Zoom (macOS): {zoom_level:.1f}x")
        
        # Atualizar a visualização com novo zoom
        if analysis_frame is not None:
            display_frame = draw_simple_frame(analysis_frame)
            cv2.imshow("VAR Analysis System", display_frame)
            cv2.waitKey(1)
            
        print(f"Zoom ajustado para {zoom_level:.1f}x. Use pan (arrastar) para posicionar a imagem.")

def save_calibration(filename="calibration.json"):
    """Salva as configurações de calibração em JSON."""
    global field_points, area_points, goal_line_points, homography_matrix, calibration_complete, perspective_correction
    
    data = {
        "field_points": field_points,
        "area_points": area_points,
        "goal_line_points": goal_line_points,
        "perspective_correction": perspective_correction,
        "attacking_direction": attacking_direction
    }
    # Converter p/ list
    data["field_points"] = [list(p) for p in field_points]
    data["area_points"] = [list(p) for p in area_points]
    data["goal_line_points"] = [list(p) for p in goal_line_points]
    
    if homography_matrix is not None:
        data["homography_matrix"] = homography_matrix.tolist()
    
    try:
        import json
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Calibração salva em {filename}")
        return True
    except Exception as e:
        print(f"Erro ao salvar calibração: {e}")
        return False

def load_calibration(filename="calibration.json"):
    """Carrega a calibração de um arquivo JSON."""
    global field_points, area_points, goal_line_points, homography_matrix
    global calibration_complete, perspective_correction, attacking_direction
    
    try:
        import json
        with open(filename, 'r') as f:
            data = json.load(f)
        field_points = [tuple(p) for p in data["field_points"]]
        area_points = [tuple(p) for p in data["area_points"]]
        goal_line_points = [tuple(p) for p in data["goal_line_points"]]
        perspective_correction = data["perspective_correction"]
        attacking_direction = data["attacking_direction"]
        
        if "homography_matrix" in data:
            homography_matrix = np.array(data["homography_matrix"])
            calibration_complete = True
        else:
            calculate_homography_from_field_lines()
        
        print(f"Calibração carregada de {filename}")
        return True
    except Exception as e:
        print(f"Erro ao carregar calibração: {e}")
        return False

def export_analysis_image(frame, filename=None):
    """Exporta o frame atual para arquivo de imagem."""
    if filename is None:
        import time
        filename = f"var_analysis_{int(time.time())}.jpg"
    try:
        cv2.imwrite(filename, frame)
        print(f"Análise exportada como imagem: {filename}")
        return True
    except Exception as e:
        print(f"Erro ao exportar imagem: {e}")
        return False

def export_analysis_video(cap, start_frame, end_frame, filename=None, duration_seconds=10):
    """Exporta um pequeno vídeo da análise."""
    global current_frame_num
    if filename is None:
        import time
        filename = f"var_analysis_{int(time.time())}.mp4"
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if start_frame is None:
        start_frame = max(0, current_frame_num - int(fps*duration_seconds/2))
    if end_frame is None:
        end_frame = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1, start_frame+int(fps*duration_seconds))
    
    # Definir duração das linhas (segundos) e calcular o número de frames
    line_duration_seconds = 3.0  # Mostrar linhas por 3 segundos
    line_duration_frames = int(fps * line_duration_seconds)
    line_start_frame = current_frame_num
    line_end_frame = line_start_frame + line_duration_frames
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    if not out.isOpened():
        print("Erro ao criar arquivo de vídeo.")
        return False
    
    try:
        current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        for fnum in range(start_frame, end_frame+1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detecção
            results = model(frame, classes=[0], conf=CONFIDENCE_THRESHOLD)
            detections = []
            if len(results)>0 and hasattr(results[0],'boxes'):
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    conf = boxes.conf[i].cpu().numpy()
                    cls = boxes.cls[i].cpu().numpy()
                    detections.append((box, conf, cls))
            
            display_frame = frame.copy()
            
            # Verificar se estamos no período para mostrar as linhas
            show_lines = fnum >= line_start_frame and fnum <= line_end_frame
            
            # Desenhar linhas offside se houver e se for o momento de mostrá-las
            if defender_point and show_lines:
                # Usar o mesmo cálculo da linha de impedimento melhorado
                off_line = calculate_offside_line(defender_point)
                p1_orig, p2_orig = off_line
                
                # Recortar para caber na tela
                clipped_off = clip_line_to_screen([p1_orig, p2_orig], width, height)
                cv2.line(display_frame, clipped_off[0], clipped_off[1], (255, 255, 255), 5)  # Borda branca
                cv2.line(display_frame, clipped_off[0], clipped_off[1], (0, 0, 255), 3)      # Linha vermelha
                
                # Adicionar texto "LINHA DO DEFENSOR"
                mid_x = (clipped_off[0][0] + clipped_off[1][0]) // 2
                mid_y = (clipped_off[0][1] + clipped_off[1][1]) // 2
                cv2.putText(display_frame, "LINHA DO DEFENSOR", (mid_x-100, mid_y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if attacker_point and show_lines:
                # Usar o mesmo cálculo da linha do atacante melhorado
                att_line = calculate_attacker_line(attacker_point)
                p1_orig, p2_orig = att_line
                
                # Recortar para caber na tela
                clipped_att = clip_line_to_screen([p1_orig, p2_orig], width, height)
                cv2.line(display_frame, clipped_att[0], clipped_att[1], (255, 255, 255), 5)  # Borda branca
                cv2.line(display_frame, clipped_att[0], clipped_att[1], (255, 0, 0), 3)      # Linha azul
                
                # Adicionar texto "LINHA DO ATACANTE"
                mid_x = (clipped_att[0][0] + clipped_att[1][0]) // 2
                mid_y = (clipped_att[0][1] + clipped_att[1][1]) // 2 + 30
                cv2.putText(display_frame, "LINHA DO ATACANTE", (mid_x-100, mid_y+10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
                # Checar impedimento
                is_offside = check_offside(defender_point, attacker_point, attacking_direction)
                status_text = "IMPEDIDO!" if is_offside else "POSICAO LEGAL"
                status_color = (0,0,255) if is_offside else (0,255,0)
                
                # Exibir status maior e mais visível
                text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                text_x = width // 2 - text_size[0] // 2
                cv2.rectangle(display_frame, (text_x-20, 30), (text_x+text_size[0]+20, 80), (0,0,0), -1)
                cv2.putText(display_frame, status_text, (text_x, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)
            
            # Desenhar jogadores com destaque apenas durante o período das linhas
            for det in detections:
                box = det[0]
                x1,y1,x2,y2 = [int(v) for v in box]
                foot_point = (int((x1+x2)/2), y2)
                
                # Determinar cor
                if show_lines and defender_point and abs(foot_point[0]-defender_point[0])<20 and abs(foot_point[1]-defender_point[1])<20:
                    color = DEFENDER_COLOR
                    cv2.putText(display_frame, "DEFENSOR", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                elif show_lines and attacker_point and abs(foot_point[0]-attacker_point[0])<20 and abs(foot_point[1]-attacker_point[1])<20:
                    color = ATTACKER_COLOR
                    cv2.putText(display_frame, "ATACANTE", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    # Destacar todos os jogadores com retângulos verdes durante o resto do vídeo
                    color = (0,255,0)
                
                cv2.rectangle(display_frame, (x1,y1), (x2,y2), color, 2)
                cv2.circle(display_frame, foot_point, 8, color, -1)
            
            # Adicionar contador de tempo para quando as linhas estão visíveis
            if show_lines:
                seconds_remaining = (line_end_frame - fnum) / fps
                if seconds_remaining > 0:
                    cv2.putText(display_frame, f"Análise: {seconds_remaining:.1f}s", (width-200, height-50),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            # Watermark
            timestamp = fnum/fps
            cv2.putText(display_frame, f"VAR Analysis - {timestamp:.2f}s", (20, height-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            out.write(display_frame)
            if (fnum - start_frame)%10==0:
                prog = (fnum - start_frame)/(end_frame - start_frame)*100
                print(f"Exportando video: {prog:.1f}%")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        out.release()
        print(f"Video exportado: {filename}")
        return True
    except Exception as e:
        print(f"Erro ao exportar video: {e}")
        if out.isOpened():
            out.release()
        return False

def create_control_panel():
    """Exemplo de painel de controle (opcional)."""
    pass  # Se quiser criar trackbars etc.

############################################
#                 MAIN LOOP                #
############################################

def main():
    global paused, current_frame_num, frame_offset, analysis_frame, analysis_detections
    global field_points, area_points, goal_line_points
    global defender_point, attacker_point
    global attacking_direction, analysis_mode, selection_step
    global zoom_level, pan_x, pan_y
    global playback_mode, playback_frames, playback_index, playback_speed
    global calibration_complete, homography_matrix
    global playback_speed_normal, show_instructions
    global slow_motion_mode, slow_motion_factor, perspective_correction
    global model, vertical_factor, auto_vanishing_point, use_auto_vanishing
    
    # Carregar modelo YOLO
    try:
        print("Carregando modelo YOLO...")
        model = YOLO("yolov8n.pt")
        model.to(device)
        print("Modelo YOLO carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return
    
    # Tentar abrir um vídeo
    video_paths = [
        "data/match_video.mp4",
        "data/videoplayback.mp4",
        # Adicione outros caminhos
    ]
    cap = None
    for path in video_paths:
        print(f"Tentando abrir video: {path}")
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            print(f"Video aberto: {path}")
            break
    
    if not cap or not cap.isOpened():
        print("Erro: Nenhum vídeo encontrado.")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height}, {fps} fps, {total_frames} frames")
    
    window_name = "VAR Analysis System"
    cv2.namedWindow(window_name)
    # Define o callback do mouse para a janela
    cv2.setMouseCallback(window_name, mouse_click_handler)
    
    # Configuração inicial do debug
    debug_mouse = True  # Ativa mensagens de debug para interações do mouse
    
    running = True
    last_processed_frame = -1
    frame_read_errors = 0
    
    print("\nInstruções:")
    print("S - Iniciar/parar análise")
    print("Espaço - Pausar/continuar vídeo")
    print("Q - Sair")
    print("Selecione pontos conforme instruções na tela.\n")
    
    # Mostrar status do modo de análise
    if debug_mouse:
        print(f"STATUS INICIAL: analysis_mode={analysis_mode}, selection_step={selection_step}")
    
    while running:
        try:
            if paused:
                frame_num = current_frame_num + frame_offset
            else:
                frame_num = current_frame_num
                if not analysis_mode:
                    current_frame_num += 1
            
            frame_num = max(0, min(frame_num, total_frames-1))
            
            if not paused and slow_motion_mode:
                wait_time = int(30/(playback_speed_normal*slow_motion_factor))
                if current_frame_num % int(1/slow_motion_factor)!=0:
                    frame_num = current_frame_num
                else:
                    current_frame_num += 1
            else:
                wait_time = 0 if paused else max(1,int(30/playback_speed_normal))
            
            if playback_mode:
                if 0<=playback_index<len(playback_frames):
                    pb_frame, pb_detections = playback_frames[playback_index]
                    display_frame = pb_frame.copy()
                    
                    # Desenhar lines
                    if defender_point:
                        off_line = calculate_offside_line(defender_point)
                        clipped_off = clip_line_to_screen(off_line, width, height)
                        cv2.line(display_frame, clipped_off[0], clipped_off[1], (0,0,255), 3)
                    if attacker_point:
                        att_line = calculate_attacker_line(attacker_point)
                        clipped_att = clip_line_to_screen(att_line, width, height)
                        cv2.line(display_frame, clipped_att[0], clipped_att[1], (255,0,0), 3)
                        
                        is_offside = check_offside(defender_point, attacker_point, attacking_direction)
                        stxt = "IMPEDIDO!" if is_offside else "POSICAO LEGAL"
                        scol = (0,0,255) if is_offside else (0,255,0)
                        cv2.putText(display_frame, stxt, (width//2-100,50), cv2.FONT_HERSHEY_SIMPLEX,1.0,scol,2)
                    
                    # Jogadores
                    for det in pb_detections:
                        box = det[0]
                        x1,y1,x2,y2 = [int(v) for v in box]
                        foot_point = (int((x1+x2)/2), y2)
                        color = (0,255,0)
                        cv2.rectangle(display_frame, (x1,y1), (x2,y2), color, 2)
                        cv2.circle(display_frame, foot_point, 8, color, -1)
                    
                    frame_info = f"Frame: {playback_index+1}/{len(playback_frames)} (Veloc: {playback_speed}x)"
                    cv2.putText(display_frame, frame_info, (20,height-20), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
                    cv2.putText(display_frame, "P: Parar reproducao | +/-: Veloc", (20,30), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
                    
                    cv2.imshow(window_name, display_frame)
                    playback_index = (playback_index + playback_speed) % len(playback_frames)
                    cv2.waitKey(30)
                else:
                    playback_index = 0
            elif last_processed_frame!=frame_num or analysis_mode:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret:
                    frame_read_errors +=1
                    print(f"Erro ao ler frame {frame_num} (Erro {frame_read_errors}/5)")
                    if frame_read_errors>=5:
                        print("Muitos erros de leitura. Reiniciando video...")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        current_frame_num=0
                        frame_offset=0
                        frame_read_errors=0
                        last_processed_frame=-1
                        continue
                    if not paused:
                        current_frame_num+=1
                    frame_offset=0
                    continue
                frame_read_errors=0
                last_processed_frame=frame_num
                
                results = model(frame, classes=[0], conf=CONFIDENCE_THRESHOLD)
                detections = []
                if len(results)>0 and hasattr(results[0],'boxes'):
                    boxes = results[0].boxes
                    for i in range(len(boxes)):
                        box = boxes.xyxy[i].cpu().numpy()
                        conf = boxes.conf[i].cpu().numpy()
                        cls = boxes.cls[i].cpu().numpy()
                        detections.append((box, conf, cls))
                
                if analysis_mode and analysis_frame is None:
                    analysis_frame = frame.copy()
                    analysis_detections = detections.copy()
                
                if analysis_mode and analysis_frame is not None:
                    display_frame = draw_simple_frame(analysis_frame)
                else:
                    display_frame = frame.copy()
                    for det in detections:
                        box = det[0]
                        x1,y1,x2,y2 = [int(v) for v in box]
                        cv2.rectangle(display_frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    if not analysis_mode:
                        cv2.putText(display_frame, "Pressione 'S' para analise", (20,height-20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255),2)
                
                frame_text = f"Frame: {frame_num+1}/{total_frames}"
                if paused:
                    frame_text+=" (PAUSADO)"
                else:
                    frame_text+=f" (Veloc: {playback_speed_normal:.1f}x)"
                cv2.putText(display_frame, frame_text, (width-400, height-20),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
                
                # Adicionar status de modo de análise e seleção para depuração
                if debug_mouse and analysis_mode:
                    debug_text = f"analysis_mode: {analysis_mode}, step: {selection_step}"
                    cv2.putText(display_frame, debug_text, (20, height-50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                
                cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(wait_time)&0xFF
            if key==ord(' '):
                paused=not paused
                print("Video pausado" if paused else "Video continuando")
            elif key in [ord('s'), ord('S')]:
                if not analysis_mode:
                    paused=True
                    analysis_mode=True
                    selection_step=1
                    field_points=[]
                    area_points=[]
                    goal_line_points=[]
                    defender_point=None
                    attacker_point=None
                    print("Modo de analise iniciado!")
                    if debug_mouse:
                        print(f"analysis_mode={analysis_mode}, selection_step={selection_step}")
                    
                    try:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                        ret, frame = cap.read()
                        if ret:
                            analysis_frame=frame.copy()
                            # YOLO
                            results=model(analysis_frame, classes=[0], conf=CONFIDENCE_THRESHOLD)
                            analysis_detections=[]
                            if len(results)>0 and hasattr(results[0],'boxes'):
                                boxes=results[0].boxes
                                for i in range(len(boxes)):
                                    box=boxes.xyxy[i].cpu().numpy()
                                    conf=boxes.conf[i].cpu().numpy()
                                    cls=boxes.cls[i].cpu().numpy()
                                    analysis_detections.append((box, conf, cls))
                            print(f"{len(analysis_detections)} detecções no frame de análise.")
                            print("SELECIONE PONTO 1 DA LINHA LATERAL (clique na imagem)")
                            display_frame=draw_simple_frame(analysis_frame)
                            cv2.imshow(window_name, display_frame)
                            cv2.waitKey(1)
                        else:
                            print("ERRO: nao foi possivel capturar frame para analise.")
                            analysis_mode=False
                    except Exception as e:
                        print(f"Erro ao iniciar analise: {e}")
                        analysis_mode=False
                else:
                    print("Saindo do modo de análise.")
                    analysis_mode=False
                    analysis_frame=None
            elif key==ord('n') and analysis_mode:
                # Adicionar tecla 'N' para pular a etapa atual de calibração
                if 3 <= selection_step <= 6:
                    if selection_step == 3 or selection_step == 4:
                        # Pular pontos da área
                        if selection_step == 3:
                            print("Pulando seleção da área.")
                            area_points = []
                        selection_step = 5
                        print("Agora selecione o ponto 1 da linha de gol.")
                    elif selection_step == 5 or selection_step == 6:
                        # Pular pontos da linha de gol
                        if selection_step == 5:
                            print("Pulando seleção da linha de gol.")
                            goal_line_points = []
                        selection_step = 7
                        print("Agora selecione o defensor.")
                    
                    if len(field_points) >= 2:
                        calculate_homography_from_field_lines()
                    
                    if analysis_frame is not None:
                        display_frame = draw_simple_frame(analysis_frame)
                        cv2.imshow(window_name, display_frame)
                        cv2.waitKey(1)
            elif key==ord('a'):
                attacking_direction='left'
                print("Ataque para a ESQUERDA.")
                if analysis_frame is not None:
                    df=draw_simple_frame(analysis_frame)
                    cv2.imshow(window_name, df)
            elif key==ord('d'):
                attacking_direction='right'
                print("Ataque para a DIREITA.")
                if analysis_frame is not None:
                    df=draw_simple_frame(analysis_frame)
                    cv2.imshow(window_name, df)
            elif key==ord('r'):
                if analysis_mode:
                    selection_step=1
                    field_points=[]
                    area_points=[]
                    goal_line_points=[]
                    defender_point=None
                    attacker_point=None
                    homography_matrix=None
                    calibration_complete=False
                    zoom_level=1.0
                    pan_x=0
                    pan_y=0
                    print("Analise resetada. Selecione linha lateral.")
                    if debug_mouse:
                        print(f"RESET: analysis_mode={analysis_mode}, selection_step={selection_step}")
                    if analysis_frame is not None:
                        df=draw_simple_frame(analysis_frame)
                        cv2.imshow(window_name, df)
            elif key==ord('q'):
                running=False
            elif key==ord('.') and paused:
                frame_offset+=1
            elif key==ord(',') and paused:
                frame_offset-=1
            elif key in [ord('h'), ord('H')]:
                show_instructions=not show_instructions
                if analysis_mode and analysis_frame is not None:
                    df=draw_simple_frame(analysis_frame)
                    cv2.imshow(window_name, df)
            elif key==ord('p'):
                if analysis_mode and selection_step>=9:
                    if not playback_mode:
                        playback_mode=True
                        playback_frames=[]
                        playback_index=0
                        current_pos=frame_num
                        startf=max(0, current_pos-30)
                        endf=min(total_frames-1, current_pos+30)
                        for f in range(startf, endf+1):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                            ret, frm=cap.read()
                            if ret:
                                rs=model(frm, classes=[0], conf=CONFIDENCE_THRESHOLD)
                                dets=[]
                                if len(rs)>0 and hasattr(rs[0],'boxes'):
                                    boxes=rs[0].boxes
                                    for i in range(len(boxes)):
                                        box=boxes.xyxy[i].cpu().numpy()
                                        conf=boxes.conf[i].cpu().numpy()
                                        cls=boxes.cls[i].cpu().numpy()
                                        dets.append((box, conf, cls))
                                playback_frames.append((frm.copy(), dets))
                        cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
                        print(f"{len(playback_frames)} frames capturados p/ reproducao.")
                    else:
                        playback_mode=False
                        print("Modo de reproducao encerrado.")
            elif key in [ord('+'), ord('=')]:
                if analysis_mode and (selection_step == 7 or selection_step == 8):
                    # Zoom in durante seleção de defensor/atacante - PRIORIDADE
                    old_zoom = zoom_level
                    zoom_level = min(zoom_max, zoom_level + zoom_step)
                    print(f"Zoom in: {zoom_level:.1f}x")
                    
                    # Atualizar a visualização
                    if analysis_frame is not None:
                        df = draw_simple_frame(analysis_frame)
                        cv2.imshow(window_name, df)
                        cv2.waitKey(1)
                    
                    print(f"NOTA: As coordenadas originais são preservadas com zoom={zoom_level:.1f}x")
                    if defender_point:
                        print(f"Defensor em: {defender_point}")
                    if attacker_point:
                        print(f"Atacante em: {attacker_point}")
                        
                elif playback_mode:
                    playback_speed=min(5, playback_speed+1)
                    print(f"Velocidade playback: {playback_speed}x")
                else:
                    playback_speed_normal=min(5.0, playback_speed_normal+0.5)
                    print(f"Velocidade: {playback_speed_normal:.1f}x")
            elif key in [ord('-'), ord('_')]:
                if analysis_mode and (selection_step == 7 or selection_step == 8):
                    # Zoom out durante seleção de defensor/atacante - PRIORIDADE
                    old_zoom = zoom_level
                    zoom_level = max(zoom_min, zoom_level - zoom_step)
                    print(f"Zoom out: {zoom_level:.1f}x")
                    
                    # Atualizar a visualização
                    if analysis_frame is not None:
                        df = draw_simple_frame(analysis_frame)
                        cv2.imshow(window_name, df)
                        cv2.waitKey(1)
                    
                    print(f"NOTA: As coordenadas originais são preservadas com zoom={zoom_level:.1f}x")
                    if defender_point:
                        print(f"Defensor em: {defender_point}")
                    if attacker_point:
                        print(f"Atacante em: {attacker_point}")
                elif playback_mode:
                    playback_speed=max(1, playback_speed-1)
                    print(f"Velocidade playback: {playback_speed}x")
                else:
                    playback_speed_normal=max(0.5, playback_speed_normal-0.5)
                    print(f"Velocidade: {playback_speed_normal:.1f}x")
            # Adicionar teclas alternativas para zoom: 'i' para zoom in e 'o' para zoom out
            elif key == ord('i') and analysis_mode and (selection_step == 7 or selection_step == 8):
                # 'i' para zoom in (alternativa para +)
                old_zoom = zoom_level
                zoom_level = min(zoom_max, zoom_level + zoom_step)
                print(f"Zoom in (i): {zoom_level:.1f}x")
                
                # Atualizar a visualização
                if analysis_frame is not None:
                    df = draw_simple_frame(analysis_frame)
                    cv2.imshow(window_name, df)
                    cv2.waitKey(1)
                
                print(f"NOTA: As coordenadas originais são preservadas com zoom={zoom_level:.1f}x")
                if defender_point:
                    print(f"Defensor em: {defender_point}")
                if attacker_point:
                    print(f"Atacante em: {attacker_point}")
            elif key == ord('o') and analysis_mode and (selection_step == 7 or selection_step == 8):
                # 'o' para zoom out (alternativa para -)
                old_zoom = zoom_level
                zoom_level = max(zoom_min, zoom_level - zoom_step)
                print(f"Zoom out (o): {zoom_level:.1f}x")
                
                # Atualizar a visualização
                if analysis_frame is not None:
                    df = draw_simple_frame(analysis_frame)
                    cv2.imshow(window_name, df)
                    cv2.waitKey(1)
                
                print(f"NOTA: As coordenadas originais são preservadas com zoom={zoom_level:.1f}x")
                if defender_point:
                    print(f"Defensor em: {defender_point}")
                if attacker_point:
                    print(f"Atacante em: {attacker_point}")
            elif key in [ord('['), ord(']')]:
                if key==ord('['):
                    perspective_correction=max(5, perspective_correction-5)
                else:
                    perspective_correction=min(100, perspective_correction+5)
                print(f"Correção de perspectiva: {perspective_correction}")
                if analysis_mode and analysis_frame is not None:
                    df=draw_simple_frame(analysis_frame)
                    cv2.imshow(window_name, df)
            elif key==ord('f'):
                use_auto_vanishing=not use_auto_vanishing
                print("Ponto de fuga AUTO" if use_auto_vanishing else "Ponto de fuga MANUAL")
                if analysis_mode and analysis_frame is not None and defender_point:
                    # Remover a atribuição desnecessária
                    df=draw_simple_frame(analysis_frame)
                    cv2.imshow(window_name, df)
            elif key==ord('c'):
                # Salvar calibração
                if analysis_mode and len(field_points) >= 2:
                    save_calibration()
            elif key==ord('l'):
                # Carregar calibração
                if analysis_mode:
                    if load_calibration():
                        selection_step = 7 if defender_point is None else (9 if attacker_point is not None else 8)
                        if analysis_frame is not None:
                            df=draw_simple_frame(analysis_frame)
                            cv2.imshow(window_name, df)
            elif key == ord('v'):
                # Tecla 'V' para exportar vídeo da análise
                if analysis_mode and defender_point is not None:
                    export_analysis_video(cap, None, None)
            elif key == ord('e'):
                # Tecla 'E' para exportar imagem da análise
                if analysis_mode and analysis_frame is not None:
                    frame_to_save = draw_simple_frame(analysis_frame)
                    export_analysis_image(frame_to_save)
            elif key == ord('d') and debug_mouse:
                # Tecla 'D' para alternar modo de debug
                debug_mouse = not debug_mouse
                print(f"Debug mode: {debug_mouse}")
            # Teclas de zoom
            elif (key==ord('+') or key==ord('i')) and analysis_mode:
                zoom_level = min(5.0, zoom_level + 0.2)
                print(f"Zoom: {zoom_level:.1f}x")
                df = draw_simple_frame(analysis_frame)
                cv2.imshow(window_name, df)
            elif (key==ord('-') or key==ord('o')) and analysis_mode:
                zoom_level = max(1.0, zoom_level - 0.2)
                print(f"Zoom: {zoom_level:.1f}x")
                df = draw_simple_frame(analysis_frame)
                cv2.imshow(window_name, df)
            elif key==ord('z') and analysis_mode:
                # Resetar zoom e pan
                zoom_level = 1.0
                pan_x = 0
                pan_y = 0
                print("Zoom e pan resetados")
                df = draw_simple_frame(analysis_frame)
                cv2.imshow(window_name, df)
            
        except Exception as e:
            print(f"Erro no loop principal: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
