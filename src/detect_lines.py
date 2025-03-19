import cv2
import numpy as np

# Cache para armazenar linhas de campo anteriores e reduzir flutuações
_previous_lines = None
_stability_counter = 0

def detect_field_lines(frame):
    """
    Detecta linhas de campo em um frame de vídeo de futebol com estabilidade entre frames.
    
    Args:
        frame: O frame do vídeo.
        
    Returns:
        Lista de linhas detectadas no formato [(x1, y1, x2, y2), ...].
    """
    global _previous_lines, _stability_counter
    
    # Converter para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar blur para reduzir ruído
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Melhorar o contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    # Detectar bordas com parâmetros mais restritivos
    edges = cv2.Canny(enhanced, 70, 200)
    
    # Aplicar operações morfológicas para limpar ruídos
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    # Detectar linhas usando transformada de Hough com parâmetros mais restritivos
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=80, maxLineGap=15)
    
    # Processar as linhas detectadas
    field_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calcular o ângulo e comprimento da linha
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Filtrar linhas (manter apenas as quase horizontais ou verticais de comprimento significativo)
            if length > 80 and (abs(angle) < 15 or abs(angle) > 165 or  # quase horizontais
                abs(angle - 90) < 15 or abs(angle + 90) < 15):  # quase verticais
                field_lines.append((x1, y1, x2, y2))
    
    # Se não houver linhas detectadas, usar as anteriores se disponíveis
    if not field_lines and _previous_lines is not None:
        _stability_counter += 1
        if _stability_counter < 10:  # Usar as linhas anteriores por até 10 frames
            return _previous_lines
        else:
            _stability_counter = 0
            return []
    
    # Limitar o número de linhas para evitar processamento excessivo
    if len(field_lines) > 8:
        # Ordenar por comprimento (do maior para o menor)
        field_lines.sort(key=lambda line: np.sqrt((line[2] - line[0])**2 + (line[3] - line[1])**2), reverse=True)
        field_lines = field_lines[:8]
    
    # Estabilização: se tivermos linhas anteriores, suavizar as mudanças
    if _previous_lines is not None and field_lines:
        _stability_counter = 0
        stabilized_lines = []
        
        # Para cada linha atual, encontrar a correspondente mais próxima da anterior e fazer a média
        for line in field_lines:
            x1, y1, x2, y2 = line
            
            # Se houver uma linha anterior próxima, fazer a média para estabilizar
            found_match = False
            for prev_line in _previous_lines:
                px1, py1, px2, py2 = prev_line
                
                # Distância entre os pontos das linhas
                dist = (abs(x1 - px1) + abs(y1 - py1) + abs(x2 - px2) + abs(y2 - py2)) / 4
                
                if dist < 30:  # Limiar para considerar a mesma linha
                    # Média ponderada (70% atual, 30% anterior)
                    nx1 = int(0.7 * x1 + 0.3 * px1)
                    ny1 = int(0.7 * y1 + 0.3 * py1)
                    nx2 = int(0.7 * x2 + 0.3 * px2)
                    ny2 = int(0.7 * y2 + 0.3 * py2)
                    
                    stabilized_lines.append((nx1, ny1, nx2, ny2))
                    found_match = True
                    break
            
            if not found_match:
                stabilized_lines.append(line)
        
        # Atualizar linhas anteriores
        _previous_lines = stabilized_lines
        return stabilized_lines
    
    # Atualizar as linhas anteriores para o próximo frame
    _previous_lines = field_lines
    return field_lines 
