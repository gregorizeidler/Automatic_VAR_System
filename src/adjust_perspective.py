import cv2
import numpy as np

# Cache para armazenar a matriz de transformação
_last_field_lines = None
_last_transform_matrix = None

def adjust_perspective(frame, field_lines):
    """
    Ajusta a perspectiva do frame com base nas linhas do campo detectadas.
    Implementa caching para evitar recálculos desnecessários.
    
    Args:
        frame: O frame do vídeo.
        field_lines: Lista de linhas do campo detectadas no formato [(x1, y1, x2, y2), ...].
        
    Returns:
        O frame com a perspectiva corrigida.
    """
    global _last_field_lines, _last_transform_matrix
    
    # Verificar se temos linhas suficientes
    if field_lines is None or len(field_lines) < 4:
        print("Não há linhas suficientes para ajuste de perspectiva")
        return frame
        
    try:
        # Verificar se as linhas são iguais às anteriores para usar o cache
        if (_last_field_lines is not None and 
            len(_last_field_lines) == len(field_lines) and 
            np.array_equal(np.array(field_lines), np.array(_last_field_lines))):
            # Usar a matriz de transformação em cache
            if _last_transform_matrix is not None:
                # Aplicar a transformação
                height, width = frame.shape[:2]
                return cv2.warpPerspective(frame, _last_transform_matrix, (width, height))
        
        # Selecionar os 4 pontos da origem (usar apenas as 4 primeiras linhas)
        src_pts = []
        for i in range(min(4, len(field_lines))):
            x1, y1, x2, y2 = field_lines[i]
            src_pts.append([x1, y1])
            src_pts.append([x2, y2])
        
        # Usar apenas os 4 primeiros pontos
        src_pts = np.array(src_pts[:4], dtype=np.float32)
        
        # Definir os pontos de destino (retângulo)
        height, width = frame.shape[:2]
        dst_pts = np.array([
            [0, 0],
            [width, 0],
            [0, height],
            [width, height]
        ], dtype=np.float32)
        
        # Calcular a matriz de transformação
        _last_transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        _last_field_lines = field_lines
        
        # Aplicar a transformação
        corrected_frame = cv2.warpPerspective(frame, _last_transform_matrix, (width, height))
        
        return corrected_frame
        
    except Exception as e:
        print(f"Erro no ajuste de perspectiva: {e}")
        return frame  # Retornar o frame original em caso de erro
