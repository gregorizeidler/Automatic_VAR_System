import cv2
import numpy as np

# Cache para armazenar cálculos frequentes
_last_field_lines = None
_last_line_points = None
_last_y_position = None  # Para estabilizar a posição vertical da linha
_stability_counter = 0   # Contador para estabilidade

def draw_var_line(frame, tracked_players, field_lines):
    """
    Desenha a linha VAR no frame baseado nos jogadores rastreados e linhas do campo.
    
    Args:
        frame: O frame do vídeo.
        tracked_players: Objeto TrackPlayers contendo os jogadores rastreados.
        field_lines: Lista de linhas do campo detectadas.
    
    Returns:
        O frame com a linha VAR desenhada.
    """
    global _last_field_lines, _last_line_points, _last_y_position, _stability_counter
    
    # Se não houver jogadores ou linhas, retorne o frame original
    if tracked_players is None or field_lines is None or len(field_lines) < 2:
        return frame
    
    # Copiar o frame para não modificar o original
    result_frame = frame.copy()
    height, width = frame.shape[:2]
    
    try:
        # Obter as posições dos jogadores
        player_positions = tracked_players.get_track_positions()
        if not player_positions or len(player_positions) < 1:
            # Se não houver jogadores, mas temos uma linha anterior, use-a
            if _last_line_points is not None:
                x1, y1, x2, y2 = _last_line_points
                # Desenhar uma linha mais visível
                draw_enhanced_var_line(result_frame, x1, y1, x2, y2)
                return result_frame
            return result_frame
        
        # Calcular posição y para a linha VAR com base na posição do jogador mais avançado
        advanced_player_y = height
        for pos in player_positions:
            if pos[1] < advanced_player_y:  # Menor y é mais alto na imagem
                advanced_player_y = pos[1]
        
        # Se temos uma posição y anterior, estabilizar a linha
        if _last_y_position is not None:
            # Suavizar as mudanças na posição y
            if abs(_last_y_position - advanced_player_y) > 50:
                # Mudança grande, verificar estabilidade
                if _stability_counter < 5:
                    # Usar posição anterior para evitar saltos
                    advanced_player_y = _last_y_position
                    _stability_counter += 1
                else:
                    # Aceitar nova posição após vários frames consistentes
                    _stability_counter = 0
            else:
                # Mudança pequena, média ponderada
                advanced_player_y = int(0.7 * advanced_player_y + 0.3 * _last_y_position)
                _stability_counter = 0
        
        # Atualizar posição y para o próximo frame
        _last_y_position = advanced_player_y
        
        # Criar pontos para a linha VAR horizontal no nível do jogador mais avançado
        _last_line_points = (0, advanced_player_y, width, advanced_player_y)
        
        # Desenhar a linha VAR aprimorada
        x1, y1, x2, y2 = _last_line_points
        draw_enhanced_var_line(result_frame, x1, y1, x2, y2)
        
        # Destacar os jogadores com cores baseadas na posição em relação à linha VAR
        for player_pos in player_positions:
            x, y = int(player_pos[0]), int(player_pos[1])
            
            # Cor baseada na posição (verde = onside, vermelho = possível offside)
            color = (0, 255, 0) if y >= advanced_player_y else (0, 0, 255)
            
            # Desenhar círculo e identificador
            cv2.circle(result_frame, (x, y), 8, color, -1)
            cv2.circle(result_frame, (x, y), 8, (255, 255, 255), 2)  # Borda branca para melhor visibilidade
        
        return result_frame
    
    except Exception as e:
        print(f"Erro ao desenhar linha VAR: {e}")
        return frame  # Retornar o frame original em caso de erro

def draw_enhanced_var_line(frame, x1, y1, x2, y2):
    """Desenha uma linha VAR melhorada com efeitos visuais para aumentar a visibilidade."""
    height, width = frame.shape[:2]
    
    # Desenhar linha principal (vermelha)
    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    
    # Adicionar efeito de brilho/sombra
    cv2.line(frame, (x1, y1-2), (x2, y2-2), (255, 255, 255), 1)  # Linha branca acima
    cv2.line(frame, (x1, y1+2), (x2, y2+2), (0, 0, 100), 1)      # Linha azul escuro abaixo
    
    # Adicionar marcadores a cada 100 pixels
    for x in range(0, width, 100):
        cv2.line(frame, (x, y1-10), (x, y1+10), (0, 0, 255), 2)
    
    # Adicionar texto "VAR Line"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "VAR Line", (width//2-50, y1-15), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
