import numpy as np

def refine_field_lines(field_lines):
    """
    Filtra e ajusta as linhas do campo para melhorar a precisão.
    """
    field_lines = sorted(field_lines, key=lambda line: line[1])  # Ordena pelas coordenadas Y
    return field_lines[:4]  # Mantém apenas as 4 primeiras linhas principais
