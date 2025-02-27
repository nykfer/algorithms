import numpy as np

def is_diagonally_dominant(matrix)->bool:
    n = matrix.shape[0]
    for i in range(n):
        # Сума абсолютних значень всіх елементів рядка крім діагонального
        row_sum = sum(abs(matrix[i][j]) for j in range(n) if i != j)
        # Якщо діагональний елемент не перевищує цю суму, матриця не домінантна
        if abs(matrix[i][i]) <= row_sum:
            return False
    return True


