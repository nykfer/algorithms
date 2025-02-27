import numpy as np
import pandas as pd
import diagonally_dominant_matrix as dm

def SOR_method(*, A:np.array, b:np.array, w:float, eps:float, trial:int):
    size = A.shape[0]
    diagonal_matrix = np.diag(np.diag(A)) # матриця з елементами матриці А по діагоналі
    matrix_without_diagonal= np.copy(A)
    np.fill_diagonal(matrix_without_diagonal, 0) # матриця з елементами матриці А, по діагоналі 0
    # перевірка існування єдиного розв'язку
    norma = dm.is_diagonally_dominant(A) #Знаходимо норму для перевірка на існування єдиного розв'язку
    is_symetric_matrix = np.array_equal(A, A.T)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    if((norma ==False) or (not is_symetric_matrix and not np.all(eigenvalues>0))): return f"Не існує єдиного розв'язку"
    
    X = np.array([283.208,-109.95, 275.515, -202.18, -187.17,-116.31], dtype=float) # np.zeros(size,dtype=float) # вектор, який зберігає xi 
    old_X = np.array([283.208,-109.95, 275.515, -202.18, -187.17,-116.31], dtype=float) # np.zeros(size, dtype=float)
    mistakes = np.ones(size) * np.inf # вектор для збереження модулю різниці між значеннями на k та k+1 кроках
    steps = 0
    results = {
        "Step": [steps],
        "Mistake": [0],
        "X1": [X[0]],
        "X2": [X[1]],
        "X3": [X[2]],
        "X4": [X[3]],
        "X5": [X[4]],
        "X6": [X[5]]
    } 
    for x in range(trial):
        
        for  i in range(size):
            total = np.dot(matrix_without_diagonal[i,:], X) # обчислення скалярного добутку 
            X[i] = (b[i]-total)/diagonal_matrix[i,i] # обчислюємо нове значення xi
            X[i] = (1-w)*old_X[i] + w*X[i]
        steps+=1
        mistakes = abs(X -old_X) # заміна значень вектора на нові, після оновлення всіх xi
        results["Mistake"].append(max(mistakes))
        results["Step"].append(steps)
        results["X2"].append(X[1])
        results["X3"].append(X[2])
        results["X4"].append(X[3])
        results["X5"].append(X[4])
        results["X6"].append(X[5])
        results["X1"].append(X[0])
        if(max(mistakes)<eps): # якщо найбільше значення з модулю різниць менше за eps, програма завершує роботу
            return results
        old_X = np.copy(X)
            
    return results

# A = np.array([[67.95, 9.39, -1.4, 0, 1.5, 1.3],
#                [8.39, 101.45, 1.7, 4, 3.9, -2.3],
#                [7.39, -2.1, 105.45, 4, 4.7, 2.9],
#                [1.39, 1.6, 1.8, 49.95, 3.9, -1.3],
#                [8.39, -1.7, 3.9, 2.7, 92.45, 1.8],
#                [1.39, 2.3, 4.1, -2.1, 1.4, 56.45]])
# b = np.array([283.208,-109.95, 275.515, -202.18, -187.17,-116.31])
# res = SOR_method(A=A, b=b, w=1.1, eps=0.0001, trial=30)
# table = pd.DataFrame(res)
# table.to_excel("SOR_method.xlsx", index=False)
# read_table = pd.read_excel("SOR_method.xlsx")
# print(read_table)