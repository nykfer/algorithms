import numpy as np
import pandas as pd
import diagonally_dominant_matrix as dm
pd.set_option('display.float_format', '{:.8f}'.format)

def Jacobi_Method_for_SLAE(*, A:np.array, b:np.array, eps:float, trial: int)-> dict:
    size = A.shape[0]
    diagonal_matrix = np.diag(np.diag(A)) # матриця з елементами матриці А по діагоналі
    matrix_without_diagonal= np.copy(A)
    np.fill_diagonal(matrix_without_diagonal, 0) # матриця з елементами матриці А, по діагоналі 0
    norma = dm.is_diagonally_dominant(A) #Знаходимо норму для перевірка на існування єдиного розв'язку
    if(norma ==False): return f"Норма матриці більша за 1, тому метод не спрацює" #перевірка, якщо норма менше одиниці, код виконується далі
    
    old_x = np.array([283.208,-109.95, 275.515, -202.18, -187.17,-116.31], dtype=float) # np.zeros(size) # вектор, який зберігає x на k-тому кроці
    new_x = np.array([283.208,-109.95, 275.515, -202.18, -187.17,-116.31], dtype=float) # np.zeros(size) # вектор, який зберігає x на k+1-тому кроці
    mistakes = np.ones(size) * np.inf # вектор для збереження модулю різниці між значеннями на k та k+1 кроках
    steps = 0
    results = {
        "Step": [steps],
        "Mistake": [0],
        "X1": [new_x[0]],
        "X2": [new_x[1]],
        "X3": [new_x[2]],
        "X4": [new_x[3]],
        "X5": [new_x[4]],
        "X6": [new_x[5]]
    }
   
    for x in range(trial):
        for  i in range(size):
            total = np.dot(matrix_without_diagonal[i,:], old_x) # обчислення скалярного добутку 
            new_x[i] = (b[i]-total)/diagonal_matrix[i,i] # обчислюємо нове значення xi
        steps+=1
        mistakes = abs(new_x -old_x) # заміна значень ветора на нові, після оновлення всіх xi
        results["Mistake"].append(max(mistakes))
        results["Step"].append(steps)
        results["X2"].append(new_x[1])
        results["X3"].append(new_x[2])
        results["X4"].append(new_x[3])
        results["X5"].append(new_x[4])
        results["X6"].append(new_x[5])
        results["X1"].append(new_x[0])
        if(max(mistakes)<eps): # якщо найбільше значення з модулю різниць менше за eps, програма завершує роботу
            return results
        old_x = np.copy(new_x) # ще не дішли до потрібного результату, тому оновлюємо значення x на k-тому кроці
            
    
    return results
    


            
