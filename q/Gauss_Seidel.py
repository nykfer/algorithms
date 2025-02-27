import numpy as np
import diagonally_dominant_matrix as dm
import pandas as pd
def Gauss_Seidel_method(*, A:np.array, b:np.array, eps:float, trial:int)->dict:
    size = A.shape[0]
    diagonal_matrix = np.diag(np.diag(A)) # матриця з елементами матриці А по діагоналі
    matrix_without_diagonal= np.copy(A)
    np.fill_diagonal(matrix_without_diagonal, 0) # матриця з елементами матриці А, по діагоналі 0
    # перевірка існування єдиного розв'язку
    norma = dm.is_diagonally_dominant(A) #Знаходимо норму для перевірка на існування єдиного розв'язку
    is_symetric_matrix = np.array_equal(A, A.T)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    if((norma ==False) and (not is_symetric_matrix and not np.all(eigenvalues>0))): return f"Не існує єдиного розв'язку"
    
    X = np.copy(b) # np.zeros(size,dtype=float) # вектор, який зберігає xi 
    old_X = np.copy(b) # np.zeros(size, dtype=float)
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
            data = pd.DataFrame(results)
            print(data)
            return 
        old_X = np.copy(X)
            
    data = pd.DataFrame(results)
    print(data)
    
# A = np.array([[10,8,12,0,0,0,0],
#               [8,34,12,0,0,0,0],
#               [0,12,34,6,0,0,0],
#               [0,0,6,6,10,0,0],
#               [0,0,0,-2,12,-2,7],
#               [0,0,0,2,3,-3,11],
#               [0,0,0,14,0,11,4]])
# b = np.array([30,54,52,22,31,44,29])
A = np.array([[10,8,12,0,78,12,35],
              [8,12,34.1,0,0,0,0],
              [0,34,12,6,0,5,0],
              [0,5.5,6,6,10,0,1.1],
              [0,4.3,0,10,9,12,0],
              [0,0,2,12,122,0,11],
              [0,1,0,14,0,11,4]])
x = np.array([1,1,1,1,1,1,1])
A_T  =np.transpose(A)
matrix = np.dot(A_T,A)
b = np.dot(A,x)
vector = np.dot(A_T,b)

result = Gauss_Seidel_method(A=matrix, b=vector, eps=0.000001, trial=1000)
# print(result)