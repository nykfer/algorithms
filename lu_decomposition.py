import numpy as np
import pandas as pd
# функція для перевірки можливості LU розкладу
# перевірка, щоб всі головні мінори матриці були не нульовими
def check_principal_minor(*, A:np.array, size:int):
    for i in range(1, size+1):
        minor = A[:i, :i] # матриця з елементів перетину перших і рядків і перших і стовпців
        determinant = np.linalg.det(minor)
        if determinant == 0: return False
        
    return True

def decomposition(*,A:np.array, size:int)->list:
    if check_principal_minor(A=A, size=size) == False: return 0, 0, False
        
    U = np.copy(A).astype(np.float64) # копіюємо матрицю А
    L = np.identity(size, dtype=np.float64) # створюємо одиничну матрицю
   
    for i in range(size-1):
        for j in range(i+1, size):
            factor = U[j,i]/U[i,i] # обчислення множника на який домножаємо рядок
            U[j,:] -= factor*U[i,:] # оновлюємо рядки роблячи нулі під діагоналлю
            L[j,i] =  factor 
    return U, L, True
            
            
def solving_SLAE(*, A:np.array, b:np.array)->np.array:
    size = A.shape[0]
    U, L, flag = decomposition(A=A, size=size)
    if flag == False: return None
    # створюємо вектори невідомих
    Y = np.ones(size)
    X = np.ones(size)
    
    for i in range(size):
        total = 0
        # знаходимо суму всіх відомих значень в лівій частині рівняння
        for j in range(i):
            total += L[i,j]*Y[j] 
        # знаходимо значення yі як різницю між відповідним елементом вектора b та сумою, що знайшли на минулому кроці, та поділивши це на відповідний множник     
        Y[i] = (b[i]- total)/L[i,i]
    # робимо тіж операції тільки в зворотньому кроці
    for i in range(size-1, -1, -1):
        total = 0
        for j in range (size-1,i,-1):
            total += U[i,j]*X[j]
        X[i] = (Y[i]-total)/U[i,i]  
    
    
    
    return X   

# A = np.array([[10,8,12,0,0,0,0],
#               [8,12,34,0,0,0,0],
#               [0,34,12,6,0,0,0],
#               [0,0,6,6,10,0,0],
#               [0,0,0,10,9,12,0],
#               [0,0,0,12,12,9,11],
#               [0,0,0,14,0,11,4]])
# b = np.array([30, 54, 52, 22, 31, 44, 29])
# b1 = np.array([30.1, 54.01, 52.01, 22, 31, 44, 29])
# A = np.array([[400,-201],[-800,401]])
# b = np.array([2,4])
A = np.identity(5, dtype=float)
num = [0.75, 1.25, 1.75, 2.25, 2.75]
for i in range(len(num)):
    A[:,i] = [x**i for x in num]
print(f"Визначник А = {np.linalg.det(A)}")  
b = [-0.161, -0.019, 0.531, 1.552, 3.058]
result = solving_SLAE(A=A, b=b)
print(result)
