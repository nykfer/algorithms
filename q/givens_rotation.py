import numpy as np
import pandas as pd
# функція для обчислення елементів матриці Гівенса
def givens_matrix_elements(*, a:float, b:float)->list[float]:
    r = np.sqrt(a**2+b**2)
    c = a/r
    s = -b/r
    return c, s

def solve_slae_using_qr(*, A:np.array, b:np.array)->np.array:
    n = A.shape[0]
    # ініцілізуємо спочатку R як матрицю А, а Q як одиничну матрицю
    R = A.copy()
    Q = np.identity(n)
    
    for i in range(n-1):
        for j in range(i+1, n):
            c, s = givens_matrix_elements(a=R[i,i],b= R[j,i])
# під час множення матриця Гівенса на матрицю R змінюються тільки рядки на, яких знаходяться елементи у матриці Гівенса
# інші рядки у матриці АR залишаються без змін
            R[i], R[j] = (R[i]*c + R[j]*(-s)), (R[i]*s + R[j]*c)
# створюємо матриці оберенену до матриці Q, яку використаємо під час кінцевого розв'язку
            Q[i], Q[j] = (Q[i]*c + Q[j]*(-s)), (Q[i]*s + Q[j]*c)
# знаходимо допоміжний вектор Y та розв'язуємо систему          
    
    Y = np.dot(Q,b)
    X = np.zeros(n)
    for i in range(n-1, -1,-1):
        total = 0
        for j in range(n-1,i, -1):
            total += R[i,j]*X[j]
        X[i] = (Y[i]-total)/R[i,i] 
    return X
A = np.array([[10,8,12,0,78,12,35],
              [8,12,34.1,0,0,0,0],
              [0,34,12,6,0,5,0],
              [0,5.5,6,6,10,0,1.1],
              [0,4.3,0,10,9,12,0],
              [0,0,2,12,122,0,11],
              [0,1,0,14,0,11,4]])
b = np.array([30, 54, 52, 22, 31, 44, 29])
b1 = np.array([30.1, 54.01, 52.01, 22, 31, 44, 29])

result = solve_slae_using_qr(A=A, b=b1)
 

columns = {"x1":[result[0]],
           "x2":[result[1]],
           "x3":[result[2]],
           "x4":[result[3]],
           "x5":[result[4]],
           "x6": [result[5]],
           "x7":[result[6]]}
data = pd.DataFrame(columns, dtype=float)
print(data)