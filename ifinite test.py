import numpy as np
from scipy.linalg import eigh  # 用于对称矩阵的广义特征值问题
N = 100
e = np.zeros((N))
for i in range(N):
 # 初始化 A 和 B 矩阵
 A = np.zeros((i+1, i+1))
 B = np.zeros((i+1, i+1))
 f = np.zeros((i+1))

 # 根据给定规则生成矩阵 A 和 B
 for m in range(i+1):
     for n in range(i+1):
         if (m + n) % 2 == 0:  # m+n 为偶数的情况
             A[m, n] = -8 * (1-m-n-2*m*n) /((m+n+3)*(m+n+1)*(m+n-1))
             B[m, n] = 2 / (m + n + 5) - 4 / (m + n + 3) + 2 / (m + n + 1)
         else:  # m+n 为奇数的情况
             A[m, n] = 0
             B[m, n] = 0
 eigvals, eigvecs = eigh(B)  # 如果 A 和 B 是对称的
 e[i] = eigvals[0]


print(e)

