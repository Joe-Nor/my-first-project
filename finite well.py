import numpy as np
from scipy.linalg import eigh  # 用于对称矩阵的广义特征值问题
# from scipy.linalg import eig  # 如果矩阵 A 和 B 不对称，可以用这个
import matplotlib.pyplot as plt

# 定义矩阵的大小
N = 6 #设定矩阵大小为 N x N
j = 1  # 定义第几个能级
L = 5
# 初始化 A 和 B 矩阵
A = np.zeros((N, N))
B = np.zeros((N, N))
f = np.zeros((N))
k = np.zeros((N))

for i in range(N):
   k[i] =  i * np.pi / L 
# 根据给定规则生成矩阵 A 和 B
print("矩阵 k:")
print(k)
for m in range(N):
    for n in range(N):
          if m == n: 
            if n != 0 :
               A[m, n] =  k[m]**2 - 1/L*(1+np.sin(2*k[n])/(2*k[n]))
               B[m, n] = 1
            else:
               A[m, n] =  k[m]**2 - 1/L
               B[m, n] = 1
          else:  
            A[m, n] = - 1/L * (np.sin(k[m]-k[n])/(k[m]-k[n])+np.sin(k[m]+k[n])/(k[m]+k[n]))
            B[m, n] = 0
# 输出生成的矩阵 A 和 B
print("矩阵 A:")
print(A)

print("矩阵 B:")
print(B)

# 使用 eigh 求解广义特征值问题 A v = λ B v
eigvals, eigvecs = eigh(A, B)  # 如果 A 和 B 是对称的
#eigvals, eigvecs = eigh(B)
# 如果 A 和 B 不对称，使用 eig(A, B)

# 输出特征值和特征向量
print("特征值：")
print(eigvals)
print("特征向量：")
print(eigvecs)

v_normalized = eigvecs / np.linalg.norm(eigvecs)

# 求解的波函数与严格解之间的可视化
def f(x):
     result = 0 #初始化函数值为0
     for i in range(N):
         
         result += eigvecs[i,j-1] * (1/np.sqrt(L)) * np.cos(k[i]*x)
        
     return result



# 生成 x 的值范围
x = np.linspace(-L, L, 400)


y = f(x)

     


# 创建图形
plt.plot(x, y, label='calculation')


# 添加标题和标签
plt.title("Plot of calculation")
plt.xlabel("x")
plt.ylabel("y")

# 显示图例
plt.legend()

# 显示图形
plt.show()
