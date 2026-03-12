""" 重要性采样 """
import numpy as np
np.random.seed(42)

x = np.array([1, 2, 3])
pi = np.array([0.1, 0.1, 0.8])

# 真实期望
e = np.sum(x * pi)
print("E_pi[x]:",e)

# 蒙特卡洛方法
n = 100
samples = []
for _ in range(n):
    s = np.random.choice(x, p=pi)
    samples.append(s)
mean = np.mean(samples)
var = np.var(samples)
print("MC: {:.2f} (var: {:.2f})".format(mean, var))  # 输出：MC: 2.78 (var: 0.27)

b = np.array([1/3, 1/3, 1/3])
n = 100
samples = []
for _ in range(n):
    idx = np.arange(len(b)) # [0, 1, 2]
    i = np.random.choice(idx, p=b) # 使用 b 进行采样
    s = x[i]
    rho = pi[i] / b[i] # 𝜌
    samples.append(rho * s)
mean = np.mean(samples)
var = np.var(samples)
print("IS: {:.2f} (var: {:.2f})".format(mean, var)) # 输出：IS: 2.95 (var: 10.63)