import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi, copysign
from utils import Sigmoid as sig
from scipy.integrate import cumtrapz

# 理论的最大值
Jm = 375
Am = 125
Vm = 100

# 目标距离
D = 20
Sm = 4000

# 计算时间
Ts, Tj, Ta, Tv, jm = sig.calculate_Tall(D, Sm, Vm, Am, Jm)

# 计算实际的最大值
Jm = jm         # 让jm取代Jm
am = jm * (Ts + Tj)
vm = am * (2 * Ts + Tj + Ta)
dm = vm * (4 * Ts + 2 * Tj + Ta + Tv)

# 生成时间序列和初始jerk曲线
T_all = 8 * Ts + 4 * Tj + 2 * Ta + Tv
t = np.linspace(0, T_all, num=1000)
jerk = np.zeros(t.shape)

# 计算对应的时间段
tk = sig.calculate_t(Ts, Tj, Ta, Tv)

# 生成jerk曲线
a = sqrt(3) / 2  # sqrt(3)/2是一个分界点，sigmoid的模型参数
jerk = copysign(1,D)*sig.calculate_jerk(t, tk, jerk, Jm, a)

# 计算加速度曲线，jerk的积分
acceleration = cumtrapz(jerk, t, initial=0)

# 计算速度曲线，加速度的积分
velocity = cumtrapz(acceleration, t, initial=0)

# 计算位移曲线，速度的积分
displacement = cumtrapz(velocity, t, initial=0)

# 绘制曲线
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.grid()
plt.plot(t, displacement, color='red', linewidth=2)
plt.title('Displacement Profile')
plt.xlabel('Time (s)')
plt.ylabel('Displacement')

plt.subplot(4, 1, 2)
plt.grid()
plt.plot(t, velocity, color='red', linewidth=2)
plt.title('Velocity Profile')
plt.ylabel('Velocity')

plt.subplot(4, 1, 3)
plt.grid()
plt.plot(t, acceleration, color='red', linewidth=2)
plt.title('Acceleration Profile')
plt.ylabel('Acceleration')

plt.subplot(4, 1, 4)
plt.grid()
plt.plot(t, jerk, color='red', linewidth=2)
plt.title('Jerk Profile')
plt.ylabel('Jerk')

plt.tight_layout()
plt.show()

# 打印几个关键值
print('Ts:', Ts)
print('Tj:', Tj)
print('Ta:', Ta)
print('Tv:', Tv)
print('dm:', dm)
print('vm:', vm)
print('am:', am)
print('Tall:', T_all)
