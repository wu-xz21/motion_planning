import numpy as np
import matplotlib.pyplot as plt
from math import ceil, sin, sqrt, pi
from scipy.integrate import cumtrapz

# 定义速度、加速度和jerk的限制
Vm = 10  # 单位为deg/s
Am = 10
Jm = 20
Sr = 20  # 设定的关节角之间的变化
n = 2 * pi * (Am * Jm) / (Vm * Jm - Am ** 2)

if n < pi * Jm / Am:  # 如果2As>Am，那么tc<0,此时设置tc = 0，即 n = pi*(Jm/Am)
    n = pi * Jm / Am
if n < pi * (Am * Jm) / (Vm * Jm - Am ** 2):  # 如果2Vj>Vm，那此时ta<0,令
    n = pi * (Am * Jm) / (Vm * Jm - Am ** 2)

As = pi * Jm / (2 * n)
Sup = Vm * Am / (2 * Jm) + Vm ** 2 / (2 * Am) + Vm * pi / (2 * n)  # 速度从0到Vm的距离

ts = pi / n  # jerk从0变换到Jm的时间
tc = Am / Jm - pi / n  # jerk保持Jm的时间
tj = 2 * ts + tc  # 加速度从0到Am的时间
Vj = tj * Am / 2  # tj时间时的速度
ta = (Vm - 2 * Vj) / Am  # 加速度保持Am的时间
tv = (Sr - 2 * Sup) / Vm  # 速度保持Vm的时间
t_all = 8 * ts + 4 * tc + 2 * ta + tv  # 总时间

# 不能达到最大速度Vm且能达到加速度Am的距离Sa
Sa = ((2 * n * Jm * 2 * Vj + n * Am ** 2 + pi * Am * Jm) ** 2 - (n * Am ** 2 + pi * Am * Jm) ** 2) / (
            4 * Am * (n * Jm) ** 2)
if 2 * Sup >= Sr > Sa:  # 不能达到最大速度Vm且能达到加速度Am
    tv = 0
    Vj = tj * Am / 2  # tj时间时的速度
    V_temp = (sqrt((n * Am ** 2 + pi * Am * Jm) ** 2 + 4 * Am * Sr * (n * Jm) ** 2) - n * Am ** 2 - pi * Am * Jm) / (
            2 * n * Jm)
    ta = (V_temp - 2 * Vj) / Am  # 加速度保持Am的时间
    t_all = 8 * ts + 4 * tc + 2 * ta


Sj = Am ** 3 / (6 * Jm ** 2) + pi * Am ** 2 / (4 * n * Jm) + Am * (pi ** 2 - 4) / (4 * n ** 2)

# 无法达到最大加速度Am但能达到最大jerkJm，tc>0
if Sr <= Sa:
    ta = 0
    tv = 0
    Vj = 0
    a = 2 * n ** 2
    b = 4 * n * pi * Jm
    c = 2 * (n * Jm) ** 2
    d = -(n * Jm) ** 2 * Sr
    A_temp = np.roots([a, b, c, d])
    # 筛选出正解
    A_temp = [root.real for root in A_temp if root.real > 0]
    A_temp = A_temp[0]
    A_0 = Jm*pi/n       # 使得tc = 0时的最大加速度
    Sj = (a*A_0**3 + b*A_0**2 + c*A_0)/((n*Jm)**2)
    if Sr > Sj:
        tc = A_temp / Jm - pi / n
        t_all = 8 * ts + 4 * tc
    if Sr <= Sj:
        tc = 0

# 定义时间和各个时间变量

t = np.linspace(0, t_all, 400)  # 创建时间序列
jerk = np.zeros(t.shape)

tk = np.zeros(16)
tk[0] = 0
tk[1] = tk[0] + ts
tk[2] = tk[1] + tc
tk[3] = tk[2] + ts
tk[4] = tk[3] + ta
tk[5] = tk[4] + ts
tk[6] = tk[5] + tc
tk[7] = tk[6] + ts
tk[8] = tk[7] + tv
tk[9] = tk[8] + ts
tk[10] = tk[9] + tc
tk[11] = tk[10] + ts
tk[12] = tk[11] + ta
tk[13] = tk[12] + ts
tk[14] = tk[13] + tc
tk[15] = tk[14] + ts

# 建立Jerk和时间的关系
# 1,13
k = (t >= tk[0]) & (t < tk[1])
jerk[k] = Jm / 2 * (1 - np.cos(n * t[k]))
k = (t >= tk[12]) & (t < tk[13])
jerk[k] = Jm / 2 * (1 - np.cos(n * t[k]))

# 2,14
k = (t >= tk[1]) & (t < tk[2]) | (t >= tk[13]) & (t < tk[14])
jerk[k] = Jm

# 3,15
k = (t >= tk[2]) & (t < tk[3])
jerk[k] = Jm / 2 * (1 + np.cos(n * t[k]))
k = (t >= tk[14]) & (t <= tk[15])
jerk[k] = Jm / 2 * (1 + np.cos(n * t[k]))
# 4,8,12
k = (t >= tk[3]) & (t < tk[4]) | (t >= tk[7]) & (t < tk[8]) | (t >= tk[11]) & (t < tk[12])
jerk[k] = 0

# 5,9
k = (t >= tk[4]) & (t < tk[5])
jerk[k] = -Jm / 2 * (1 - np.cos(n * t[k]))
k = (t >= tk[8]) & (t < tk[9])
jerk[k] = -Jm / 2 * (1 - np.cos(n * t[k]))

# 6,10
k = (t >= tk[5]) & (t < tk[6]) | (t >= tk[9]) & (t < tk[10])
jerk[k] = -Jm
# 7,11
k = (t >= tk[6]) & (t < tk[7])
jerk[k] = -Jm / 2 * (1 + np.cos(n * t[k]))
k = (t >= tk[10]) & (t < tk[11])
jerk[k] = -Jm / 2 * (1 + np.cos(n * t[k]))

# 计算加速度曲线，jerk的积分
acceleration = cumtrapz(jerk, t, initial=0)

# 计算速度曲线，加速度的积分
velocity = cumtrapz(acceleration, t, initial=0)

# 计算位移曲线，速度的积分
displacement = cumtrapz(velocity, t, initial=0)

fig, ax = plt.subplots()
ax.plot(t, displacement, label='displacement', color='blue', linestyle='-')
ax.plot(t, velocity, label='velocity', color='brown', linestyle='--')
ax.plot(t, acceleration, label='acceleration', color='orange', linestyle='-.')
ax.plot(t, jerk, label='jerk', color='purple', linestyle=':')
ax.set_xlabel('time(s)')
ax.set_ylabel('value')
ax.set_title('S-curve plot')
ax.legend(loc='upper left', frameon=False, ncol=2)
plt.grid()
plt.show()

print('ts = ', ts)
print('tc = ', tc)
print('ta = ', ta)
print('tv = ', tv)
print('t_all = ', t_all)
print('Sa = ', Sa)
