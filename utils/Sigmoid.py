from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, cos, sin, sqrt, pi, copysign
from scipy.integrate import cumtrapz


def calculate_Ts(D, S_max, V_max, A_max, J_max):
    """
    Calculate Ts_d, Ts_v, Ts_a, Ts_j, than select tha maximum value as Ts
    :param D:
    :param S_max:
    :param V_max:
    :param A_max:
    :param J_max:
    :return:
    """

    Ts_d = (sqrt(3) * abs(D) / (8 * S_max)) ** (0.25)
    Ts_v = (sqrt(3) * V_max / (2 * S_max)) ** (1 / 3)
    Ts_a = (sqrt(3) * A_max / S_max) ** 0.5
    Ts_j = sqrt(3) * J_max / S_max

    return Ts_d, Ts_v, Ts_a, Ts_j


def calculate_Tj(D, Ts, V_max, A_max, J_max):
    """
    Calculate Tj_d, Tj_v, Tj_a, than select tha maximum value as Tj
    :param D:
    :param Ts:
    :param V_max:
    :param A_max:
    :param J_max:
    :return:
    """
    Tj_d = (Ts ** 3 / 27 + abs(D) / (4 * J_max) + sqrt(
        abs(D) * Ts ** 3 / (54 * J_max) + D ** 2 / (16 * J_max ** 2))) ** (1 / 3) + (
                   Ts ** 3 / 27 + abs(D) / (4 * J_max) - sqrt(
               abs(D) * Ts ** 3 / (54 * J_max) + D ** 2 / (16 * J_max ** 2))) ** (1 / 3) - 5 * Ts / 3
    Tj_v = -3 * Ts / 2 + sqrt(Ts ** 2 / 4 + V_max / J_max)
    Tj_a = A_max / J_max - Ts

    return Tj_d, Tj_v, Tj_a


def calculate_Ta(D, Ts, Tj, V_max, A_max):
    """
    Calculate Ta_d, Ta_v, than select tha maximum value as Ta
    :param D:
    :param Ts:
    :param Tj:
    :param V_max:
    :param A_max:
    :return:
    """
    Ta_d = (-(6 * Ts + 3 * Tj) + sqrt((2 * Ts + Tj) ** 2 + 4 * abs(D) / A_max)) / 2
    Ta_v = (V_max - A_max * (Ts + Tj)) / A_max

    return Ta_d, Ta_v


def calculate_Tv(Ts, Tj, Ta, V_max, D):
    """
    Calculate Tv
    """
    Tv = abs(D) / V_max - (4 * Ts + 2 * Tj + Ta)

    return Tv


def calculate_Tall(D, Sm, Vm, Am, Jm):
    """
    Calculate Ts, Tj, Ta, Tv, jm
    """
    Tj = 0
    Ta = 0
    Tv = 0
    jm = 0
    Ts_d, Ts_v, Ts_a, Ts_j = calculate_Ts(D, Sm, Vm, Am, Jm)
    Ts = min(Ts_d, Ts_v, Ts_a, Ts_j)
    if Ts == Ts_d:
        jm = Sm * Ts / sqrt(3)
    elif Ts == Ts_v:
        jm = Sm * Ts / sqrt(3)
        Tv = calculate_Tv(Ts, Tj, Ta, Vm, D)

    elif Ts == Ts_a:
        jm = Sm * Ts / sqrt(3)
        Ta_d, Ta_v = calculate_Ta(D, Ts, Tj, Vm, Am)
        Ta = min(Ta_d, Ta_v)
        if Ta == Ta_v:
            Tv = calculate_Tv(Ts, Tj, Ta, Vm, D)
    elif Ts == Ts_j:
        jm = Jm
        Tj_d, Tj_v, Tj_a = calculate_Tj(D, Ts, Vm, Am, Jm)
        Tj = min(Tj_d, Tj_v, Tj_a)
        if Tj == Tj_v:
            Tv = calculate_Tv(Ts, Tj, Ta, Vm, D)
        elif Tj == Tj_a:
            Ta_d, Ta_v = calculate_Ta(D, Ts, Tj, Vm, Am)
            Ta = min(Ta_d, Ta_v)
            if Ta == Ta_v:
                Tv = calculate_Tv(Ts, Tj, Ta, Vm, D)

    return Ts, Tj, Ta, Tv, jm


def calculate_t(Ts, Tj, Ta, Tv):
    """
    Calculate time interval
    """
    # 计算各个时间段

    time_interval = np.zeros(16)
    time_interval[0] = 0
    time_interval[1] = Ts
    time_interval[2] = Ts + Tj
    time_interval[3] = Ts + Tj + Ts
    time_interval[4] = Ts + Tj + Ts + Ta
    time_interval[5] = Ts + Tj + Ts + Ta + Ts
    time_interval[6] = Ts + Tj + Ts + Ta + Ts + Tj
    time_interval[7] = Ts + Tj + Ts + Ta + Ts + Tj + Ts
    time_interval[8] = Ts + Tj + Ts + Ta + Ts + Tj + Ts + Tv
    time_interval[9] = Ts + Tj + Ts + Ta + Ts + Tj + Ts + Tv + Ts
    time_interval[10] = Ts + Tj + Ts + Ta + Ts + Tj + Ts + Tv + Ts + Tj
    time_interval[11] = Ts + Tj + Ts + Ta + Ts + Tj + Ts + Tv + Ts + Tj + Ts
    time_interval[12] = Ts + Tj + Ts + Ta + Ts + Tj + Ts + Tv + Ts + Tj + Ts + Ta
    time_interval[13] = Ts + Tj + Ts + Ta + Ts + Tj + Ts + Tv + Ts + Tj + Ts + Ta + Ts
    time_interval[14] = Ts + Tj + Ts + Ta + Ts + Tj + Ts + Tv + Ts + Tj + Ts + Ta + Ts + Tj
    time_interval[15] = Ts + Tj + Ts + Ta + Ts + Tj + Ts + Tv + Ts + Tj + Ts + Ta + Ts + Tj + Ts

    return time_interval


def calculate_jerk(t, tk, jerk, J_max, a):
    """
    Calculate jerk curve
    """
    # 计算jerk曲线

    # 1,13
    k = (t >= tk[0]) & (t < tk[1])
    tau = (t[k] - tk[0]) / (tk[1] - tk[0])
    jerk[k] = J_max / (1 + np.exp(-a * (1 / (1 - tau) - 1 / tau)))
    k = (t >= tk[12]) & (t < tk[13])
    tau = (t[k] - tk[12]) / (tk[13] - tk[12])
    jerk[k] = J_max / (1 + np.exp(-a * (1 / (1 - tau) - 1 / tau)))
    # 2,14
    k = (t >= tk[1]) & (t < tk[2])
    jerk[k] = J_max
    k = (t >= tk[13]) & (t < tk[14])
    jerk[k] = J_max
    # 3,15
    k = (t >= tk[2]) & (t < tk[3])
    tau = (t[k] - tk[2]) / (tk[3] - tk[2])
    jerk[k] = J_max / (1 + np.exp(a * (1 / (1 - tau) - 1 / tau)))
    k = (t >= tk[14]) & (t <= tk[15])
    tau = (t[k] - tk[14]) / (tk[15] - tk[14])
    jerk[k] = J_max / (1 + np.exp(a * (1 / (1 - tau) - 1 / tau)))
    # 4,8,12
    k = (t >= tk[3]) & (t < tk[4])
    jerk[k] = 0
    k = (t >= tk[7]) & (t < tk[8])
    jerk[k] = 0
    k = (t >= tk[11]) & (t < tk[12])
    jerk[k] = 0
    # 5,9
    k = (t >= tk[4]) & (t < tk[5])
    tau = (t[k] - tk[4]) / (tk[5] - tk[4])
    jerk[k] = -J_max / (1 + np.exp(-a * (1 / (1 - tau) - 1 / tau)))
    k = (t >= tk[8]) & (t < tk[9])
    tau = (t[k] - tk[8]) / (tk[9] - tk[8])
    jerk[k] = -J_max / (1 + np.exp(-a * (1 / (1 - tau) - 1 / tau)))
    # 6,10
    k = (t >= tk[5]) & (t < tk[6])
    jerk[k] = -J_max
    k = (t >= tk[9]) & (t < tk[10])
    jerk[k] = -J_max
    # 7,11
    k = (t >= tk[6]) & (t < tk[7])
    tau = (t[k] - tk[6]) / (tk[7] - tk[6])
    jerk[k] = -J_max / (1 + np.exp(a * (1 / (1 - tau) - 1 / tau)))
    k = (t >= tk[10]) & (t < tk[11])
    tau = (t[k] - tk[10]) / (tk[11] - tk[10])
    jerk[k] = -J_max / (1 + np.exp(a * (1 / (1 - tau) - 1 / tau)))

    return jerk


def adjust_time_parameters(Dk, Sm, Vm, Am, Jm):
    """
    调整时间参数
    :param Dk: 每个轴的位移列表
    :param Sm: 最大加加加速度
    :param Vm: 每个轴的最大速度列表
    :param Am: 每个轴的最大加速度列表
    :param Jm: 每个轴的初始最大jerk列表
    :return: 调整后的时间参数
    """
    # 自由度数
    n = len(Dk)
    # 定义各个时间列表
    T_all = np.zeros(n)
    Ts = np.zeros(n)
    Tj = np.zeros(n)
    Ta = np.zeros(n)
    Tv = np.zeros(n)
    jm = np.zeros(n)

    # 计算每个轴的时间参数
    for i in range(n):
        Ts[i], Tj[i], Ta[i], Tv[i], jm[i] = calculate_Tall(Dk[i], Sm, Vm[i], Am[i], Jm[i])
        T_all[i] = 8 * Ts[i] + 4 * Tj[i] + 2 * Ta[i] + Tv[i]

    # 时间同步
    T_sync_all = max(T_all)  # 计算所有时间中的最大值
    Lambda = T_sync_all / T_all  # 计算同步因子
    Ts_sync = Ts * Lambda  # 同步时间
    Tj_sync = Tj * Lambda  # 同步时间
    Ta_sync = Ta * Lambda  # 同步时间
    Tv_sync = Tv * Lambda  # 同步时间
    Jm_sync = 1 / Lambda ** 3 * jm  # 同步Jm

    return T_sync_all, Ts_sync, Tj_sync, Ta_sync, Tv_sync, Jm_sync

def curve(D,Dk,Sm,Vm,Am,Jm,num = 1000):
    """
    Sigmoid_scurve
    :param D: 当前位置
    :param Dk: 变动位置量
    :param Sm: 最大加加加速度
    :param Vm: 每个轴的最大速度列表
    :param Am: 每个轴的最大加速度列表
    :param Jm: 每个轴的初始最大jerk列表
    :param num: 时间点数
    :return: 时间序列和曲线

    注意：支持使用弧度制和角度制
    """

    # 自由度数
    n = len(Dk)
    # 计算时间
    a = sqrt(3) / 2  # sqrt(3)/2是一个分界点，sigmoid的模型参数

    T_sync_all, Ts_sync, Tj_sync, Ta_sync, Tv_sync, Jm_sync = adjust_time_parameters(Dk, Sm, Vm, Am, Jm)

    # 生成时间序列和初始jerk曲线
    t = np.linspace(0, T_sync_all, num=num)
    jerk = np.zeros((n, len(t)))  # 初始jerk曲线
    acceleration = np.zeros((n, len(t)))  # 加速度曲线
    velocity = np.zeros((n, len(t)))  # 速度曲线
    displacement = np.zeros((n, len(t)))  # 位移曲线
    tk = np.zeros((n, 16))  # 计算对应的时间段

    # 生成曲线
    for i in range(n):
        # 计算对应的时间段
        tk[i] = calculate_t(Ts_sync[i], Tj_sync[i], Ta_sync[i], Tv_sync[i])
        jerk[i] = copysign(1, Dk[i]) * calculate_jerk(t, tk[i], jerk[i], Jm_sync[i], a)

        # 计算加速度曲线，jerk的积分
        acceleration[i] = cumtrapz(jerk[i], t, initial=0)

        # 计算速度曲线，加速度的积分
        velocity[i] = cumtrapz(acceleration[i], t, initial=0)

        # 计算位移曲线，速度的积分
        displacement[i] = D[i] * np.ones(t.shape) + cumtrapz(velocity[i], t, initial=0)

    return t, displacement, velocity, acceleration, jerk

def plot_curve(t, d, v, a, j):
    """
    绘制曲线
    :param t:
    :param d:
    :param v:
    :param a:
    :param j:
    :return:
    """
    # 获取最大时间
    maxt = max(t)
    n = 6
    # 绘制曲线
    plt.figure(figsize=(6, 12))
    labels = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
    plt.subplot(4, 1, 1)
    plt.xlim(0, maxt + 0.25)
    plt.grid()
    for i in range(n):
        plt.plot(t, d[i], label=labels[i], linewidth=1.5)
        # plt.plot(t, Dk[i] * np.ones(t.shape), '--', linewidth=1.5, color='gray')
    plt.ylabel('Displacement$(rad)$', fontsize=14)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=3, fontsize=14)

    plt.subplot(4, 1, 2)
    plt.xlim(0, maxt + 0.25)
    plt.grid()
    for i in range(n):
        plt.plot(t, v[i], linewidth=1.5)
    plt.ylabel('Velocity$(rad/s)$', fontsize=14)

    plt.subplot(4, 1, 3)
    plt.xlim(0, maxt + 0.25)
    plt.grid()
    for i in range(n):
        plt.plot(t, a[i], linewidth=1.5)
    plt.ylabel('Acceleration$(rad/s^2)$', fontsize=14)

    plt.subplot(4, 1, 4)
    plt.xlim(0, maxt + 0.25)
    plt.grid()
    for i in range(n):
        plt.plot(t, j[i], linewidth=1.5)
    plt.xlabel('Time$(s)$', fontsize=14)
    plt.ylabel('Jerk$(rad/s^3)$', fontsize=14)
    plt.tight_layout()
    plt.show()
    return []
