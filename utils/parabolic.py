import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz

"""
此文件包含抛物线轨迹规划，使中间点的速度不为零。
可以手动设置每一段的加速度和每一段的时间，同时算法提供了能够每一段能够达到的最短时间。
"""


def assess_time(pos, a, t):
    """
    计算每一段抛物线的时间，并且评估时间是否足够
    :return: T_init T_end（第一段和最后一段的过渡时间）    Tmin（前后段能达到的最短时间）
    """
    if pos[1] < pos[0]:
        a[0] = -a[0]
    if pos[-2] < pos[-1]:
        a[-1] = -a[-1]
    Tmin1= sqrt(2 * (pos[1] - pos[0]) / a[0])  # 第一段的最短时间
    Tmin2= sqrt(-2 * (pos[-1] - pos[-2]) / a[-1])  # 第二段的最短时间
    Tmin = np.array([Tmin1, Tmin2])

    judge1 = t[0] ** 2 - 2 * (pos[1] - pos[0]) / a[0]
    judge2 = t[-1] ** 2 + 2 * (pos[-1] - pos[-2]) / a[-1]

    # 判断时间是否足够
    if judge1 < 0:
        raise ValueError(f"第一段曲线时间太短，轨迹无法生成，请至少提高时间{-judge1 + 0.1:.2f}s！")
    if judge2 < 0:
        raise ValueError(f"最后一段曲线时间太短，轨迹无法生成，请至少提高时间{-judge2 + 0.1:.2f}s！")

    # 计算抛物线的时间
    T_init = t[0] - sqrt(judge1)
    T_end = t[-1] - sqrt(judge2)

    return T_init, T_end, Tmin


def calculate_tv(Point, Acc, Td, Vm):
    """
    计算每一段抛物线的速度和时间
    :param Point:   经过的路径点，包括起始点和终点以及中间的控制点
    :param Acc:   每一段的加速度，一般取最大加速度的80%
    :param Td:   每一段的时间，如果报错，需要调整各段的时间
    :return:     tk: 包含各段时间节点的列表    v_interval: 每一段的速度     tmin: 每一段能够达到的最短时间
    """
    n = len(Point)
    t_interval = np.zeros(2 * n - 1)
    v_interval = np.zeros(n - 1)  # 每一段抛物线的速度
    tmin = np.zeros(Td.shape)       # 定义每一段的最短时间

    # 计算抛物线和线性区域的时间以及每一段的速度
    t_interval[0], t_interval[-1], [tmin[0], tmin[-1]] = assess_time(Point, Acc, Td)  # 前后两端的时间
    v_interval[0] = (Point[1] - Point[0]) / (Td[0] - 0.5 * t_interval[0])  # 起始段速度
    if abs(Point[1] - Point[0])/Vm + 0.5 * t_interval[0] > tmin[0]:    # 检查是否时间太短
        tmin[0] = abs(Point[1] - Point[0])/Vm + 0.5 * t_interval[0]
    v_interval[-1] = (Point[-1] - Point[-2]) / (Td[-1] - 0.5 * t_interval[-1])  # 终点段速度
    if abs(Point[-1] - Point[-2])/Vm + 0.5 * t_interval[-1] > tmin[-1]:    # 检查是否时间太短
        tmin[-1] = abs(Point[-1] - Point[-2])/Vm + 0.5 * t_interval[-1]
    if n > 3:  # 如果中间有控制点
        for i in range(1, n - 2):
            v_interval[i] = (Point[i + 1] - Point[i]) / Td[i]  # 内部段速度
            if abs(Point[i + 1] - Point[i]) / Vm > tmin[i]:    # 检查是否时间太短
                tmin[i] = abs(Point[i + 1] - Point[i]) / Vm
    for i in range(1, n - 1):
        if v_interval[i] < v_interval[i - 1]:
            Acc[i] = -Acc[i]
        t_interval[2 * i] = (v_interval[i] - v_interval[i - 1]) / Acc[i]  # 每一段抛物线的时间
    t_interval[1] = Td[0] - t_interval[0] - 0.5 * t_interval[2]  # 第一段线性区域的时间
    if t_interval[0] + 0.5 * t_interval[2] > tmin[0]:   # 检查是否时间太短
        tmin[0] = t_interval[0] + 0.5 * t_interval[2]
    if t_interval[1] < 0:
        raise ValueError(f"第一段曲线时间过短，请至少提高时间{-t_interval[1] + 0.1:.2f}s！")
    t_interval[-2] = Td[-1] - t_interval[-1] - 0.5 * t_interval[-3]  # 最后一段线性区域的时间
    if t_interval[-1] + 0.5 * t_interval[-3] > tmin[-1]:    # 检查是否时间太短
        tmin[-1] = t_interval[-1] + 0.5 * t_interval[-3]
    if t_interval[-2] < 0:
        raise ValueError(f"最后一段曲线时间过短，请至少提高时间{-t_interval[-2] + 0.1:.2f}s！")
    if n > 3:  # 如果中间有控制点
        for i in range(1, n - 2):
            t_interval[2 * i + 1] = Td[i] - 0.5 * t_interval[2 * i] - 0.5 * t_interval[2 * i + 2]
            if 0.5 * t_interval[2 * i] + 0.5 * t_interval[2 * i + 2] > tmin[i]:    # 检查是否时间太短
                tmin[i] = 0.5 * t_interval[2 * i] + 0.5 * t_interval[2 * i + 2]
            # 检查是否时间太短
            try:
                if t_interval[2 * i + 1] < 0:
                    raise ValueError(f"第{i + 1}段时间过短，请至少提高时间{-t_interval[2 * i + 1] + 0.1:.2f}s！")
            except ValueError as e:
                exit(3)
    # 计算每段抛物线的起始时间
    tk = np.zeros(2 * n)
    for i in range(1, 2 * n):
        tk[i] = tk[i - 1] + t_interval[i - 1]
    tk[-1] = sum(Td)

    # 检查速度是否超过最大值
    for i in range(n - 1):
        V_abs = abs(v_interval[i])
        if V_abs > Vm:
            rate = V_abs / Vm
            raise ValueError(
                f"第{i + 1}段速度大小为{V_abs:.2f}°/s,超过最大值{Vm:.2f}°/s！请至少提高第{i + 1}段的时间{rate + 0.05:.2f}倍")
    return tk, v_interval, tmin


def calculate_vp(tk, v_interval, a, f, n):
    """
    积分计算每一段的速度和位置
    :param tk:   每一段抛物线的时间
    :param v_interval:   每一段线性区域的速度
    :param a:    过渡区域的加速度
    :param f:    轨迹更新频率
    :param n:   经过点的个数
    :return:     t, v, p（时间、速度和位置）
    """
    # 定义各段的抛物线轨迹
    t = np.arange(0, tk[-1] + 1 / f, 1 / f)
    v = np.zeros(t.shape)
    # 起始段
    k = (t >= tk[0]) & (t < tk[1])
    tau = t[k] - tk[0]
    v[k] = a[0] * np.array(tau)
    if n > 2:  # 如果中间有控制点
        for i in range(1, 2 * n - 2, 2):
            k = (t >= tk[i]) & (t < tk[i + 1])
            tau = t[k] - tk[i]
            v[k] = v_interval[int((i - 1) / 2)]
        for i in range(2, 2 * n - 2, 2):
            # 2,3
            k = (t >= tk[i]) & (t < tk[i + 1])
            tau = t[k] - tk[i]
            v[k] = v_interval[int((i - 2) / 2)] + a[int(i / 2)] * tau
    # 终点段
    k = (t >= tk[-2]) & (t <= tk[-1])
    tau = t[k] - tk[-2]
    v[k] = v_interval[-1] + a[-1] * tau
    p = cumtrapz(v, t, initial=0)
    return t, v, p


def curve(J, Am, Td, Vm, f):
    """
    计算所有轴的轨迹
    :param Point:   控制点
    :param Acc:     加速度
    :param Td:      每一段的时间
    :param Vm:      最大速度
    :param f:        轨迹更新频率
    :return:        t, v_all, pos（时间、速度和位置）和 tmin（最短时间）
    """
    J = J.T
    n = len(Am)
    numofpoints = len(J[0])
    tmin = np.zeros(numofpoints - 1)    # 记录最短需要的时间
    k = True
    for i in range(n):
        Point = J[i]
        Acc = Am[i] * np.ones(numofpoints)
        tk, v_interval, tmin_k = calculate_tv(Point, Acc, Td, Vm[i])  # 计算每段抛物线的时间和线性区域的速度
        for j in range(numofpoints - 1):
            if tmin_k[j] > tmin[j]:
                tmin[j] = tmin_k[j]
        t, v, p = calculate_vp(tk, v_interval, Acc, f, numofpoints)  # 计算整段曲线的速度和位置
        if k:
            pos = np.zeros([n, len(t)])
            v_all = np.zeros([n, len(t)])
            k = False
        pos[i] = Point[0] + p
        v_all[i] = v
    return t, v_all, pos, tmin


if __name__ == '__main__':
    f = 200  # 轨迹更新频率
    # 6轴理论的最大值
    Vm = np.array([100, 100, 150, 150, 180, 180])
    Am = np.array([360, 360, 360, 360, 360, 360])
    Jm = np.array([375, 375, 400, 400, 450, 450])
    Sm = 4000

    n = 6       # 自由度数量

    # numofpoints = 4     # 经过点的个数
    # # 各个位移点
    # J = np.zeros([numofpoints, 6])
    # J[0] = [-98.266, -2.042, -105.752, -0.666, -76.064, 36.678]
    # J[1] = [-53.007, -1.49, -79.635, 0.002, -101.862, 81.98]
    # J[2] = [57.235, -4.219, -80.696, 0.002, -103.518, -167.697]
    # J[3] = [97.567, -2.906, -106.881, -0.722, -74.602, -127.26]

    numofpoints = 5
    J = np.zeros([5, 6])
    J[0] = [-98.266, -2.042, -105.75200000000001, -0.666, -76.064, 36.67799999999999]
    J[1] = [-59.40675810375775, 3.9911245414573915, -79.68075151351047, 26.20631430014368, -63.59781730453054, 49.79423307294334]
    J[2] = [87.96875172854794, -0.9614177542192305, -98.76389421015637, 7.522834146132491, -70.94063802023562, -105.23040210518761]
    J[3] = [95.94334562707107, -1.5788788854289766, -101.27759174643226, 4.997634788239193, -72.02378714868595, -116.18996031101267]
    J[4] = [97.567, -2.906, -106.881, -0.7220000000000001, -74.602, -127.25999999999999]

    Td = np.array([0.53, 1.475, 0.157, 0.25])  # 每一段的时间

    # Td = np.array([0.63,1.39,0.6])  # 每一段的时间
    # Td = np.array([0.62668,1.3871,0.5991])  # 每一段的时间
    t, v_all, pos, tmin = curve(J, Am, Td, Vm, f)  # 计算所有轴的轨迹
    print("给定条件下的各段曲线的最短时间分别为：{}s".format(tmin))
    print("给定曲线下的总时间为：{}s".format(max(t)))

    # 获取最大时间
    maxt = max(t)
    # 绘制曲线
    plt.figure(figsize=(8, 8))
    labels = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
    plt.subplot(2, 1, 1)
    plt.xlim(0, maxt + 0.25)
    for i in range(n):
        plt.plot(t, pos[i], label=labels[i], linewidth=1.5)
    plt.ylabel('Displacement$(deg)$', fontsize=14)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.27), ncol=3, fontsize=14)
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.xlim(0, maxt + 0.25)
    for i in range(n):
        plt.plot(t, v_all[i], label=labels[i], linewidth=1.5)
    plt.xlabel('Time$(s)$', fontsize=14)
    plt.ylabel('Velocity$(deg/s)$', fontsize=14)
    plt.grid()
    plt.show()
