from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from math import radians, sin, cos, atan2, sqrt


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def dh_matrix(alpha, a, d, theta):
    # 传入四个DH参数，根据DH公式，输出一个T矩阵。
    alpha = alpha / 180 * np.pi
    theta = theta / 180 * np.pi
    matrix = np.identity(4)

    matrix[0, 0] = cos(theta)
    matrix[0, 1] = -sin(theta) * cos(alpha)
    matrix[0, 2] = sin(theta) * sin(alpha)
    matrix[0, 3] = cos(theta) * a
    matrix[1, 0] = sin(theta)
    matrix[1, 1] = cos(theta) * cos(alpha)
    matrix[1, 2] = -cos(theta) * sin(alpha)
    matrix[1, 3] = sin(theta) * a
    matrix[2, 0] = 0
    matrix[2, 1] = sin(alpha)
    matrix[2, 2] = cos(alpha)
    matrix[2, 3] = d
    matrix[3, 0] = 0
    matrix[3, 1] = 0
    matrix[3, 2] = 0
    matrix[3, 3] = 1
    return matrix


def isRotationMatrix(R) :
    # 判断是否是旋转矩阵
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R) :
    # 由旋转矩阵求解对应的XZY欧拉角。这里求解的是外旋方式下的XYZ顺序
    assert(isRotationMatrix(R))
    sy = sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = atan2(R[2,1] , R[2,2])
        y = atan2(-R[2,0], sy)
        z = atan2(R[1,0], R[0,0])
    else :
        x = atan2(-R[1,2], R[1,1])
        y = atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])






def cal_eepose(joints_angle):
    # 由各个关节角计算相应的末端位姿

    eepose = np.zeros((1, 6))
    joint_num = 6
    # DH参数表，分别用一个列表来表示每个关节的东西
    joints_alpha = [-90, 0, 90, -90, 90, 0]
    joints_a = [0, 480, 0, 0, 0, 0]
    joints_d = [260, 0, 0, 520, 0, 173.5]
    joints_theta = [0, -90, 90, 0, 0, 0]
    joint_hm = []
    for i in range(joint_num):
        joint_hm.append(dh_matrix(joints_alpha[i], joints_a[i], joints_d[i], joints_theta[i] + joints_angle[i]))
    # -----------计算齐次变换矩阵----------------------
    for i in range(joint_num - 1):
        joint_hm[i + 1] = joint_hm[i] @ joint_hm[i + 1]
    ee = joint_hm[5]   # 末端的齐次变换矩阵
    eepose[0, 0:3] = ee[:3, 3]
    xe = ee[0:3, 0]
    ye = ee[0:3, 1]
    ze = ee[0:3, 2]
    # # 下面的几个角度对应着RPY角,以弧度为单位
    # eepose[0, 3] = atan2(ze[0], ze[1])
    # eepose[0, 4] = atan2(sqrt(ze[0]**2+ze[1]**2), -ze[2])
    # eepose[0, 5] = atan2(xe[2], ye[2])
    eepose[0, 3:6] = rotationMatrixToEulerAngles(ee[0:3, 0:3])

    return eepose

#
# pose = cal_eepose(joints_angle=[0, 0, 0, 0, 90, 90])
# print(np.round(pose, 3))


