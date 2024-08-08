from sko.PSO import PSO
import matplotlib.pyplot as plt
import numpy as np
import time
def fitness_t(x):
    # 取决于x的项数，中间有多少个点就有多少个时间段
    return sum(x)

def cal_T(particle_x):

    '''
    基于多项式插值的矩阵求解函数
    :param particle_x: 每一个粒子的位置
    :return:    用于求解多项式系数的矩阵T

    '''
    dimension = len(particle_x)
    tk = particle_x.copy()
    T = np.zeros([4*dimension, 4*dimension])
    if dimension == 2:
        A = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, tk[0], tk[0] ** 2, tk[0] ** 3], [0, 0, 0, 0]])
        C = np.array(
            [[0, -1, 0, 0], [0, 0, -2, 0], [1, tk[-1], tk[-1] ** 2, tk[-1] ** 3], [0, 1, 2 * tk[-1], 3 * tk[-1]**2]])
        D = np.zeros([4, 4])
        D[3, 0] = 1
        E = np.array([[0, 1, 2 * tk[0], 3 * tk[0] ** 2], [0, 0, 2, 6 * tk[0]], [0, 0, 0, 0], [0, 0, 0, 0]])
        T[0:4, 0:4] = A
        T[-4:, -4:] = C
        T[0:4, -4:] = D
        T[-4:, 0:4] = E
    elif dimension >= 3:
        num_B = dimension-2
        A = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, tk[0], tk[0] ** 2, tk[0] ** 3], [0, 0, 0, 0]])
        C = np.array(
            [[0, -1, 0, 0], [0, 0, -2, 0], [1, tk[-1], tk[-1] ** 2, tk[-1] ** 3], [0, 1, 2 * tk[-1], 3 * tk[-1] ** 2]])
        D = np.zeros([4, 4])
        D[3, 0] = 1
        B = [np.array([[0,-1,0,0],[0,0,-2,0],[1,tk[i+1],tk[i+1]**2,tk[i+1]**3],[0,0,0,0]]) for i in range(num_B)]
        E = [np.array([[0,1,2*tk[i],3*tk[i]**2],[0,0,2,6*tk[i]],[0,0,0,0],[0,0,0,0]]) for i in range(num_B+1)]
        T[0:4, 0:4] = A
        T[-4:, -4:] = C
        for i in range(num_B):
            T[4*(i+1):4*(i+2), 4*(i+1):4*(i+2)] = B[i]
        for i in range(num_B+1):
            T[4*(i+1):4*(i+2), 4*i:4*(i+1)] = E[i]
        for i in range(num_B+1):
            T[4*i:4*(i+1), 4*(i+1):4*(i+2)] = D

    t = [np.linspace(0, tk[i], 50) for i in range(dimension)]
    return t, T

def assess_feasibility(particle_x,theta,Vmax,Amax,Coeff=False):
    '''
    ACTION:

    TO CHECK IF THE SOLUTION IS FEASIBLE OR NOT

    :param particle_x: 每一个nest的位置
    :param theta:       需要经过的路径点,因为4x6的数组
    :param Vm:      各个关节的最大速度，应为1x6的数组
    :param Am:       各个关节的最大加速度，应为1x6的数组
    :param Coeff:   是否需要返回多项式系数，默认为False
    :return: feasibility 是否满足约束条件，若满足，则返回True，否则返回False
    '''

    dimension = len(particle_x)
    t,T = cal_T(particle_x.copy())
    theta = theta.copy().T
    Vm = Vmax.copy()
    Am = Amax.copy()        # 都写成copy了，防止在函数内修改原数组
    n = len(Vm)
    Trans2 = np.array([[2, 0, 0], [0, 1, 0]])
    Trans3 = np.array([[3, 0, 0, 0], [0, 2, 0, 0], [0, 0, 1, 0]])
    if Coeff:       # 需要输出各个轴的多项式系数
        coeff = [np.zeros([n, 4]) for i in range(dimension)]
        thetai = np.zeros(4*dimension)
        for i in range(n):
            thetai[0] = theta[i][0]
            thetai[-2] = theta[i][-1]
            for j in range(1, dimension):
                thetai[4*j-2:4*j+2] = theta[i][j], theta[i][j], 0, 0
            coeffs = np.linalg.solve(T, thetai.T)
            for j in range(dimension):
                coeff[j][i] = coeffs[4*j:4*j+4]
                coeff[j][i] = coeff[j][i][::-1]

        return coeff        # 返回6个轴分别的多项式系数

    for i in range(n):
        thetai = np.zeros(4 * dimension)
        thetai[0] = theta[i][0]
        thetai[-2] = theta[i][-1]
        for j in range(1, dimension):
            thetai[4 * j - 2:4 * j + 2] = theta[i][j], theta[i][j], 0, 0
        coeffs = np.linalg.solve(T, thetai.T)
        coeff = [coeffs[4 * j:4 * j + 4][::-1] for j in range(dimension)]
        coef = [Trans3 @ coeff[j] for j in range(dimension)] # 每一段的速度的系数
        coe = [Trans2 @ coef[j] for j in range(dimension)] # 每一段的加速度的系数

        for j in range(dimension):
            v = np.polyval(coef[j], t[j])    # 每一段的速度
            acc = np.polyval(coe[j], t[j])  # 每一段的加速度
            if not np.all((abs(v) <= Vm[i]) & (abs(acc) <= Am[i])):
                return 1
    return 0

f = 200             # 轨迹更新频率
# 6轴理论的最大值
Vm = np.array([100, 100, 150, 150, 180, 180])
Am = np.array([360, 360, 360, 360, 360, 360])
n = 6        # 自由度数量
numofpoints = 4     # 经过点的个数
# 各个位移点
J = np.zeros([numofpoints, 6])
J[0] = [-98.266, -2.042, -105.752, -0.666, -76.064, 36.678]
J[1] = [-53.007, -1.49, -79.635, 0.002, -101.862, 81.98]
J[2] = [57.235, -4.219, -80.696, 0.002, -103.518, -167.697]
J[3] = [97.567, -2.906, -106.881, -0.722, -74.602, -127.26]

constraint_uep = (lambda x: assess_feasibility(x,J,Vm,Am),)
n_dim = numofpoints-1
# 自适应上下界
lb, ub = np.zeros(n_dim), np.zeros(n_dim)
for i in range(n_dim):
    delta = []
    for j in range(n):
        delta.append(abs(J[i+1][j] - J[i][j])/Vm[j])
    ub[i] = 2.5*max(delta)
    lb[i] = 0.1 * max(delta)

pso = PSO(func=fitness_t, n_dim=n_dim, pop=50, max_iter=100,
          lb=lb, ub=ub, w=0.8, c1=0.5, c2=0.5, constraint_ueq=constraint_uep)

t1 = time.time()
pso.run()
t2 = time.time()
print('time is ', t2 - t1)
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
plt.plot(pso.gbest_y_hist)
plt.title('PSO Result')
plt.xlabel('Iteration',fontsize=14)
plt.ylabel('Time_all(s)',fontsize=14)
plt.grid()
plt.show()


coeff = assess_feasibility(pso.gbest_x,J,Vm,Am,Coeff=True)
dimension = len(pso.gbest_x)
t = np.arange(0, sum(pso.gbest_x) + 1 / f, 1 / f)
y = np.zeros([6,len(t)])
v_all = np.zeros([6,len(t)])
tk = [sum(pso.gbest_x[:j]) for j in range(dimension+1)]
for i in range(n):
    for j in range(dimension):
        k = (t >= tk[j]) & (t <= tk[j+1]+1/f)
        tau = t[k] - tk[j]
        y[i][k] = np.polyval(coeff[j][i], tau)
    
    # 计算速度
    v_all[i] = np.gradient(y[i], t)

# 获取最大时间
maxt = max(t)
# 绘制曲线
plt.figure(figsize=(8, 8))
labels = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
plt.subplot(2, 1, 1)
plt.xlim(0, maxt + 0.25)
for i in range(n):
    plt.plot(t, y[i], label=labels[i], linewidth=1.5)
plt.ylabel('Displacement$(deg)$', fontsize=14)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.27), ncol=3, fontsize=14)
plt.grid()
plt.subplot(2, 1, 2)
plt.xlim(0, maxt + 0.25)

a = np.zeros(v_all.shape)
for i in range(6):
    a[i] = np.gradient(v_all[i],t)

for i in range(n):
    plt.plot(t, v_all[i], label=labels[i], linewidth=1.5)
plt.xlabel('Time$(s)$', fontsize=14)
plt.ylabel('Velocity$(deg/s)$', fontsize=14)
plt.grid()
plt.show()
