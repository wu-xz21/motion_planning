from utils.cso import CSO
import numpy as np
import time

def fitness_0(X):
    return sum(X)
if __name__ == '__main__':
    f = 200  # 轨迹更新频率
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

    model = CSO(fitness_0, J,Vm,Am,Tmax=100,P=50, verbose=True,n=3,bound=[(0.5,5),(0.5,5),(0.5,5)])
    t1 = time.time()
    model.execute()
    t2 = time.time()
    print('time:',t2-t1)
    model.Fplot()
