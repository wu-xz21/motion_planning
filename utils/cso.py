################################################################################
#                                                                              #
#	UJJWAL KHANDELWAL                                                      #    
#	CSO (CUCKOO SEARCH OPTIMIZATION)                                       #
#	PYTHON 3.7.10                                                          #
#    这个代码经过伍贤知的修改，其中用到了while True的地方，都是约束条件的判断         #
################################################################################

#######################   IMPORT DEPENDENCIES   ################################

import numpy as np
import matplotlib.pyplot as plt
from math import gamma


###########################  CSO CLASS  ########################################

class CSO:

    def __init__(self, fitness, theta,Vm,Am,P=50, n=3, pa=0.25, beta=1.5, bound=None,
                 plot=False, min=True, verbose=False, Tmax=100):

        '''

        PARAMETERS:
        
        fitness: A FUNCTION WHICH EVALUATES COST (OR THE FITNESS) VALUE

        P: POPULATION SIZE

        n: TOTAL DIMENSIONS

        pa: ASSIGNED PROBABILITY

        beta: LEVY PARAMETER

        bound: AXIS BOUND FOR EACH DIMENSION

        X: PARTICLE POSITION OF SHAPE (P,n)

        ################ EXAMPLE #####################
        
        If ith egg Xi = [x,y,z], n = 3, and if
        bound = [(-5,5),(-1,1),(0,5)]
        Then, x∈(-5,5); y∈(-1,1); z∈(0,5)

        ##############################################

        Tmax: MAXIMUM ITERATION

        best: GLOBAL BEST POSITION OF SHAPE (n,1)
        
        '''
        self.Vm = Vm
        self.theta = theta
        self.Am = Am
        self.fitness = fitness
        self.P = P
        self.n = n
        self.Tmax = Tmax
        self.pa = pa
        self.beta = beta
        self.bound = bound
        self.plot = plot
        self.min = min  # IF TRUE, THEN MINIMIZATION PROBLEM, ELSE MAXIMIZATION PROBLEM
        self.verbose = verbose

        # X = (U-L)*rand + L (U AND L ARE UPPER AND LOWER BOUND OF X)
        # U AND L VARY BASED ON THE DIFFERENT DIMENSION OF X

        self.X = []

        # INIT THE NEST POSITION
        if bound is not None:
            for _ in range(P):
                while True:
                    x = []
                    for (U, L) in bound:
                        ret = (U - L) * np.random.rand() + L        # 产生一个给定范围内的随机数
                        x.append(ret)
                    if self.assess_feasibility(x,self.theta, self.Vm, self.Am):  # 只有满足约束的值才能被视为一个解，不然会被直接丢弃
                        self.X.append(x)
                        break
            self.X = np.array(self.X)
        else:
            self.X = np.random.randn(P, n)          # 这里没有做改动，因为不考虑不设边界的情况
        self.update_position_1()

    def update_position_1(self):

        '''
        
        ACTION:

        TO CALCULATE THE CHANGE OF POSITION 'X = X + rand*C' USING LEVY FLIGHT METHOD

        C = 0.01*S*(X-best) WHERE S IS THE RANDOM STEP, and β = beta (TAKEN FROM [1])

              u
        S = -----
                1/β
             |v|

        beta = 1.5

        u ~ N(0,σu) # NORMAL DISTRIBUTION WITH ZERO MEAN AND 'σu' STANDARD DEVIATION

        v ~ N(0,σv) # NORMAL DISTRIBUTION WITH ZERO MEAN AND 'σv' STANDARD DEVIATION

        σv = 1
        
                     Γ(1+β)*sin(πβ/2)       
        σu^β = --------------------------
                   Γ((1+β)/2)*β*(2^((β-1)/2))

        Γ IS THE GAMMA FUNCTION

        '''

        num = gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2)
        den = gamma((1 + self.beta) / 2) * self.beta * (2 ** ((self.beta - 1) / 2))
        σu = (num / den) ** (1 / self.beta)
        σv = 1
        u = np.random.normal(0, σu, self.n)
        v = np.random.normal(0, σv, self.n)
        S = u / (np.abs(v) ** (1 / self.beta))

        # DEFINING GLOBAL BEST SOLUTION BASED ON FITNESS VALUE

        for i in range(self.P):
            if i == 0:
                self.best = self.X[i, :].copy()
            else:
                self.best = self.optimum(self.best, self.X[i, :])

        Xnew = self.X.copy()
        for i in range(self.P):
            while True:
                Xnew[i, :] += np.random.randn(self.n) * 0.01 * S * (Xnew[i, :] - self.best)
                for j in range(self.n):
                    Xnew[i, j] = np.clip(Xnew[i, j], self.bound[j][0], self.bound[j][1])
                if self.assess_feasibility(Xnew[i, :], self.theta, self.Vm, self.Am):       # 只有满足约束的值才能被视为一个解，不然会被直接丢弃
                    break
            self.X[i, :] = self.optimum(Xnew[i, :], self.X[i, :])


    def update_position_2(self):

        '''
        
        ACTION:

        TO REPLACE SOME NEST WITH NEW SOLUTIONS

        HOST BIRD CAN THROW EGG AWAY (ABANDON THE NEST) WITH FRACTION
        
        pa ∈ [0,1] (ALSO CALLED ASSIGNED PROBABILITY) AND BUILD A COMPLETELY 
        
        NEW NEST. FIRST WE CHOOSE A RANDOM NUMBER r ∈ [0,1] AND IF r < pa,

        THEN 'X' IS SELECTED AND MODIFIED ELSE IT IS KEPT AS IT IS. 

        '''

        Xnew = self.X.copy()
        Xold = self.X.copy()
        for i in range(self.P):
            # d1, d2 = np.random.randint(0, self.P, 2)        # SELECTING TWO RANDOM PARTICLES(这里原来是（0,5,2）不知道为什么，我觉得应该是把5改成nest的数量。
            while True:
                for j in range(self.n):
                    d1, d2 = np.random.randint(0, self.P, 2)
                    r = np.random.rand()
                    xmin, xmax = self.bound[j]      # 定义域
                    if r < self.pa:
                        Xnew[i, j] += np.random.rand() * (Xold[d1, j] - Xold[d2, j])
                        Xnew[i, j] = np.clip(Xnew[i, j], xmin, xmax)        # 限制解在定义域内
                if self.assess_feasibility(Xnew[i, :], self.theta, self.Vm, self.Am):       # 只有满足约束的值才能被视为一个解，不然会被直接丢弃
                    break
            self.X[i, :] = self.optimum(Xnew[i, :], self.X[i, :])


    def cal_T(self,particle_x):

        '''

        :param particle_x: 每一个nest的位置
        :return:    用于求解多项式系数的矩阵T

        '''
        t1 = particle_x[0]
        t2 = particle_x[1]
        t3 = particle_x[2]

        A = np.array([[t1**3,t1**2,t1,1],[3*t1**2,2*t1,1,0],[6*t1,2,0,0]])
        B = np.array([[0,0,0,0,0,-1],[0,0,0,0,-1,0],[0,0,0,-2,0,0]])
        C = np.array([[t2**5,t2**4,t2**3,t2**2,t2,1],[5*t2**4,4*t2**3,3*t2**2,2*t2,1,0],[20*t2**3,12*t2**2,6*t2,2,0,0]])
        D = np.array([[0,0,0,-1],[0,0,-1,0],[0,-2,0,0]])
        E = np.array([[t3**3,t3**2,t3,1],[3*t3**2,2*t3,1,0],[6*t3,2,0,0]])
        F = np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]])
        G = np.zeros([5,6])
        G[4][5] = 1
        H = np.zeros([5,4])
        H[3][3] = 1

        T = np.zeros([14,14])
        T[0:3,0:4] = A
        T[0:3,4:10] = B
        T[3:6,4:10] = C
        T[3:6,10:14] = D
        T[6:9,10:14] = E
        T[9:14,0:4] = F
        T[9:14,4:10] = G
        T[9:14,10:14] = H

        t1 = np.linspace(0, t1, 30)
        t2 = np.linspace(0, t2, 30)
        t3 = np.linspace(0, t3, 30)
        return t1,t2,t3,T

    def assess_feasibility(self,particle_x,theta,Vmax,Amax):
        '''
        ACTION:

        TO CHECK IF THE SOLUTION IS FEASIBLE OR NOT

        :param particle_x: 每一个nest的位置
        :param theta:       需要经过的路径点,因为4x6的数组
        :param Vm:      各个关节的最大速度，应为1x6的数组
        :param Am:       各个关节的最大加速度，应为1x6的数组
        :return: feasibility 是否满足约束条件，若满足，则返回True，否则返回False
        '''

        t1,t2,t3,T = self.cal_T(particle_x.copy())
        theta = theta.copy().T
        Vm = Vmax.copy()
        Am = Amax.copy()        # 都写成copy了，防止在函数内修改原数组
        Trans2 = np.array([[2, 0, 0], [0, 1, 0]])
        Trans3 = np.array([[3, 0, 0, 0], [0, 2, 0, 0], [0, 0, 1, 0]])
        Trans4 = np.array([[4, 0, 0, 0, 0], [0, 3, 0, 0, 0], [0, 0, 2, 0, 0], [0, 0, 0, 1, 0]])
        Trans5 = np.array(
            [[5, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0], [0, 0, 3, 0, 0, 0], [0, 0, 0, 2, 0, 0], [0, 0, 0, 0, 1, 0]])

        for i in range(len(Vm)):
            thetai = np.array([0, 0, 0, 0, 0, 0, theta[i][3], 0, 0, theta[i][0], 0, 0, theta[i][2], theta[i][1]])
            coeffs = np.linalg.solve(T, thetai.T)
            a1 = coeffs[0:4]
            a2 = coeffs[4:10]
            a3 = coeffs[10:14]

            aa1 = Trans3 @ a1
            aa2 = Trans5 @ a2
            aa3 = Trans3 @ a3

            aaa1 = Trans2 @ aa1
            aaa2 = Trans4 @ aa2
            aaa3 = Trans2 @ aa3

            y1 = np.polyval(aa1, t1)
            y2 = np.polyval(aa2, t2)
            y3 = np.polyval(aa3, t3)

            yy1 = np.polyval(aaa1, t1)
            yy2 = np.polyval(aaa2, t2)
            yy3 = np.polyval(aaa3, t3)

            cond = np.all((abs(y1)<=Vm[i])& (abs(y2)<=Vm[i]) & (abs(y3)<=Vm[i])& (abs(yy1)<=Am[i]) & (abs(yy2)<=Am[i]) & (abs(yy3)<=Am[i]))
            if not cond:
                return False
        return True



    def optimum(self, best, particle_x):

        '''

        PARAMETERS:

        best: GLOBAL BEST SOLUTION 'best'

        particle_x: PARTICLE POSITION

        ACTION:

        COMPARE PARTICLE'S CURRENT POSITION WITH GLOBAL BEST POSITION
        
            1. IF PROBLEM IS MINIMIZATION (min=TRUE), THEN CHECKS WHETHER FITNESS VALUE OF 'best'

            IS LESS THAN THE FITNESS VALUE OF 'particle_x' AND IF IT IS GREATER, THEN IT

            SUBSTITUTES THE CURRENT PARTICLE POSITION AS THE BEST (GLOBAL) SOLUTION
            
            2. IF PROBLEM IS MAXIMIZATION (min=FALSE), THEN CHECKS WHETHER FITNESS VALUE OF 'best'

            IS GREATER THAN THE FITNESS VALUE OF 'particle_x' AND IF IT IS LESS, THEN IT

            SUBSTITUTES THE CURRENT PARTICLE POSITION AS THE BEST (GLOBAL) SOLUTION
        
        '''
        if self.min:
            if self.fitness(best) > self.fitness(particle_x):
                best = particle_x.copy()
        else:
            if self.fitness(best) < self.fitness(particle_x):
                best = particle_x.copy()
        return best

    def clip_X(self):

        # IF BOUND IS SPECIFIED THEN CLIP 'X' VALUES SO THAT THEY ARE IN THE SPECIFIED RANGE

        if self.bound is not None:
            for i in range(self.n):
                xmin, xmax = self.bound[i]
                self.X[:, i] = np.clip(self.X[:, i], xmin, xmax)

    def execute(self):

        '''
        
        PARAMETERS:

        t: ITERATION NUMBER
        
        fitness_time: LIST STORING FITNESS (OR COST) VALUE FOR EACH ITERATION
        
        time: LIST STORING ITERATION NUMBER ([0,1,2,...])
        
        ACTION:
        
        AS THE NAME SUGGESTS, THIS FUNCTION EXECUTES CUCKOO SEARCH ALGORITHM
        
        BASED ON THE TYPE OF PROBLEM (MAXIMIZATION OR MINIMIZATION).

        NOTE: THIS FUNCTION PRINTS THE GLOBAL FITNESS VALUE FOR EACH ITERATION
        
        IF THE VERBOSE IS TRUE
        
        '''

        self.fitness_time, self.time = [], []

        for t in range(self.Tmax):
            self.update_position_1()
            # self.clip_X()
            self.update_position_2()
            # self.clip_X()
            self.fitness_time.append(self.fitness(self.best))
            self.time.append(t)
            if self.verbose:
                print('Iteration:  ', t, '| best global fitness (cost):', round(self.fitness(self.best), 7))

        print('\nOPTIMUM SOLUTION\n  >', np.round(self.best.reshape(-1), 7).tolist())
        print('\nOPTIMUM FITNESS\n  >', np.round(self.fitness(self.best), 7))
        print()
        if self.plot:
            self.Fplot()

    def Fplot(self):

        # PLOTS GLOBAL FITNESS (OR COST) VALUE VS ITERATION GRAPH

        plt.plot(self.time, self.fitness_time)
        plt.title('Fitness value vs Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness value')
        plt.grid()
        plt.show()

################################################# END OF CSO CLASS ######################################################################

#########################################################################################################################################
#                                                                                                                                       #
# REFERENCES:                                                                                                                           #
#                                                                                                                                       #
# [1] X. YANG AND SUASH DEB, "CUCKOO SEARCH VIA LÉVY FLIGHTS,"                                                                          #
#     2009 WORLD CONGRESS ON NATURE & BIOLOGICALLY INSPIRED COMPUTING (NABIC),                                                          #
#     2009, PP. 210-214, DOI: 10.1109/NABIC.2009.5393690.                                                                               #
#                                                                                                                                       #
# [2] RAJIB KUMAR BHATTACHARJYA, INTRODUCTION TO PARTICLE SWARM OPTIMIZATION                                                            #
#     (http://www.iitg.ac.in/rkbc/CE602/CE602/Particle%20Swarm%20Algorithms.pdf)                                                        #
#                                                                                                                                       #
#########################################################################################################################################

#####################################################   THATS ALL FOLKS!   ##############################################################
