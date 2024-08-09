import numpy as np
from model.F_Kine import cal_eepose
import plotly.graph_objects as go
import pandas as pd

data1 = pd.read_csv('data/pos1.csv')
pos1 = [data1['X'].values, data1['Y'].values, data1['Z'].values, data1['RX'].values, data1['RY'].values, data1['RZ'].values]
pos1 = np.array(pos1).T

data2 = pd.read_csv('data/pos2.csv')
pos2 = [data2['X'].values, data2['Y'].values, data2['Z'].values, data2['RX'].values, data2['RY'].values, data2['RZ'].values]
pos2 = np.array(pos2).T

data3 = pd.read_csv('data/data.csv')
J3 = [data3['J1'].values, data3['J2'].values, data3['J3'].values, data3['J4'].values, data3['J5'].values, data3['J6'].values]
J3 = np.array(J3).T
pos3 = np.zeros(J3.shape)
for i in range(len(pos3)):
    pos3[i] = cal_eepose(J3[i])


# 绘制工作空间
# 使用你的数据
x1, y1, z1 = pos1[:, 0], pos1[:, 1], pos1[:, 2]
x2, y2, z2 = pos2[:, 0], pos2[:, 1], pos2[:, 2]
x3, y3, z3 = pos3[:, 0], pos3[:, 1], pos3[:, 2]
# 计算x, y, z方向的最小值和最大值
x_min, x_max = np.min(x1), np.max(x1)
y_min, y_max = np.min(y1), np.max(y1)
z_min, z_max = np.min(z1), np.max(z1)

# 创建一个3D散点图
scatter = go.Scatter3d(
    x=x1,
    y=y1,
    z=z1,
    mode='markers',
    marker=dict(
        size=6,
        color=z1,  # 设置颜色为z轴的值
        colorscale='Viridis',  # 选择一种颜色映射
        opacity=0.8
    )
)
scatter2 = go.Scatter3d(
    x=x2,
    y=y2,
    z=z2,
    mode='markers',
    marker=dict(
        size=6,
        color=-z2,  # 设置颜色为z轴的值
        colorscale='Viridis',  # 选择一种颜色映射
        opacity=0.8
    )
)
scatter3 = go.Scatter3d(
    x=x3,
    y=y3,
    z=z3,
    mode='markers',
    marker=dict(
        size=6,
        color=-z3,  # 设置颜色为z轴的值
        colorscale='Viridis',  # 选择一种颜色映射
        opacity=0.8
                )
)
fig = go.Figure(data=[scatter3,scatter2, scatter])

# 创建半透明的3D平面
planes = [
    go.Surface(x=[[x_min, x_max], [x_min, x_max]], y=[[y_min, y_min], [y_max, y_max]], z=[[z_min, z_min], [z_min, z_min]], showscale=False, opacity=0.2, colorscale=[(0, 'blue'), (1, 'blue')]),
    go.Surface(x=[[x_min, x_max], [x_min, x_max]], y=[[y_min, y_min], [y_max, y_max]], z=[[z_max, z_max], [z_max, z_max]], showscale=False, opacity=0.2, colorscale=[(0, 'blue'), (1, 'blue')]),
    go.Surface(y=[[y_min, y_max], [y_min, y_max]], z=[[z_min, z_min], [z_max, z_max]], x=[[x_min, x_min], [x_min, x_min]], showscale=False, opacity=0.2, colorscale=[(0, 'blue'), (1, 'blue')]),
    go.Surface(y=[[y_min, y_max], [y_min, y_max]], z=[[z_min, z_min], [z_max, z_max]], x=[[x_max, x_max], [x_max, x_max]], showscale=False, opacity=0.2, colorscale=[(0, 'blue'), (1, 'blue')]),
    go.Surface(z=[[z_min, z_max], [z_min, z_max]], x=[[x_min, x_min], [x_max, x_max]], y=[[y_min, y_min], [y_min, y_min]], showscale=False, opacity=0.2, colorscale=[(0, 'blue'), (1, 'blue')]),
    go.Surface(z=[[z_min, z_max], [z_min, z_max]], x=[[x_min, x_min], [x_max, x_max]], y=[[y_max, y_max], [y_max, y_max]], showscale=False, opacity=0.2, colorscale=[(0, 'blue'), (1, 'blue')])
]

D = np.zeros([4,6])
D[0] = [72.1412735,483.3956909,448.4580078,179.6145477,-0.566292524,45.21763611]
D[1] = [-298.7023926,396.4761047,678.6668091,179.9934692,0.003054707,45.01348877]
D[2] = [-254.5127869,-395.4613342,692.2990112,179.997406,0.004578727,44.93190765]
D[3] = [61.77036667,-480.4981384,445.8680725,179.2924194,1.417901039,45.00098801]
# 创建一个新的3D散点图，用于显示D的坐标
scatter_d = go.Scatter3d(
    x=D[:, 0],
    y=D[:, 1],
    z=D[:, 2]-2,
    mode='markers',
    marker=dict(
        size=6,
        color='red',  # 设置颜色为红色
        colorscale='Viridis',  # 选择一种颜色映射
        opacity=0.8
    )
)

# 将新的散点图添加到图中
fig.add_trace(scatter_d)

# 设置布局
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

fig.show()

