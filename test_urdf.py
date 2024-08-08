import pybullet
import pybullet as p
import pybullet_data
import time
from pprint import pprint
import numpy as np
from pybullet_planning import create_box, set_pose, create_obj, link_from_name, get_link_pose, create_attachment, \
    set_joint_positions, dump_world
from pybullet_planning import Pose, Point, Euler, connect


# 添加资源路径
connect(use_gui=True)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
_ = p.loadURDF('plane.urdf')
robot_id = p.loadURDF("./urdf_file/dazu.urdf",
                      basePosition=[0, 0, 0], useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
ee_body = create_obj("./urdf_file/meshes/dms_bar_gripper.obj")

# 可以使用的关节
Joint_index = [i for i in range(p.getNumJoints(robot_id)) if p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED]
joint_num = len(Joint_index)

# 定义障碍物
obstacle = create_box(0.2,0.2,0.4)
obstacle_x = 0
obstacle_y = 0
obstacle_z = 0.75
set_pose(obstacle, Pose(Point(obstacle_x, obstacle_y, obstacle_z)))
print('#' * 10)
dump_world()  # 打印关节信息
print('#' * 10)

# 把夹子放到机械臂上
tool_attach_link_name = 'Link6'
tool_attach_link = link_from_name(robot_id, tool_attach_link_name)
ee_link_pose = get_link_pose(robot_id, tool_attach_link)
set_pose(ee_body, [(-0.5, -0.3999, 2.02), (0, -0.707, 0, 0.707)])
ee_attach = create_attachment(robot_id, tool_attach_link, ee_body)
ee_attach.assign()
p.changeVisualShape(robot_id, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED)


# 获取关节限制
joint_limits = []
for i in Joint_index:
    joint_info = p.getJointInfo(robot_id, i)
    joint_limits.append((joint_info[8], joint_info[9]))  # (lower limit, upper limit)

# 生成随机关节配置并计算末端执行器位置
ee_index = Joint_index[-1]  # 末端执行器链接索引
num_samples = 1000
ee_pose = []

# 各个关节的信息
Joint_tuples = [(p.getJointInfo(robot_id, i)[0], p.getJointInfo(robot_id, i)[1].decode("utf-8"))  # 0:序号 1:名称
                for i in range(p.getNumJoints(robot_id))
                if p.getJointInfo(robot_id, i)[1].decode("utf-8") != "table_joint"]

# 设置滑块调节关节位置
Joint_pos = [p.addUserDebugParameter(
    paramName=Joint_tuples[i][1] + " deg",
    rangeMin=-3,
    rangeMax=3,
    startValue=0
) for i in range(6)]
# 添加按钮控件
btn = p.addUserDebugParameter(
    paramName="reset",
    rangeMin=1,
    rangeMax=0,
    startValue=0
)
previous_btn_value = p.readUserDebugParameter(btn)

# 轨迹点
waypoints = [
    [0.1, -0.2, 1.5], [0, -0.4, 1.8],[-0.2,1,1.8]]

# 预备工作结束，重新开启渲染
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
# 关闭实时模拟步
p.setRealTimeSimulation(0)

while (1):
    p.stepSimulation()
    # 将控件的参数值作为输入控制机器人，先获取各组控件值
    degs = [p.readUserDebugParameter(param_id) for param_id in Joint_pos]
    p.setJointMotorControlArray(
        bodyUniqueId=robot_id,
        jointIndices=Joint_index,
        controlMode=p.POSITION_CONTROL,
        targetPositions=degs  # 目标位置列表
    )

    # # 遍历每个轨迹点
    # for point in waypoints:
    #     targetPos = point
    #     jointPoses = p.calculateInverseKinematics(robot_id, ee_index, targetPos)
    #     p.setJointMotorControlArray(
    #         bodyUniqueId=robot_id,
    #         jointIndices=Joint_index,
    #         controlMode=p.POSITION_CONTROL,
    #         targetPositions=jointPoses  # 目标位置列表
    #     )
    end_effector_state = p.getLinkState(robot_id, 5)[0]  # 求解当前的末端位姿
    # 检测是否reset
    if p.readUserDebugParameter(btn) != previous_btn_value:
        p.removeAllUserParameters()
        Joint_pos = [p.addUserDebugParameter(
            paramName=Joint_tuples[i][1] + " deg",
            rangeMin=-3,
            rangeMax=3,
            startValue=0
        ) for i in range(6)]
        btn = p.addUserDebugParameter(
            paramName="reset",
            rangeMin=1,
            rangeMax=0,
            startValue=0
        )

        previous_btn_value = p.readUserDebugParameter(btn)

    time.sleep(1 / 240)  # 模拟器一秒模拟迭代240步
# 断开连接
p.disconnect()

while (1):
    p.stepSimulation()
    # 遍历每个轨迹点
    for point in waypoints:
        targetPos = point
        jointPoses = p.calculateInverseKinematics(robot_id, ee_index, targetPos)
        p.setJointMotorControlArray(
            bodyUniqueId=robot_id,
            jointIndices=Joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPositions=jointPoses  # 目标位置列表
        )
    time.sleep(1 / 240)

# 断开连接
p.disconnect()
