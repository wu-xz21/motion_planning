import numpy as np
import pybullet as p
from pybullet_planning import BLUE
from pybullet_planning import Pose, Point
from pybullet_planning import connect, wait_for_user, wait_for_duration
from pybullet_planning import dump_world, set_pose
from pybullet_planning import get_collision_fn, create_box
from pybullet_planning import get_disabled_collisions
from pybullet_planning import get_joint_names, get_movable_joints, set_joint_positions, plan_joint_motion
from termcolor import cprint

from utils import pso_para as pso


# 添加资源路径
connect(use_gui=True)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# _ = p.loadURDF('plane.urdf')
robot_id = p.loadURDF("urdf_files/dazu.urdf",
                      basePosition=[0, 0, 0], useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
# 定义障碍物
obstacle = create_box(0.2,0.2,0.6, color=BLUE)
obstacle_x = -0.2
obstacle_y = -0.1
obstacle_z = 0.85
set_pose(obstacle, Pose(Point(obstacle_x, obstacle_y, obstacle_z)))

# 可以使用的关节
Joint_index = get_movable_joints(robot_id)
Joint_names = get_joint_names(robot_id,Joint_index)
cprint('Joint {} \ncorresponds to:\n{}'.format(Joint_index, Joint_names), 'green')
print('#'*10)
dump_world()    # 打印关节信息
print('#'*10)

# 定义不可能碰撞的link pairs
robot_self_collision_disabled_link_names = [
        ('base_link', 'table_link'), ('table_link', 'Link1'),
        ('base_link', 'Link1'), ('base_link', 'Link2'),
        ('base_link', 'Link3'), ('base_link', 'Link2'),
        ('Link1', 'Link2'), ('Link2', 'Link3'),
        ('Link3', 'Link4'), ('Link4', 'Link5'),
        ('Link5', 'Link6'), ('Link6', 'Link4'),
        ('Link6', 'Link3'), ('Link3', 'Link5'),
        ('Link1', 'Link3'), ('Link2', 'Link4')]
self_collision_links = get_disabled_collisions(robot_id, robot_self_collision_disabled_link_names)
cprint('self_collision_links disabled: {}'.format(self_collision_links), 'yellow')
print('#'*10)
cprint('Checking ee to obstacles (w/o links) collision', 'green')
collision_fn = get_collision_fn(robot_id, Joint_index, obstacles=[obstacle],
                                self_collisions=True, attachments=[],
                                disabled_collisions=self_collision_links,
                                )
#######
# 这里遇到了一个问题，就是这个collision_fn函数无法检测到自身links之间的碰撞，
# 我暂时还没有找到解决办法，后续如果遇到问题会继续回过来解决。
######
# conf = [-0.568,-2.558, -2.368, -1.516, -1.642, 0.095]
# assert collision_fn(conf, diagnosis=True)
# print('\n')
# set_joint_positions(robot_id, Joint_index, conf)


f = 200  # 轨迹更新频率
# 6轴理论的最大值
Vm = np.array([100, 100, 150, 150, 180, 180])
Am = np.array([360, 360, 360, 360, 360, 360])
J = np.zeros([4, 6])
J[0] = [-98.266, -2.042, -105.752, -0.666, -76.064, 36.678]
J[1] = [-53.007, -1.49, -79.635, 0.002, -101.862, 81.98]
J[2] = [57.235, -4.219, -80.696, 0.002, -103.518, -167.697]
J[3] = [97.567, -2.906, -106.881, -0.722, -74.602, -127.26]
J = J*np.pi/180
for i in range(3):
    # 起始点
    q1 = J[0]
    set_joint_positions(robot_id, Joint_index, q1)
    cprint('Start configuration: {}'.format(q1), 'cyan')
    wait_for_user()

    # 终止点
    q2 = J[-1]
    cprint('End configuration: {}'.format(q2), 'cyan')
    path = plan_joint_motion(robot_id, Joint_index, q2, obstacles=[obstacle],max_distance=1e-3,
                             self_collisions=True,disabled_collisions=self_collision_links)
    if path is None:
        raise ValueError('no plan found')
        continue
    else:
        set_joint_positions(robot_id, Joint_index, q2)
        wait_for_user('a motion plan is found! Press enter to start simulating!')
    # 对path进行规划
    path = np.array(path)
    Joint_deg = path*180/np.pi
    t, y, v_all = pso.curve(Joint_deg, Am, Vm, f)  # 求解轨迹
    pso.plot(t, y, v_all)  # 画出轨迹
    wait_for_user()

    time_step = 0.5
    for conf in path:
        set_joint_positions(robot_id, Joint_index, conf)
        wait_for_duration(time_step)
    wait_for_user()
