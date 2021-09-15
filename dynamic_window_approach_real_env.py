"""

Mobile robot motion planning sample with Dynamic Window Approach

author: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı

"""

import math
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import os
import socket,threading
from create_obstacles_real import create_ob
from matplotlib.pyplot import MultipleLocator
from TCP import tcplink
from multiprocessing import Process,Pipe
show_animation = True
parent_p_TCP,child_p_TCP=Pipe()


def dwa_control(x, config, goal, ob):
    """
    Dynamic Window Approach control
    """

    dw = calc_dynamic_window(x, config)

    u, trajectory = calc_final_input(x, dw, config, goal, ob)

    return u, trajectory


class RobotType(Enum):
    circle = 0
    rectangle = 1


class Config:
    """
    simulation parameter class
    """

    def __init__(self):
        # robot parameter
        self.max_speed = 5.0  # [m/s] 最大速度
        self.min_speed = 0.2  # [m/s] 最小速度，负数表示可以倒车
        self.max_yawrate = 20.0 * math.pi / 180.0  # [rad/s] 最大角速度
        self.max_accel = 0.1  # [m/ss] 最大加速度
        self.max_dyawrate = 40.0 * math.pi / 180.0  # [rad/ss] 最大角加速度
        self.v_reso = 0.01  # [m/s]E: 速度分辨率
        self.yawrate_reso = 0.1 * math.pi / 180.0  # [rad/s] 角速度分辨率
        self.dt = 0.1  # [s] Time tick for motion prediction 采样周期
        self.predict_time = 2.0  # [s]  向前预测时间
        self.to_goal_cost_gain = 0.5 #目标代价增益
        self.speed_cost_gain = 1.0 #速度代价增益
        self.obstacle_cost_gain = 3.0 #障碍代价增益
        self.robot_type = RobotType.rectangle

        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = 1.0  # [m] for collision check

        # if robot_type == RobotType.rectangle
        self.robot_width = 1.8  # [m] for collision check
        self.robot_length = 2.0  # [m] for collision check

    @property
    def robot_type(self):
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value


def motion(x, u, dt):
    """
    motion model
    """

    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]

    return x


def calc_dynamic_window(x, config):
    """
    calculation dynamic window based on current state x
    """

    # Dynamic window from robot specification
    Vs = [config.min_speed, config.max_speed,
          -config.max_yawrate, config.max_yawrate]

    # Dynamic window from motion model
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_dyawrate * config.dt,
          x[4] + config.max_dyawrate * config.dt]

    #  [vmin,vmax, yaw_rate min, yaw_rate max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw


def predict_trajectory(x_init, v, y, config):
    """
    predict trajectory with an input
    """

    x = np.array(x_init)
    traj = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        traj = np.vstack((traj, x))
        time += config.dt

    return traj


def calc_final_input(x, dw, config, goal, ob):
    """
    calculation final input with dynamic window
    """

    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])

    # evaluate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1], config.v_reso):
        for y in np.arange(dw[2], dw[3], config.yawrate_reso):

            trajectory = predict_trajectory(x_init, v, y, config)

            # calc cost
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
            speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
            ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, ob, config)

            final_cost = to_goal_cost + speed_cost + ob_cost

            # search minimum trajectory
            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v, y]
                best_trajectory = trajectory

    return best_u, best_trajectory


def calc_obstacle_cost(trajectory, ob, config):
    """
        calc obstacle cost inf: collision
    """
    ox = ob[:, 0]
    oy = ob[:, 1]
    dx = trajectory[:, 0] - ox[:, None]
    dy = trajectory[:, 1] - oy[:, None]
    r = np.sqrt(np.square(dx) + np.square(dy))

    if config.robot_type == RobotType.rectangle:
        yaw = trajectory[:, 2]
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        rot = np.transpose(rot, [2, 0, 1])
        local_ob = ob[:, None] - trajectory[:, 0:2]
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        local_ob = np.array([local_ob @ x for x in rot])
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        upper_check = local_ob[:, 0] <= config.robot_length / 2
        right_check = local_ob[:, 1] <= config.robot_width / 2
        bottom_check = local_ob[:, 0] >= -config.robot_length / 2
        left_check = local_ob[:, 1] >= -config.robot_width / 2
        if (np.logical_and(np.logical_and(upper_check, right_check),
                           np.logical_and(bottom_check, left_check))).any():
            return float("Inf")
    elif config.robot_type == RobotType.circle:
        if (r <= config.robot_radius).any():
            return float("Inf")

    min_r = np.min(r)
    return 1.0 / min_r  # OK


def calc_to_goal_cost(trajectory, goal):
    """
        calc to goal cost with angle difference
    """

    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - trajectory[-1, 2]
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

    return cost


def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)


def plot_robot(x, y, yaw, config):  # pragma: no cover
    if config.robot_type == RobotType.rectangle:
        outline = np.array([[-config.robot_length / 2, config.robot_length / 2,
                             (config.robot_length / 2), -config.robot_length / 2,
                             -config.robot_length / 2],
                            [config.robot_width / 2, config.robot_width / 2,
                             - config.robot_width / 2, -config.robot_width / 2,
                             config.robot_width / 2]])
        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                         [-math.sin(yaw), math.cos(yaw)]])
        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y
        plt.plot(np.array(outline[0, :]).flatten(),
                 np.array(outline[1, :]).flatten(), "-k")
    elif config.robot_type == RobotType.circle:
        circle = plt.Circle((x, y), config.robot_radius, color="b")
        plt.gcf().gca().add_artist(circle)
        out_x, out_y = (np.array([x, y]) +
                        np.array([np.cos(yaw), np.sin(yaw)]) * config.robot_radius)
        plt.plot([x, out_x], [y, out_y], "-k")

def connect():
    # TCP连接下位机
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1',9999))
    s.listen(5)
    print('Waiting for conneting')
    sock, addr = s.accept()
    t = threading.Thread(target=tcplink, args=(sock, addr, child_p_TCP))
    t.start()
    #parent_p_TCP.send(b'00')


def main(gx=1.5, gy=5.5, robot_type=RobotType.circle):
    env='real'#'simulation'
    print(__file__ + " start!!")
    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    x = np.array([5, 5, math.pi / 34.0, 0.2, 0.0])
    # goal position [x(m), y(m)]
    #goal = np.array([gx, gy])
    #goal_list=[[1.5,5.5],[2.5,6.5],[2.5,8.5],[4.5,10.5],[7.5,13.5],[6.5,16.5],[15.5,25.5],[26.5,25.5],[29.5,28.5],[29.5,29.5]]
    goal_list=[[5, 5], [9, 5], [12, 8], [13, 8], [14, 7], [15, 6], [20, 5]]
    obstacle2 = [(2.5, 2.5), (3.5, 2.5), (4.5, 2.5), (5.5, 2.5)]
    obstacle3 = [(10.5, 10.5), (10.5, 11.5), (10.5, 12.5), (10.5, 13.5), (10.5, 14.5), (10.5, 15.5), (10.5, 16.5),
                 (10.5, 17.5), (10.5, 18.5), (11.5, 18.5), (12.5, 18.5), (13.5, 18.5), (14.5, 18.5), (15.5, 18.5),
                 (16.5, 18.5), (17.5, 18.5), (18.5, 18.5), (18.5, 17.5), (18.5, 16.5), (18.5, 15.5), (18.5, 14.5),
                 (18.5, 13.5), (18.5, 12.5), (18.5, 11.5), (18.5, 10.5)]
    obstacle4_isolated = [(26.5, 9.5), (9.5, 22.5), (0.5, 4.5), (3.5, 2.5), (13.5, 26.5), (14.5, 6.5), (3.5, 12.5), (15.5, 2.5), (24.5, 2.5), (0.5, 22.5), (7.5, 23.5), (20.5, 11.5), (7.5, 16.5), (3.5, 5.5), (13.5, 9.5), (15.5, 5.5), (27.5, 12.5), (25.5, 12.5), (5.5, 7.5), (9.5, 16.5), (15.5, 13.5), (12.5, 19.5), (15.5, 28.5), (23.5, 28.5), (13.5, 24.5), (26.5, 29.5), (19.5, 9.5), (2.5, 27.5), (5.5, 5.5), (15.5, 21.5), (14.5, 0.5), (11.5, 5.5), (5.5, 9.5), (1.5, 6.5), (21.5, 22.5), (2.5, 3.5), (10.5, 10.5), (28.5, 12.5), (1.5, 18.5), (24.5, 11.5), (10.5, 22.5), (14.5, 27.5), (14.5, 20.5), (18.5, 2.5), (17.5, 5.5), (10.5, 17.5), (27.5, 28.5), (4.5, 24.5), (23.5, 8.5), (17.5, 3.5), (1.5, 6.5), (22.5, 17.5), (16.5, 26.5), (0.5, 21.5), (0.5, 19.5), (28.5, 26.5), (21.5, 23.5), (11.5, 15.5), (3.5, 6.5), (4.5, 9.5), (20.5, 18.5), (14.5, 12.5), (27.5, 23.5), (2.5, 28.5), (28.5, 1.5), (1.5, 11.5), (27.5, 29.5), (10.5, 22.5), (19.5, 1.5), (2.5, 4.5), (15.5, 18.5), (7.5, 10.5), (11.5, 29.5), (4.5, 20.5), (14.5, 19.5), (29.5, 4.5), (20.5, 17.5), (8.5, 5.5), (9.5, 11.5), (24.5, 18.5)]
    ob_real=[(11,6),(11,5),(11,4),(12,4),(13,4),(14,4),(14,5),(13,5),(12,5)]

    Map_real_corners_plot = [(0.5, 0.5), (0.5, 6.5), (9.5, 6.5), (9.5, 8.5), (28.5, 8.5), (28.5, 5.5), (29.5, 5.5),
                        (29.5, -2.5), (21.5, -2.5), (21.5, 0.5), (0.5, 0.5)]

    # obstacles [x(m) y(m), ....]
    '''
    ob = np.array([[-1, -1],
                   [0, 2],
                   [4.0, 2.0],
                   [5.0, 4.0],
                   [5.0, 5.0],
                   [5.0, 6.0],
                   [5.0, 9.0],
                   [8.0, 9.0],
                   [7.0, 9.0],
                   [8.0, 10.0],
                   [9.0, 11.0],
                   [12.0, 13.0],
                   [12.0, 12.0],
                   [15.0, 15.0],
                   [13.0, 13.0]
                   ])
    '''
    ob=create_ob()
    # input [forward speed, yaw_rate]
    config = Config()
    config.robot_type = robot_type
    trajectory = np.array(x)
    i=0;
    connect()
    pre = parent_p_TCP.recv()
    print(' demo pre is:', pre.decode('utf-8'))
    x_distance=5
    y_distance=5
    while True:
        goal=np.array(goal_list[i])
        u, predicted_trajectory = dwa_control(x, config, goal, ob)
        x = motion(x, u, config.dt)  # simulate robot
        if abs(x[2])<=0.1:
            parent_p_TCP.send(b'04004550')
            print('前进')
            x_distance=x_distance+0.02

        elif abs(x[2])>=0.1 and x[2]>0:
            a1 = str(20 + abs(int(100 * x[4])))
            b_str1='0200'+a1+'50'
            parent_p_TCP.send(bytes(b_str1,encoding='utf-8'))
            print('左转')
            x_distance = x_distance + int(a1)*math.cos(abs(x[2]))*0.0005
            y_distance=y_distance+int(a1)*math.sin(abs(x[2]))*0.0005
        elif abs(x[2])>=0.1  and x[2]<0:
            a2=str(20 + abs(int(100 * x[4])))
            b_str2 = '0'+a2+'020'+ '50'
            parent_p_TCP.send(bytes(b_str2,encoding='utf-8'))
            print('右转')
            x_distance = x_distance + int(a2)*math.cos(abs(x[2]))*0.0005
            y_distance = y_distance - int(a2)*math.sin(abs(x[2]))*0.0005


        elif x[0]==goal_list[-1][0]:
            parent_p_TCP.send(b'00000000')
            print ('到达终点')
        print('x_distance: '+str(x_distance),'y_distance: '+str(y_distance))



        print(x)

        trajectory = np.vstack((trajectory, x))  # store state history

        if show_animation:
            '''
            plt.cla()
            plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
            plt.plot(x[0], x[1], "xr")         
            plt.plot(goal[0], goal[1], "xb")
            plt.plot(ob[:, 0], ob[:, 1], "ok")
            plot_robot(x[0], x[1], x[2], config)
            plot_arrow(x[0], x[1], x[2])
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)
            '''
            plt.cla()
            ax1 = plt.gca()
            if env=='simulation':
                for point in obstacle2:
                    rect = plt.Rectangle((point[0] - 0.5, point[1] - 0.5), 1, 1, color='k')
                    ax1.add_patch(rect)
                for point in obstacle3:
                    rect = plt.Rectangle((point[0] - 0.5, point[1] - 0.5), 1, 1, color='k')
                    ax1.add_patch(rect)
                for point in obstacle4_isolated:
                    rect = plt.Rectangle((point[0] - 0.5, point[1] - 0.5), 1, 1, color='k')
                    ax1.add_patch(rect)
                plt.xlim(0, 31)
                plt.ylim(0, 31)
                spacing = 1
                minorLocator = MultipleLocator(spacing)
                ax1.xaxis.set_minor_locator(minorLocator)
                ax1.yaxis.set_minor_locator(minorLocator)
                ax1.grid(which='minor')
                ax1.grid()
                # plt.axis("equal")
                plt.grid(True)


            if env=='real':
                '''画出场地边界'''
                plt.plot([v[0] for v in Map_real_corners_plot], [v[1] for v in Map_real_corners_plot], color='k')
                for point in ob_real:
                    rect = plt.Rectangle((point[0] - 0.5, point[1] - 0.5), 1, 1, color='k')
                    ax1.add_patch(rect)
                plt.xlim(0, 30)
                plt.ylim(-3, 9)
                '''画竖直网格'''
                V_corners = [[0.5, 0.5, 9.0, 6.5], [9.5, 0.5, 21.0, 8.5], [21.5, -2.5, 29.0, 8.5],
                             [29.5, -2.5, 30.5, 5.5]]
                for rect in V_corners:
                    X_grid1 = list(np.arange(rect[0], rect[2], 1.0))
                    Y_grid1 = [rect[1], rect[3]]
                    for X1 in X_grid1:
                        plt.plot([i for i in [X1, X1]], [j for j in Y_grid1], color='grey', linewidth=0.5)
                '''画水平网格'''
                H_corners = [[21.5, -2.5, 29.5, 0.0], [0.5, 0.5, 29.5, 6.0], [0.5, 6.5, 28.5, 7.0],
                             [9.5, 7.5, 28.5, 9.0]]
                for rect in H_corners:
                    X_grid1 = [rect[0], rect[2]]
                    Y_grid1 = list(np.arange(rect[1], rect[3], 1.0))
                    for Y1 in Y_grid1:
                        plt.plot([i for i in X_grid1], [j for j in [Y1, Y1]], color='grey', linewidth=0.5)

            plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
            plt.plot(x[0], x[1], "xr")
            plt.plot(goal[0], goal[1], "xb")
            #plt.plot(ob[:, 0], ob[:, 1], "ok")
            plot_robot(x[0], x[1], x[2], config)
            plot_arrow(x[0], x[1], x[2])

            plt.pause(0.0001)

        # check reaching goal
        dist_to_goal = math.sqrt((x[0] - goal[0]) ** 2 + (x[1] - goal[1]) ** 2)
        if dist_to_goal <= config.robot_radius:
            print("{}th Goal reached!!".format(i))
            i=i+1
        if dist_to_goal <= config.robot_radius and i==len(goal_list):
            print("final Goal reached!!")
            break
    print("Done")
    if show_animation:
        plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")
        for point in goal_list:
            plt.plot(point[0],point[1],'xb')
        plt.pause(0.0001)

    path='./result_real/'
    dir_list=os.listdir(path)
    count=len(dir_list)
    img_name=str(count)+'.png'
    plt.savefig(path+img_name, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main(robot_type=RobotType.rectangle)
