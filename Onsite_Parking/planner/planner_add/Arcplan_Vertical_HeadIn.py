import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.kdtree as kd
import input.make_map as mp
from input import make_car
import math
from math import *

'''
垂直车位车头泊入代码
'''

C = make_car.C
def Arcplan_Vertical_HeadIn_tangent_circle(gx,gy,gyaw,P,mod):
    gyaw = angle_limit_pi2pi(gyaw)
    End = [gx,gy,gyaw]
    # 定义泊出的转弯半径，要超过汽车的最小转弯半径
    Rr = 2300
    # 起始点为车位的停车点
    x_now = End[0]
    y_now = End[1]
    theta_now = End[2]
    # 设置六分之pi的范围，并在这个范围内均匀取点
    Alpha0 = 0
    Alpha1 = np.pi / 6
    alpha = np.linspace(Alpha0, Alpha1, 100)
    # 定义坐标点及航向角的序列
    x3 = np.zeros(len(alpha))
    y3 = np.zeros(len(alpha))
    theta3 = np.zeros(len(alpha))
    # 定义每个序列的初始值为车位的坐标点和航向角
    x3[0] = x_now
    y3[0] = y_now
    theta3[0] = theta_now
    # 假设最终100个点都能取到
    num = 100

    for i in range(1, len(alpha)):
        # mod为2是一开始顺时针倒车，然后重新计算坐标和航向角序列中的值
        if mod == 2:
            x3[i] = x_now + Rr * (np.cos(alpha[0]) - np.cos(alpha[i]))
            y3[i] = y_now + Rr * (np.sin(alpha[i]) - np.sin(alpha[0]))
            theta3[i] = -alpha[i] - np.pi / 2
        # mod为1是一开始逆时针倒车，然后重新计算坐标和航向角序列中的值
        elif mod == 1:
            x3[i] = x_now - Rr * (np.cos(alpha[0]) - np.cos(alpha[i]))
            y3[i] = y_now + Rr * (np.sin(alpha[i]) - np.sin(alpha[0]))
            theta3[i] = - (- alpha[i] + np.pi / 2)
        # 路径旋转，即坐标系转换，这样一套代码就可以运用在不同角度的车位中
        x3[i], y3[i], theta3[i] = path_rotate(x3[i], y3[i], theta3[i], gx, gy, gyaw)
        # 进行碰撞检测，找到碰撞检测时的点
        if i > 2:
            isCollision = is_collision_rectangle_3d(x3[i], y3[i], theta3[i], P)
            if isCollision or theta3[i] == -np.pi or theta3[i] == 0:
                num = i
                break
            else:
                num = i
    
    ptIntvl = 10
    # 只取碰撞前的点
    x1 = x3[num-1]
    y1 = y3[num-1]
    theta1 = theta3[num-1]
    # 这里的一系列计算就是进行插值，让曲线更平滑一些
    Delta_alpha = theta_now - theta1
    Ls = abs(Rr * Delta_alpha)
    L = np.linspace(0, Ls, num)
    point_num = int(L[num-1] / ptIntvl) + 2
    L3i = np.zeros(point_num)
    for i in range(1, point_num - 1):
        L3i[i] = (i - 1) * ptIntvl
    L3i[point_num - 1] = L[num-1]
    x = np.interp(L3i, L, x3[:num])
    y = np.interp(L3i, L, y3[:num])
    theta = np.interp(L3i, L, theta3[:num])
    # 定义曲率序列
    K = np.ones(point_num) / Rr
    # 定义方向序列（向前行驶为1，向后行驶为-1，这里从车位向外泊出，将来要将所有点倒置，虽然这里是倒车，仍然全部取1）
    D = np.ones(point_num) * 1
    # 将各序列堆叠起来
    path = np.vstack((x, y, theta, D, K))
    print(f"第1次重规划生成{len(path[0])}个节点")

    # 出车位后沿着相切点按照最小转弯半径揉库，直到车辆角度大致与车位角度垂直，一般车位揉库一次到两次便可成功
    Direction = 1
    for i in range(1, 13):
        if (Direction < 0 and (i + 1) % 2 == 0) or (Direction > 0 and i % 2 == 0):
            isok, theta1, x1, y1, path = Backward_program(theta1, x1, y1, path, P, gyaw)
            if isok:
                print(f"重规划{i+1}次，Backward success!")
                break
        elif (Direction > 0 and (i + 1) % 2 == 0) or (Direction < 0 and i % 2 == 0):
            isok, theta1, x1, y1, path = Forward_program(theta1, x1, y1, path, P, gyaw)
            if isok:
                print(i + 1)
                print(f"重规划{i+1}次，Forward success!")
                break
    return path


def Backward_program(theta1, x_now, y_now, path, P, gyaw):
    # 512是该车辆的最小转弯半径
    Rr = 512
    Alpha0 = np.pi + theta1
    Alpha1 = 0

    alpha = np.linspace(Alpha0, Alpha1, 150)
    x3 = np.zeros(len(alpha))
    y3 = np.zeros(len(alpha))
    theta3 = np.zeros(len(alpha))
    x3[0] = x_now
    y3[0] = y_now
    theta3[0] = theta1
    isok = False
    num = 100

    for i in range(1, len(alpha)):
        x3[i] = x_now + Rr * (np.sin(alpha[0]) - np.sin(alpha[i]))
        y3[i] = y_now + Rr * (np.cos(alpha[i]) - np.cos(alpha[0]))
        theta3[i] = alpha[i] - np.pi
        len_arc = Rr * (theta1 - theta3[i])
        pVec = [x3[i], y3[i], theta3[i]]
        if len_arc >= 65:
            isCollision = is_collision_rectangle_3d(x3[i], y3[i], theta3[i], P)
            if isCollision or abs(theta3[i] - (gyaw - np.pi/2)) <= 0.1745 or abs(theta3[i] - (gyaw + np.pi/2)) <= 0.1745:
                num = i
                break

    x2, y2, theta2 = x3[num], y3[num], theta3[num]
    if abs(theta2 - (gyaw - np.pi/2)) <= 0.1745 or abs(theta2 - (gyaw + np.pi/2)) <= 0.1745:
        isok = True

    ptIntvl = 10
    Delta_alpha = theta1 - theta2
    Ls = abs(Rr * Delta_alpha)
    L = np.linspace(0, Ls, num)
    point_num = int(L[num-1] / ptIntvl) + 2
    L3i = np.zeros(point_num)
    for i in range(point_num - 1):
        L3i[i] = (i - 1) * ptIntvl
    L3i[point_num - 1] = L[num-1]
    x_interp = np.interp(L3i, L, x3[:num])
    y_interp = np.interp(L3i, L, y3[:num])
    theta_interp = np.interp(L3i, L, theta3[:num])
    K = np.zeros(point_num) + 1 / Rr
    D = np.zeros(point_num) + 1
    path_backward = np.vstack((x_interp, y_interp, theta_interp, D, K))
    path = np.hstack((path, path_backward))

    return isok, theta2, x2, y2, path

def Forward_program(theta1, x_now, y_now, path, P, gyaw):
    Rl = 512
    Alpha0 = np.pi + theta1
    Alpha1 = 0

    alpha = np.linspace(Alpha0, Alpha1, 150)
    x3 = np.zeros(len(alpha))
    y3 = np.zeros(len(alpha))
    theta3 = np.zeros(len(alpha))
    x3[0] = x_now
    y3[0] = y_now
    theta3[0] = theta1
    isok = False
    num = 100

    for i in range(1, len(alpha)):
        x3[i] = x_now - Rl * (np.sin(alpha[0]) - np.sin(alpha[i]))
        y3[i] = y_now - Rl * (np.cos(alpha[i]) - np.cos(alpha[0]))
        theta3[i] = alpha[i] - np.pi
        len_arc = Rl * (theta1 - theta3[i])
        pVec = [x3[i], y3[i], theta3[i]]
        if len_arc >= 10:
            isCollision = is_collision_rectangle_3d(x3[i], y3[i], theta3[i], P)
            #isCollision = is_collision_3circle_3d(pVec[0],pVec[1],pVec[2],P)
            if isCollision == 1 or abs(theta3[i] - (gyaw - np.pi/2)) <= 0.1745 or abs(theta3[i] - (gyaw + np.pi/2)) <= 0.1745:
                num = i
                break

    x2, y2, theta2 = x3[num], y3[num], theta3[num]
    if abs(theta3[i] - (gyaw - np.pi/2)) <= 0.1745 or abs(theta3[i] - (gyaw + np.pi/2)) <= 0.1745:
        isok = True

    ptIntvl = 10
    Delta_alpha = theta1 - theta2
    Ls = abs(Rl * Delta_alpha)
    L = np.linspace(0, Ls, num)
    point_num = int(L[num-1] / ptIntvl) + 2
    L3i = np.zeros(point_num)
    for i in range(point_num - 1):
        L3i[i] = (i - 1) * ptIntvl
    L3i[point_num - 1] = L[num-1]
    x_interp = np.interp(L3i, L, x3[:num])
    y_interp = np.interp(L3i, L, y3[:num])
    theta_interp = np.interp(L3i, L, theta3[:num])
    K = np.zeros(point_num) - 1 / Rl
    D = np.zeros(point_num) - 1
    path_forward = np.vstack((x_interp, y_interp, theta_interp, D, K))
    path = np.hstack((path, path_forward))

    return isok, theta2, x2, y2, path

def path_rotate(x, y, yaw, gx, gy, gyaw):
    x_new = (x - gx) * np.cos(gyaw + np.pi / 2) - (y - gy) * np.sin(gyaw + np.pi / 2) + gx
    y_new = (x - gx) * np.sin(gyaw + np.pi / 2) + (y - gy) * np.cos(gyaw + np.pi / 2) + gy
    yaw_new = yaw + (gyaw + np.pi / 2)
    return x_new, y_new, yaw_new


def angle_limit_pi2pi(angle):
    if angle > np.pi:
        angle = angle - 2 * np.pi
    elif angle < - np.pi :
        angle = angle + 2 * np.pi
    return angle

# 碰撞检测
def is_collision_rectangle_3d(rx, ry, yaw, P):
    '''
    三圆检测模型，后悬中心185，中心175，前悬中心185
    :param x: Vehicle's rear center in the world coordinate
    :param y: Vehicle's rear center in the world coordinate
    :param yaw: Vehicle's rear center in the world coordinate
    :param P: class P
    :return: True or False
    '''
    satety_threshold = 10

    RC = (C.RB + C.RF) / 2 - C.RB  # 车辆后轴中心到车辆中心的距离
    cx = rx + RC * math.cos(yaw)  # 车辆中心（cx，cy）
    cy = ry + RC * math.sin(yaw)

    # 查找一定范围内的障碍物点
    search_area = C.SEARCH_RADIUS
    ids = P.kdtree.query_ball_point([cx, cy], search_area)

    for i in ids:
        obsx = P.ox[i]
        obsy = P.oy[i]
        dx = cx - obsx
        dy = obsy - cy
        ddx = dx * np.cos(yaw) - dy * np.sin(yaw)
        ddy = dx * np.sin(yaw) + dy * np.cos(yaw)

        if abs(ddx) < (C.RB + C.RF) / 2 + satety_threshold and abs(ddy) < C.W / 2 + satety_threshold:
            return True
    return False