import sys
import os
import time
import numpy as np
import input.make_map as mp
from input import make_car
import matplotlib.pyplot as plt
import Arcplan_Vertical_HeadIn as Arcplan1
import scipy.spatial.kdtree as kd

'''绘制泊车路径及车辆'''

class Para:
    def __init__(self, minx, miny, minyaw, maxx, maxy, maxyaw,
                 xnum, ynum, yawnum, xyreso, yawreso, ox, oy, kdtree):
        self.minx = minx
        self.miny = miny
        self.minyaw = minyaw
        self.maxx = maxx
        self.maxy = maxy
        self.maxyaw = maxyaw
        self.xnum = xnum            # grid’s number of x
        self.ynum = ynum            # grid's number of y
        self.yawnum = yawnum        # grid's number of yaw
        self.xyreso = xyreso
        self.yawreso = yawreso
        self.ox = ox                # x of obstacles in the world coordinate
        self.oy = oy                # y of obstacles in the world coordinate
        self.kdtree = kdtree


C = make_car.C
def Car_plot(x, y, fai, c):
    Length_Car = C.RB + C.RF
    Width_Car = C.W
    Hx_Car = C.RB
    Qx_Car = C.RF

    x_r1 = x - Hx_Car * np.cos(fai) - Width_Car / 2 * np.sin(fai)
    y_r1 = y - Hx_Car * np.sin(fai) + Width_Car / 2 * np.cos(fai)

    x_r2 = x - Hx_Car * np.cos(fai) + Width_Car / 2 * np.sin(fai)
    y_r2 = y - Hx_Car * np.sin(fai) - Width_Car / 2 * np.cos(fai)

    x_f1 = x + Qx_Car * np.cos(fai) - Width_Car / 2 * np.sin(fai)
    y_f1 = y + Qx_Car * np.sin(fai) + Width_Car / 2 * np.cos(fai)

    x_f2 = x + Qx_Car * np.cos(fai) + Width_Car / 2 * np.sin(fai)
    y_f2 = y + Qx_Car * np.sin(fai) - Width_Car / 2 * np.cos(fai)

    plt.plot([x_r1, x_r2], [y_r1, y_r2], c, linewidth=1)
    plt.plot([x_f1, x_r1], [y_f1, y_r1], c, linewidth=1)
    plt.plot([x_f1, x_f2], [y_f1, y_f2], c, linewidth=1)
    plt.plot([x_f2, x_r2], [y_f2, y_r2], c, linewidth=1)


def main():
    sys.setrecursionlimit(2000)
    # 输入input文件夹下场景文件
    map_path = 'C:/Users/86188/Desktop/onsite第三赛道比赛代码暑假/20240529/Onsite_Parking/input/B01.json'

    ox, oy,sp,gp = mp.make_map(map_path)
    sx, sy, syaw0 = sp['x'], sp['y'], sp['yaw'] # 7600，200，CC.PI/2

    # 获取目标停车位
    park = '15'
    gx, gy, gyaw0 = gp[park]['x_end'], gp[park]['y_end'], gp[park]['yaw']
    outx, outy = gp[park]['x_out'], gp[park]['y_out']
    vertex = gp[park]['pos']

    # Store Obstacles' index (ox,oy) in KDTree to make it easy to find the nearest obstacles
    kdtree = kd.KDTree([[x, y] for x, y in zip(ox, oy)])

    minx = round(min(ox)/C.XY_RESO)
    miny = round(min(oy)/C.XY_RESO)
    maxx = round(max(ox)/C.XY_RESO)
    maxy = round(max(oy)/C.XY_RESO)
    xnum, ynum = maxx - minx + 1, maxy - miny + 1
    minyaw = round(0 / C.YAW_RESO)
    maxyaw = round(2 * C.PI / C.YAW_RESO)
    yawnum = maxyaw - minyaw + 1

    P = Para(minx, miny, minyaw, maxx, maxy, maxyaw,xnum, ynum, yawnum, C.XY_RESO, C.YAW_RESO, ox, oy, kdtree)
    print("Para calculating success")

    # 泊车重规划算法
    path = Arcplan1.Arcplan_Vertical_HeadIn_tangent_circle(gx, gy, gyaw0, P, 2)

    plt.figure(figsize=(8000, 8000))
    plt.scatter(ox, oy, color='black', marker='o', s=1)  # 绘制障碍物
    plt.scatter(sx, sy, color='black', marker='o', s=10)  # 绘制起点位置
    plt.text(sx, sy, f'startpoint', color='black', fontsize=12)  # 起点标签
    plt.scatter(path[0], path[1], marker='o', s=1)
    for i in range(0, len(path[0]), 5):
        Car_plot(path[0][i], path[1][i], path[2][i], 'c')
        plt.pause(0.1)
    Car_plot(path[0][-1], path[1][-1], path[2][-1], 'c')
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    main()
