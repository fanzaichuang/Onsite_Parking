import heapq
import math
import numpy as np
import matplotlib.pyplot as plt


# 定义node类，即节点（内有节点的x、y、代价、索引信息），不同节点的node不同
class Node:
    def __init__(self, x, y, cost, pind):
        self.x = x  # x position of node
        self.y = y  # y position of node
        self.cost = cost  # g cost of node
        self.pind = pind  # parent index of node


# 定义para类（内有地图障碍物x的最大最小值、y的最大最小值、x的最值之差、y的最值之差、网格尺寸、移动方向），相同地图的p相同
class Para:
    def __init__(self, minx, miny, maxx, maxy, xw, yw, reso, motion):
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy
        self.xw = xw
        self.yw = yw
        self.reso = reso  # resolution of grid world
        self.motion = motion  # motion set


# a星的主要运行函数
def astar_planning(sx, sy, gx, gy, ox, oy, reso, rr):
    """
    return path of A*.
    :param sx: starting node x [m]
    :param sy: starting node y [m]
    :param gx: goal node x [m]
    :param gy: goal node y [m]
    :param ox: obstacles x positions [m]
    :param oy: obstacles y positions [m]
    :param reso: xy grid resolution
    :param rr: robot radius
    :return: path
    """

    # 定义起点、终点的坐标为原起点、终点坐标除以网格尺寸，代价值为0，索引为-1（即最后一个）
    # 此处调用node类，其中round是指对括号内的数取四舍五入
    n_start = Node(round(sx / reso), round(sy / reso), 0.0, -1)
    n_goal = Node(round(gx / reso), round(gy / reso), 0.0, -1)

    # 障碍物的坐标也除以网格尺寸
    ox = [x / reso for x in ox]
    oy = [y / reso for y in oy]

    # 调用函数获取para和障碍物地图
    P, obsmap = calc_parameters(ox, oy, rr, reso)

    # 定义open_set与closed_set两个字典
    open_set, closed_set = dict(), dict()
    # 根据起点（node类）和地图的para计算索引，并将起点存入open_set该索引的地方
    open_set[calc_index(n_start, P)] = n_start

    # 定义优先序列
    q_priority = []
    # heappush会根据item的第一个值（这里是fvalue）进行优先排列（值小的排在前面）
    heapq.heappush(q_priority,
                   (fvalue(n_start, n_goal), calc_index(n_start, P)))

    # 一直进行循环
    while True:
        # 当open_set中没有元素时，结束循环（一般不会触发，因为之前已经把起点存入open_set中了）
        if not open_set:
            break

        # 返回q_priority中代价值最小的元素（这里取其索引值（ind））
        _, ind = heapq.heappop(q_priority)
        # 定义当前点（node类）为open_set中索引值为ind的点，这里就是取出了q_priority中代价值最小的点
        n_curr = open_set[ind]
        # 将这个点（node类）存入到closed_set中，索引值也为ind
        closed_set[ind] = n_curr
        # open_set字典去除该点
        open_set.pop(ind)

        # 遍历当前点上下左右及四个角共八个方向
        for i in range(len(P.motion)):
            # 定义node为node类，其坐标加上行驶的方向，代价值有所增加，索引依然为ind
            node = Node(n_curr.x + P.motion[i][0],
                        n_curr.y + P.motion[i][1],
                        n_curr.cost + u_cost(P.motion[i]), ind)

            # 如果当前点超出地图边界或是周围有障碍物，则结束当前循环，进行下一次循环
            if not check_node(node, P, obsmap):
                continue

            # 重新计算节点的索引值（索引值与点的坐标值有关）
            n_ind = calc_index(node, P)
            # 如果计算出的新的索引值并没有在closed_set中存在
            if n_ind not in closed_set:
                # 如果在open_set中存在
                if n_ind in open_set:
                    # 如果计算出的新的代价值比当前点的代价值要大，那么代价值和索引还是取当前点的
                    if open_set[n_ind].cost > node.cost:
                        open_set[n_ind].cost = node.cost
                        open_set[n_ind].pind = ind
                # 如果该索引并没有在open_set 中出现，就将这个点加入到open_set中，然后进行重新排序
                else:
                    open_set[n_ind] = node
                    heapq.heappush(q_priority,
                                   (fvalue(node, n_goal), calc_index(node, P)))

    # 调用extract_path函数，生成路径x、y的列表
    pathx, pathy = extract_path(closed_set, n_start, n_goal, P)

    return pathx, pathy


# 生成代价值地图，这一部分在混合a星的规划中用到，单独运行a星时不会调用此函数
def calc_holonomic_heuristic_with_obstacle(node, ox, oy, reso, rr):
    n_goal = Node(round(node.x[-1] / reso), round(node.y[-1] / reso), 0.0, -1)

    ox = [x / reso for x in ox]
    oy = [y / reso for y in oy]

    P, obsmap = calc_parameters(ox, oy, reso, rr)

    open_set, closed_set = dict(), dict()
    open_set[calc_index(n_goal, P)] = n_goal

    q_priority = []
    heapq.heappush(q_priority, (n_goal.cost, calc_index(n_goal, P)))

    while True:
        if not open_set:
            break

        _, ind = heapq.heappop(q_priority)
        n_curr = open_set[ind]
        closed_set[ind] = n_curr
        open_set.pop(ind)

        for i in range(len(P.motion)):
            node = Node(n_curr.x + P.motion[i][0],
                        n_curr.y + P.motion[i][1],
                        n_curr.cost + u_cost(P.motion[i]), ind)

            if not check_node(node, P, obsmap):
                continue

            n_ind = calc_index(node, P)
            if n_ind not in closed_set:
                if n_ind in open_set:
                    if open_set[n_ind].cost > node.cost:
                        open_set[n_ind].cost = node.cost
                        open_set[n_ind].pind = ind
                else:
                    open_set[n_ind] = node
                    heapq.heappush(q_priority, (node.cost, calc_index(node, P)))

    # 前半部分和a星规划相似就不在注释
    # 建立一个长宽分别为xw、yw的矩阵，矩阵的元素全部都为无穷大
    hmap = [[np.inf for _ in range(P.yw)] for _ in range(P.xw)]

    # 将节点的代价值替换上述无穷大的元素，生成代价地图
    for n in closed_set.values():
        hmap[n.x - P.minx][n.y - P.miny] = n.cost

    # 返回代价值地图
    return hmap


# 检测点（node类）是否超出边界或是周围有障碍物
def check_node(node, P, obsmap):
    # 如果监测点位于地图之外，则返回False
    if node.x <= P.minx or node.x >= P.maxx or \
            node.y <= P.miny or node.y >= P.maxy:
        return False

    # 如果检测点位于障碍物地图T的网格内，则返回False
    if obsmap[node.x - P.minx][node.y - P.miny]:
        return False

    # 如果两者都不满足，就返回True，表明为正常点
    return True


# 相当于方向代价，即平移一个网格或者斜着平移一个网格所花费的代价
def u_cost(u):
    return math.hypot(u[0], u[1])


# 计算点的总代价（节点代价加上h代价）
def fvalue(node, n_goal):
    return node.cost + h(node, n_goal)


# 计算h代价（该点到终点的直线距离）
def h(node, n_goal):
    return math.hypot(node.x - n_goal.x, node.y - n_goal.y)


# 计算索引（至于为什么用这样一个计算公式应该就是防止不同点有相同的索引）
def calc_index(node, P):
    return (node.y - P.miny) * P.xw + (node.x - P.minx)


# 获取地图的para以及障碍物地图（用T和F表示）
def calc_parameters(ox, oy, rr, reso):
    # 获取地图障碍物的x、y的最小最大值，以及差值
    minx, miny = round(min(ox)), round(min(oy))
    maxx, maxy = round(max(ox)), round(max(oy))
    xw, yw = maxx - minx, maxy - miny

    # 获取方向（分布在自身节点上下左右以及四个角落共八个节点）
    motion = get_motion()
    # 调用para类，赋值变量p
    P = Para(minx, miny, maxx, maxy, xw, yw, reso, motion)
    # 调用函数，获取障碍物地图（用T和F表示）
    obsmap = calc_obsmap(ox, oy, rr, P)

    return P, obsmap


# 获取障碍物地图
def calc_obsmap(ox, oy, rr, P):
    # 获取长宽为xw、yw的矩阵，并将矩阵的元素全部定义为F
    obsmap = [[False for _ in range(P.yw)] for _ in range(P.xw)]

    # 遍历上述矩阵的所有元素，并将其x、y值的位置进行调整（加上其边界值），最终将靠近障碍物的F转化为T，获取障碍物地图
    for x in range(P.xw):
        xx = x + P.minx
        for y in range(P.yw):
            yy = y + P.miny
            # 遍历所有障碍物的x、y值，如果障碍物距离遍历点的距离小于所给的rr除以网格大小，则该遍历点的F转化为T
            for oxx, oyy in zip(ox, oy):
                if math.hypot(oxx - xx, oyy - yy) <= rr / P.reso:
                    obsmap[x][y] = True
                    break

    # 返回障碍物地图
    return obsmap


# 生成路线
def extract_path(closed_set, n_start, n_goal, P):
    # 先将终点的x、y加入路线的列表中
    pathx, pathy = [n_goal.x], [n_goal.y]
    # 计算终点的索引值
    n_ind = calc_index(n_goal, P)

    while True:
        # closed_set中查找索引为n_ind的点，并赋给node
        node = closed_set[n_ind]
        # 路线x、y的列表中分别添加节点的x、y
        pathx.append(node.x)
        pathy.append(node.y)
        # 将节点的索引值赋值给n_ind，并进行下一步搜索和添加，即通过ind寻找上一个点
        n_ind = node.pind

        # 如果探索的节点是起点的话就结束搜素
        if node == n_start:
            break

    # 将路线的x和y均乘以网格大小并倒置（改为丛起点到终点）
    pathx = [x * P.reso for x in reversed(pathx)]
    pathy = [y * P.reso for y in reversed(pathy)]

    # 返回路径x、y的列表
    return pathx, pathy


# 围绕节点获取八个方向的点
def get_motion():
    motion = [[-1, 0], [-1, 1], [0, 1], [1, 1],
              [1, 0], [1, -1], [0, -1], [-1, -1]]

    return motion

# 生成用点表示的障碍物地图（用于对a星算法进行测试，与停车地图无关）
def get_env():
    ox, oy = [], []

    # 横坐标0-59，纵坐标为0，共60个点（后面的不再注释）
    for i in range(60):
        ox.append(i)
        oy.append(0.0)
    for i in range(60):
        ox.append(60.0)
        oy.append(i)
    for i in range(61):
        ox.append(i)
        oy.append(60.0)
    for i in range(61):
        ox.append(0.0)
        oy.append(i)
    for i in range(40):
        ox.append(20.0)
        oy.append(i)
    for i in range(40):
        ox.append(40.0)
        oy.append(60.0 - i)

    return ox, oy


# a星的主运行函数，只在运行此文件时运行
def main():
    # 定义起点终点坐标
    sx = 10.0  # [m]
    sy = 10.0  # [m]
    gx = 50.0  # [m]
    gy = 50.0  # [m]

    # 定义障碍物探测半径（在生成障碍物地图时用到）
    robot_radius = 2.0
    # 定义网格尺寸
    grid_resolution = 1.0
    # 获取障碍物坐标
    ox, oy = get_env()

    # 获取路径的x、y列表
    pathx, pathy = astar_planning(sx, sy, gx, gy, ox, oy, grid_resolution, robot_radius)

    # 绘制障碍物、路径、起点和终点（其中s指方块，-指实线）（k指黑色，r值红色，g指绿色，b指蓝色）
    plt.plot(ox, oy, 'sk')
    plt.plot(pathx, pathy, '-r')
    plt.plot(sx, sy, 'sg')
    plt.plot(gx, gy, 'sb')
    plt.axis("equal")
    plt.show()


if __name__ == '__main__':
    main()
