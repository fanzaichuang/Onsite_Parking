import os
import sys
import math
import heapq
from heapdict import heapdict
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.kdtree as kd
from planner.hybridastar import astar as astar
from planner.hybridastar import planer_reeds_shepp as rs
from input import make_car
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Onsite_Parking/")

# 获取车辆属性信息
C = make_car.C
# 定义node类（由坐标、航向角索引、行驶方向、坐标、航向角、挡位、代价值、点的索引组成）
class Node:
    def __init__(self, xind, yind, yawind, direction, x, y,
                 yaw, directions, steer, cost, pind):
        self.xind = xind
        self.yind = yind
        self.yawind = yawind
        self.direction = direction
        self.x = x
        self.y = y
        self.yaw = yaw
        self.directions = directions
        self.steer = steer
        self.cost = cost
        self.pind = pind

# 定义地图的para类（由障碍物的最大最小坐标、航向角、坐标航向角最值之差、网格尺寸、障碍物坐标、kdtree组成）
class Para:
    def __init__(self, minx, miny, minyaw, maxx, maxy, maxyaw,
                 xw, yw, yaww, xyreso, yawreso, ox, oy, kdtree):
        self.minx = minx
        self.miny = miny
        self.minyaw = minyaw
        self.maxx = maxx
        self.maxy = maxy
        self.maxyaw = maxyaw
        self.xw = xw
        self.yw = yw
        self.yaww = yaww
        self.xyreso = xyreso
        self.yawreso = yawreso
        self.ox = ox
        self.oy = oy
        self.kdtree = kdtree

# 定义优先序列类
class QueuePrior:
    def __init__(self):
        self.queue = heapdict()

    # 空序列则返回序列长度为0
    def empty(self):
        return len(self.queue) == 0  # if Q is empty

    # 定义queue的索引值为item时的值为priority
    def put(self, item, priority):
        self.queue[item] = priority  # push

    # 去除序列最小代价的值
    def get(self):
        return self.queue.popitem()[0]  # pop out element with smallest priority

# 定义path类（由坐标、航向角、行驶方向、代价值组成）
class Path:
    def __init__(self, x, y, yaw, direction, cost):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.direction = direction
        self.cost = cost

# 计算地图的para
def calc_parameters(ox, oy, xyreso, yawreso, kdtree):
    # 坐标的最大最小值均除以网格尺寸并且四舍五入
    minx = round(min(ox) / xyreso)
    miny = round(min(oy) / xyreso)
    maxx = round(max(ox) / xyreso)
    maxy = round(max(oy) / xyreso)
    # 定义横向和纵向尺寸大小，即最值之差
    xw, yw = maxx - minx, maxy - miny
    # 最大最小航向角也除以相应的尺寸（单位弧度）
    minyaw = round(-C.PI / yawreso) - 1
    maxyaw = round(C.PI / yawreso)
    # 定义航向角的最值之差
    yaww = maxyaw - minyaw

    # 调用并返回地图的para类
    return Para(minx, miny, minyaw, maxx, maxy, maxyaw,
                xw, yw, yaww, xyreso, yawreso, ox, oy, kdtree)

# 生成转角与行驶方向的一维数组，因为车辆属性是固定不变的，所以生成的steer和direc也是固定的
def calc_motion_set():
    # numpy.arange(start, stop, step, dtype=None)用于生成一维数组
    # 这里将车辆的最大转角分为20级并依次列入到一维数组当中
    s = np.arange(C.MAX_STEER / C.N_STEER,
                  C.MAX_STEER, C.MAX_STEER / C.N_STEER)
    # 定义转角的一维数组，长度为s的2倍+1
    steer = list(s) + [0.0] + list(-s)
    # 定义方向的一维数组，长度为steer的2倍
    direc = [1.0 for _ in range(len(steer))] + [-1.0 for _ in range(len(steer))]
    # 将steer的长度扩大2倍，与direc相同
    steer = steer + steer
    return steer, direc

# 计算点的索引值，具体公式应该可以不同，只要能满足不同点有不同的索引值就可以
def calc_index(node, P):
    ind = (node.yawind - P.minyaw) * P.xw * P.yw + \
          (node.yind - P.miny) * P.xw + \
          (node.xind - P.minx)

    return ind

# 计算总的混合代价
def calc_hybrid_cost(node, hmap, P):
    # hmap是在astar文件中生成的代价值地图，C.H_COST在车辆属性中，影响权重，node.cost是节点的代价
    cost = node.cost + \
           C.H_COST * hmap[node.xind - P.minx][node.yind - P.miny]
    return cost

def update_node_with_analystic_expantion(n_curr, ngoal, P):
    # 调用函数求出不发生碰撞的最小成本的rs路径
    path = analystic_expantion(n_curr, ngoal, P)  # rs path: n -> ngoal

    # 如果路径不存在，则返回false、none
    if not path:
        return False, None

    # 分别将path中的x、y、yaw、dir转化为列表类型并去除第一个元素存入fx、fy、fyaw、fd中
    fx = path.x[1:-1]
    fy = path.y[1:-1]
    fyaw = path.yaw[1:-1]
    fd = path.directions[1:-1]

    # 定义fcost为节点的代价加上rs曲线的代价
    fcost = n_curr.cost + calc_rs_path_cost(path)
    # 调用函数求出索引值
    fpind = calc_index(n_curr, P)
    fsteer = 0.0
    # 定义fpath为node类
    fpath = Node(n_curr.xind, n_curr.yind, n_curr.yawind, n_curr.direction,
                 fx, fy, fyaw, fd, fsteer, fcost, fpind)

    return True, fpath

# 调用函数求出rs曲线的代价
# 这里导入的rspath是在planer_reeds_shepp文件中的set_path函数生成的path，属于该文件中的path类，不是列表
def calc_rs_path_cost(rspath):
    # 初始代价值为0
    cost = 0.0
    # 计算倒车代价
    # 遍历rspath.lengths中的所有元素，元素大于0cost就增加一点，小于0cost就增加5（C.BACKWARD_COST）点
    for lr in rspath.lengths:
        if lr >= 0:
            cost += 1
        else:
            cost += abs(lr) * C.BACKWARD_COST
    # 计算换挡代价
    # 当rspath.lengths中的元素存在前后不一致时，增加换挡代价（100）,一样时代价不变
    for i in range(len(rspath.lengths) - 1):
        if rspath.lengths[i] * rspath.lengths[i + 1] < 0.0:
            cost += C.GEAR_COST
    # 计算曲线代价
    # 当rspath.ctypes中含有“S”时，表示生成的rs曲线包含曲线成分，代价值要增加曲线代价
    for ctype in rspath.ctypes:
        if ctype != "S":
            cost += C.STEER_ANGLE_COST * abs(C.MAX_STEER)
    # 计算rspath.ctypes列表的长度
    nctypes = len(rspath.ctypes)
    # 生成一个长度和rspath.ctypes相同，元素值全部为0的列表
    ulist = [0.0 for _ in range(nctypes)]
    # 遍历rspath.ctypes中的所有元素
    for i in range(nctypes):
        # 如果rspath.ctypes含有“R”，则ulist列表中与之对应的元素减去0.6（C.MAX_STEER）
        if rspath.ctypes[i] == "R":
            ulist[i] = -C.MAX_STEER
        # 如果rspath.ctypes含有“WB”，则ulist列表中与之对应的元素加上0.6（C.MAX_STEER）
        elif rspath.ctypes[i] == "WB":
            ulist[i] = C.MAX_STEER
    # 增加换挡代价，权重为5（C.STEER_CHANGE_COST）
    for i in range(nctypes - 1):
        cost += C.STEER_CHANGE_COST * abs(ulist[i + 1] - ulist[i])
    return cost

# 检测两个点的索引值是否相同（全部相同返回false，有一处相同返回true）
def is_same_grid(node1, node2):
    if node1.xind != node2.xind or \
            node1.yind != node2.yind or \
            node1.yawind != node2.yawind:
        return False

    return True


# 导入的x。y。yaw为列表类型，p即para（含有障碍物信息），进行碰撞检测（碰撞返回true）
def is_collision(x, y, yaw, P):
    # d、dl、r分别代表安全距离、车轴长度的一半和半径，cx和cy表示车辆的中心坐标
    for ix, iy, iyaw in zip(x, y, yaw):
        d = 20
        dl = (C.RF - C.RB) / 2.0
        r = (C.RF + C.RB) / 2.0 + d
        cx = ix + dl * math.cos(iyaw)
        cy = iy + dl * math.sin(iyaw)
        # 使用k-d树来查找位于一定范围内的障碍物的索引
        ids = P.kdtree.query_ball_point([cx, cy], r)

        # 如果没有发现障碍物的索引就直接进行下一循环
        if not ids:
            continue

        for i in ids:
            # xo和yo分别是障碍物距离车辆中心点的水平和垂直距离
            xo = P.ox[i] - cx
            yo = P.oy[i] - cy
            # 通过车辆坐标系的转换，转换为沿车辆方向的距离dx和垂直车辆行驶方向的距离dy
            dx = xo * math.cos(iyaw) + yo * math.sin(iyaw)
            dy = -xo * math.sin(iyaw) + yo * math.cos(iyaw)
            # 如果障碍物与车辆中心的距离小于安全距离则返回true，否则返回false
            if abs(dx) < r and abs(dy) < C.W / 2 + d:
                return True
    return False


# 生成在不发生碰撞的情况下最小成本的rs路径
def analystic_expantion(node, ngoal, P):
    # 起点的坐标和航向角均取node列表的最后一个值，终点取ngoal列表的最后一个值
    sx, sy, syaw = node.x[-1], node.y[-1], node.yaw[-1]
    gx, gy, gyaw = ngoal.x[-1], ngoal.y[-1], ngoal.yaw[-1]

    # 由车辆的的最大转角和车辆的轮胎间距计算车辆的最小转弯半径
    maxc = math.tan(C.MAX_STEER) / C.WB
    # 调用planer_reeds_shepp文件中的calc_all_paths函数计算路径
    paths = rs.calc_all_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=C.MOVE_STEP)

    # 没有生成路径就返回none
    if not paths:
        return None

    # 定义优先序列为pq
    pq = QueuePrior()
    # 将路径及其成本放入优先队列pq中
    for path in paths:
        pq.put(path, calc_rs_path_cost(path))

    # 只要pq不为空就一直进行循环
    while not pq.empty():
        # pq.get()将返回优先队列中具有最小成本的路径，并将其从队列中移除
        path = pq.get()
        # 生成0-len（path.x），间隔为5（也就是C.COLLISION_CHECK_STEP）的索引值
        ind = range(0, len(path.x), C.COLLISION_CHECK_STEP)

        # 每隔5个数在path.x中取出一个数并放入新的列表中，下同
        pathx = [path.x[k] for k in ind]
        pathy = [path.y[k] for k in ind]
        pathyaw = [path.yaw[k] for k in ind]

        # 调用函数检测碰撞，如果不碰撞就返回路径，否则就继续循环，直到找出不碰撞的最小成本的路径
        if not is_collision(pathx, pathy, pathyaw, P):
            return path

    return None

# 判断生成的索引是否可用
def is_index_ok(xind, yind, xlist, ylist, yawlist, P):
    # 如果索引值小于最小值或者大于最大值则返回false
    if xind <= P.minx or \
            xind >= P.maxx or \
            yind <= P.miny or \
            yind >= P.maxy:
        return False

    # 生成碰撞检测列表（0-len(xlist)，间隔为5）
    ind = range(0, len(xlist), C.COLLISION_CHECK_STEP)

    # 每隔5个点取出一个点并生成一个新的列表（这样碰撞检测时间缩小5倍，并且基本不影响结果）
    nodex = [xlist[k] for k in ind]
    nodey = [ylist[k] for k in ind]
    nodeyaw = [yawlist[k] for k in ind]

    # 调用碰撞检测函数，检测是否发生碰撞，发生碰撞时返回false
    if is_collision(nodex, nodey, nodeyaw, P):
        return False

    return True

# 生成路径
def extract_path(closed, ngoal, nstart):
    rx, ry, ryaw, direc = [], [], [], []
    cost = 0.0
    node = ngoal

    while True:
        # [::-1]这个切片操作会返回一个反向的列表，这里将node内的各部分进行反向切片后添加到r的列表中
        rx += node.x[::-1]
        ry += node.y[::-1]
        ryaw += node.yaw[::-1]
        direc += node.directions[::-1]
        cost += node.cost

        # 调用函数检测是否有node与nstart是否有相同的索引，即判断两节点是否相同，相同则结束循环
        if is_same_grid(node, nstart):
            break

        # 在closed列表中根据索引值调取新的node进行下一次循环
        node = closed[node.pind]

    # 对r中的各部分进行反向切片操作
    rx = rx[::-1]
    ry = ry[::-1]
    ryaw = ryaw[::-1]
    direc = direc[::-1]

    # 对direc的第一个元素进行赋值，与第二个元素相同
    direc[0] = direc[1]
    # 调用path类赋给path并返回
    path = Path(rx, ry, ryaw, direc, cost)

    return path


def calc_next_node(n_curr, c_id, u, d, P):
    step = C.XY_RESO * 2

    # math.ceil是向上取整，nlist是分段的数量
    nlist = math.ceil(step / C.MOVE_STEP)
    # 定义新的x_list列表的第一个元素就是当前点的最后一个元素做一定的平移
    xlist = [n_curr.x[-1] + d * C.MOVE_STEP * math.cos(n_curr.yaw[-1])]
    ylist = [n_curr.y[-1] + d * C.MOVE_STEP * math.sin(n_curr.yaw[-1])]
    # 新的航向角序列，大小为当前点航向角做一定的变化
    yawlist = [rs.pi_2_pi(n_curr.yaw[-1] + d * C.MOVE_STEP / C.WB * math.tan(u))]

    # 将其他段的变化也加到新列表当中
    for i in range(nlist - 1):
        xlist.append(xlist[i] + d * C.MOVE_STEP * math.cos(yawlist[i]))
        ylist.append(ylist[i] + d * C.MOVE_STEP * math.sin(yawlist[i]))
        yawlist.append(rs.pi_2_pi(yawlist[i] + d * C.MOVE_STEP / C.WB * math.tan(u)))

    # 对新列表进行网格划分并四舍五入，作为索引值
    xind = round(xlist[-1] / P.xyreso)
    yind = round(ylist[-1] / P.xyreso)
    yawind = round(yawlist[-1] / P.yawreso)

    # 判断新的索引值是否是可用索引，不可用的话结束函数并返回none
    if not is_index_ok(xind, yind, xlist, ylist, yawlist, P):
        return None

    cost = 0.0

    # 计算行驶的代价，若d为正数，则行驶方向为正，代价值增加步长大小
    if d > 0:
        direction = 1
        cost += abs(step)
    # 反之，代价值需要增加步长大小并乘以倒车代价
    else:
        direction = -1
        cost += abs(step) * C.BACKWARD_COST

    # 换挡代价
    if direction != n_curr.direction:  # switch back penalty
        cost += C.GEAR_COST

    # 转向角代价
    cost += C.STEER_ANGLE_COST * abs(u)  # steer angle penalyty
    # 转向角变换的代价
    cost += C.STEER_CHANGE_COST * abs(n_curr.steer - u)  # steer change penalty
    # 最后加上节点本身的代价
    cost = n_curr.cost + cost

    directions = [direction for _ in range(len(xlist))]

    # 调用node类赋值给node并返回
    node = Node(xind, yind, yawind, direction, xlist, ylist,
                yawlist, directions, u, cost, c_id)

    return node

# 混合a星的主运行函数
def hybrid_astar_planning(sx, sy, syaw, gx, gy, gyaw, ox, oy, xyreso, yawreso):
    # 将输入的数据除以网格尺寸并四舍五入
    sxr, syr = round(sx / xyreso), round(sy / xyreso)
    gxr, gyr = round(gx / xyreso), round(gy / xyreso)
    syawr = round(rs.pi_2_pi(syaw) / yawreso)
    gyawr = round(rs.pi_2_pi(gyaw) / yawreso)

    # 调用node类，定义起点和终点
    nstart = Node(sxr, syr, syawr, 1, [sx], [sy], [syaw], [1], 0.0, 0.0, -1)
    ngoal = Node(gxr, gyr, gyawr, 1, [gx], [gy], [gyaw], [1], 0.0, 0.0, -1)

    # 生成障碍物的kd树，方便之后的碰撞检测
    kdtree = kd.KDTree([[x, y] for x, y in zip(ox, oy)])
    # 调用函数，生成地图的para
    P = calc_parameters(ox, oy, xyreso, yawreso, kdtree)

    # 调用astar文件的函数生成代价值地图
    hmap = astar.calc_holonomic_heuristic_with_obstacle(ngoal, P.ox, P.oy, P.xyreso, 1.0)
    # 调用函数生成角度和行驶方向的列表
    steer_set, direc_set = calc_motion_set()
    # 定义open_set和closed_set为字典类型，并定义open_set中第一个key个value
    open_set, closed_set = {calc_index(nstart, P): nstart}, {}

    # 定义优先序列
    qp = QueuePrior()
    # 将起点的索引和总代价放入优先序列中
    qp.put(calc_index(nstart, P), calc_hybrid_cost(nstart, hmap, P))

    while True:
        # 如果没有生成open_set，就结束函数并返回none
        if not open_set:
            return None

        # 获取优先序列中代价值最小的索引
        ind = qp.get()
        # 根据索引找出当前点
        n_curr = open_set[ind]
        # 将当前点存入closed_set中
        closed_set[ind] = n_curr
        # 在open_set中移除该索引
        open_set.pop(ind)

        # 调用函数求出代价值最小的rs路径
        update, fpath = update_node_with_analystic_expantion(n_curr, ngoal, P)

        # 如果存在代价值最小的路径，就将该路径赋值给fnode
        if update:
            fnode = fpath
            break

        # 如果无法生成rs路径，则以当前点为基点，求出下一个点
        for i in range(len(steer_set)):
            node = calc_next_node(n_curr, ind, steer_set[i], direc_set[i], P)

            # 若无法求出下一个点，直接进行下一轮循环
            if not node:
                continue

            # 计算点的索引值
            node_ind = calc_index(node, P)

            # 若点已经存在于closed_set中，则不采用该点，就直接进行下一轮循环
            if node_ind in closed_set:
                continue

            # 如果点不在open_set中就将点存入openset中，并获取最小代价值的点
            if node_ind not in open_set:
                open_set[node_ind] = node
                qp.put(node_ind, calc_hybrid_cost(node, hmap, P))
            # 如果点在open_set中就比较代价值并获取代价值最小的点
            else:
                if open_set[node_ind].cost > node.cost:
                    open_set[node_ind] = node
                    qp.put(node_ind, calc_hybrid_cost(node, hmap, P))

    # 调用函数，返回路径
    return extract_path(closed_set, fnode, nstart)

