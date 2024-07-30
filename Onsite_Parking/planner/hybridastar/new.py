import heapq

# 定义网格大小
ROW_COUNT = 5
COL_COUNT = 5

# 定义网格地图，0表示可通过的空地，1表示障碍物
grid = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0]
]


# 定义启发式函数，这里使用曼哈顿距离作为启发式估计
def heuristic_cost_estimate(current, goal):
    return abs(current[0] - goal[0]) + abs(current[1] - goal[1])


# 定义A*算法函数
def astar(start, goal):
    # 定义开放列表和关闭列表
    open_list = []
    heapq.heappush(open_list, (0, start))  # 使用堆来实现优先队列
    came_from = {}
    g_score = {start: 0}

    while open_list:
        current_cost, current_node = heapq.heappop(open_list)

        if current_node == goal:
            path = []
            while current_node in came_from:
                path.append(current_node)
                current_node = came_from[current_node]
            path.append(start)
            path.reverse()
            return path

        for neighbor in neighbors(current_node):
            tentative_g_score = g_score[current_node] + 1  # 假设所有移动的成本都为1

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current_node
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic_cost_estimate(neighbor, goal)
                heapq.heappush(open_list, (f_score, neighbor))

    return None  # 如果开放列表为空但没有找到路径，则返回None


# 定义获取邻居节点的函数
def neighbors(node):
    row, col = node
    candidates = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]  # 上下左右四个方向
    valid_neighbors = []
    for r, c in candidates:
        if 0 <= r < ROW_COUNT and 0 <= c < COL_COUNT and grid[r][c] == 0:
            valid_neighbors.append((r, c))
    return valid_neighbors


# 测试
start_node = (0, 0)
goal_node = (4, 4)
path = astar(start_node, goal_node)
if path:
    print("找到路径：", path)
else:
    print("无法找到路径。")
