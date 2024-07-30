import json
import input.make_map as mp
from planner.hybridastar import planner as planner
from input import make_car
from utils import replay
from utils import map_display as mdp
import matplotlib.pyplot as plt

# 主运行文件
def main():

    # 输入input文件夹下场景文件
    map_path = '../input/Atest.json'
    # 调用map_display文件（在utils文件夹里），绘制场景图片
    # mdp.map_display(map_path)

    # 调用make_map文件，读取13行json文件的信息，并将障碍物信息（ox，oy），起点信息（sp），终点信息（gp）提取出来
    ox, oy,sp,gp = mp.make_map(map_path)
    # 将起点的x、y坐标以及航向角信息提取出来
    sx, sy, syaw0 = sp['x'], sp['y'], sp['yaw']
    # 调用make_car文件，获取车辆的属性信息
    C = make_car.C

    # 获取目标停车位
    park = '3'
    # 停车终点的x、y、航向角信息
    gx, gy, gyaw0 = gp[park]['x_end'], gp[park]['y_end'], gp[park]['yaw']

    # 规划算法（混合a星）
    path = planner.hybrid_astar_planning(sx, sy, syaw0, gx, gy, gyaw0, ox, oy, C.XY_RESO, C.YAW_RESO)
    # 算法测试结果保存
    # 如果道路获取失败，就打印信息并结束运行
    if not path:
        print("Searching failed!")
        return
    # 将车位号、车辆每时每刻的x、y、航向角以及行驶方向（1或-1）存入到字典（输出）中
    output_dit={
        "parking":park,
        "output_x":path.x,
        "output_y": path.y,
        "output_yaw": path.yaw,
        "output_dir": path.direction,
    }
    # 在output文件夹中创建一个文件，文件为json类型，文件根据map_path命名（与map_path略有不同）
    with open(f"../output/result_{map_path.split('/')[-1].split('.json')[0]}_{park}.json", "w") as file:
        json.dump(output_dit, file)

    # 仿真回放
    # 将上一步生成的文件定义为result_path
    result_path = f"../output/result_{map_path.split('/')[-1].split('.json')[0]}_{park}.json"
    # 调用replay文件的replay函数，进行仿真分析
    replay.replay(map_path, result_path)

# 定义主要运行的函数，直接运行主函数
if __name__ == '__main__':
    main()