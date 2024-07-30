"""
算法结果回放

"""
import json
import math
import matplotlib.pyplot as plt
import input.make_map as mp
from utils import drawcar as tools, reeds_shepp as rs
from input import make_car

def replay(map_scene, output_result):
    picture_scene = map_scene.replace('/', '/').replace('.json', '.jpg')
    C = make_car.C
    ox, oy, sp, gp = mp.make_map(map_scene)

    with open(output_result, 'r', encoding='UTF-8') as f:
        result = json.load(f)
    x = result['output_x']
    y = result['output_y']
    yaw = result['output_yaw']
    direction = result['output_dir']
    p = result['parking']

    sx, sy, syaw0 = sp['x'], sp['y'], sp['yaw']
    gx, gy, gyaw0 = gp[p]['x_end'], gp[p]['y_end'], gp[p]['yaw']
    plt.rcParams['xtick.direction'] = 'in'
    plt.cla()
    plt.plot(ox, oy, ",k")
    plt.tick_params(axis='x', direction='in', top=True, bottom=False, labelbottom=False, labeltop=True)
    plt.axis("equal")
    tools.draw_car(plt, gx, gy, gyaw0, 0.0, color='dimgray')
    tools.draw_car(plt, sx, gy, gyaw0, 0.0, color='dimgray')
    picture = plt.imread(picture_scene)
    ox1, ox2, oy1, oy2 = min(ox), max(ox), min(oy), max(oy)
    plt.imshow(picture, extent=[ox1, ox2, oy1, oy2], aspect='auto')
    for k in range(len(x)):
        plt.cla()
        plt.imshow(picture, extent=[ox1, ox2, oy1, oy2], aspect='auto')
        # plt.plot(ox, oy, ",k")
        plt.plot(x, y, linewidth=0.5, color='b', linestyle='--')
        if k < len(x) - 2:
            dy = (yaw[k + 1] - yaw[k]) / C.MOVE_STEP
            steer = rs.pi_2_pi(math.atan(-C.WB * dy / direction[k]))
        else:
            steer = 0.0
        tools.draw_car(plt, gx, gy, gyaw0, 0.0, 'dimgray')
        tools.draw_car(plt, x[k], y[k], yaw[k], steer)
        plt.title("Simulation Result", loc='left', fontweight="heavy")
        plt.axis("equal")
        plt.pause(0.001)

    print("仿真结束!")
    plt.show()


# 对某一个车位的结果进行仿真回放
def main():
    map_path = '../input/Atest.json'
    result_path = '../output/result_Atest_3.json'
    replay(map_path, result_path)

if __name__ == '__main__':
    main()