"""

 仅绘制地图
"""
import matplotlib.pyplot as plt
import input.make_map as mp


# 绘制jpg类型的地图场景
def map_display(map_scene):
    # 将地图场景的文件有json类型的文件转化为jpg类型的文件
    picture_scene =  map_scene.replace('/', '/').replace('.json', '.jpg')
    # 读取这个路径所指定的图像文件
    picture = plt.imread(picture_scene)
    # 获取地图（json类型文件）内部的障碍物（ox，oy）、起点（sp）、终点（gp）信息
    ox, oy,sp,gp = mp.make_map(map_scene)
    #  x 轴的刻度线将会朝向图形内部（注：我也不是太理解，这里不重要）
    plt.rcParams['xtick.direction'] = 'in'
    # 清除生成图片内的坐标轴
    plt.cla()
    # 将障碍物绘制在图片内（“，”表示以小圆点的样式绘制，“k”表示黑色）
    plt.plot(ox, oy, ",k")
    # 对坐标轴的样式做一些标注，例如刻度参数以及刻度线的绘制（不重要）
    plt.tick_params(axis='x', direction='in', top=True, bottom=False, labelbottom=False, labeltop=True)
    # 将x轴与y轴等比例绘制，绘制效果一般比不等比例绘制要好
    plt.axis("equal")
    # 将14行生成的图片显示出来，并且设置其四个角的横纵坐标
    plt.imshow(picture, extent=[min(ox), max(ox), min(oy), max(oy)])
    print("绘制地图结束!")
    # 显示出图片（调用plt模块时要加上，否则不会显示图片）
    plt.show()

