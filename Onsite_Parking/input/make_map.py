import json

# 打开map_path（json文件）,并且获取内部的障碍物、起点、终点信息
def make_map(map_path):

    # 读取json文件
    with open(map_path,'r',encoding='UTF-8') as f:
        map=json.load(f)
    ox = map['obstacles']['ox']
    oy = map['obstacles']['oy']
    sp = map['start_position']
    gp = map['parking_sport']
    # 返回障碍物、起点、终点信息
    return ox, oy, sp, gp

