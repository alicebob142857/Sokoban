from controller import Controller
import numpy as np
import random
from utility import more_advanced_astar_search, box_move

def random_map(width=6, height=6, num=2, mode='arbitrary_map'):
    """
    生成随机地图，该算法原理为：
    首先生成一个空白地图，使用适用于空地的算法求解后，如果得到解，那么将没有经过的路径填充墙。
    然后根据这个结果，反推需要一一对应的地图
    """
    map = np.zeros((height, width))
    given_mode_map = np.zeros((height, width))
    end = np.zeros((height, width))
    mask = np.zeros((height, width))
    coordinates = []
    holes = []

    for i in range(height):
        for j in range(width):
            if i == 0 or j == 0 or i == height-1 or j == width-1:
                map[i, j] = 1

    while True:
        x = random.randint(2, width-3)
        y = random.randint(2, height-3)
        if (x, y) in coordinates:
            continue
        else:
            coordinates.append((x, y))
        if len(coordinates) == num + 1:
            break
    initial_pos = coordinates[0]
    map[coordinates[0]] = 3
    if mode == 'given_map':
        given_mode_map[initial_pos] = 3
    coordinates.pop(0)
    count = 1
    for coordinate in coordinates:
        map[coordinate] = 2
        if mode == 'given_map':
            given_mode_map[coordinate] = 2 + count * 10
            count += 1

    while True:
        x = random.randint(1, width-2)
        y = random.randint(1, height-2)
        if (x, y) in coordinates:
            continue
        else:
            holes.append((x, y))
        if len(holes) == num:
            break
    for hole in holes:
        end[hole] = 1

    res, count = more_advanced_astar_search(map, end, initial_pos, width, height)
    if mode == 'given_map':
        map = given_mode_map.copy()
    print(res)
    if res and res != -1:
        tmp_map = map.copy()
        mask[map != 0] = 1
        mask[end != 0] = 1
        for direction in res:
            position = box_move(initial_pos, direction, width, height)
            initial_pos = position
            if tmp_map[position] == 2 or tmp_map[position] % 10 == 2:
                next_position = box_move(position, direction, width, height)
                mask[next_position] = 1
                mask[position] = 1
                tmp_map[next_position] = tmp_map[position]
                tmp_map[position] = 0
            else:
                mask[position] = 1

        map[mask == 0] = 1
        if mode == 'given_map':
            same_indices = np.where(tmp_map > 10)
            coordinates = [(x, y) for x, y in zip(same_indices[0], same_indices[1])]
            for coordinate in coordinates:
                end[coordinate] = int(tmp_map[coordinate] / 10)
        return map, end
    else:
        return None, None

if __name__ == "__main__":
    c = Controller()
    map, end = random_map(mode='given_map')
    if map.any():
        c.make_map(map, end, folder_path='given_map')
