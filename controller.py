import numpy as np
import os
import random
from utility import *
import time


class Controller:
    """
    Controller类负责整个游戏的逻辑
        self.map：地图中墙，箱子和空地的位置
        self.end：地图中洞口的位置
    """
    def __init__(self):
        self.width = 0
        self.height = 0
        self.map = None
        self.end = None
        self.current_pos = (0, 0)
        self.mode = 'arbitrary_map'
        self.current_map = None

    def set_mode(self, mode):
        """
        设置mode
        """
        self.mode = mode

    def initialize(self, given_file_path=None):
        """
        随机从地图文件夹中选取一个地图对地图进行初始化
        :param
            folder_path：游戏默认地图文件的位置
            given_file_path：强制指定的文件位置，用于AI进行回放
        :return: None
        """
        folder_path = self.mode
        files = os.listdir(folder_path)
        if files:
            if given_file_path:
                self.current_map = given_file_path
            else:
                random_file = random.choice(files)
                file_path = os.path.join(folder_path, random_file)
                self.current_map = file_path
            loaded_matrix = np.load(self.current_map)
            self.map = loaded_matrix[0, :, :]
            self.end = loaded_matrix[1, :, :]
            current_pos_index = np.where(self.map == 3)
            self.current_pos = (current_pos_index[0][0], current_pos_index[1][0])
            self.height, self.width = self.map.shape
        else:
            print("Fail to initialize")

    def make_map(self, map_matrix, end_matrix, folder_path='arbitrary_map'):
        """
        制作地图
        :param
            path：存储地图文件的位置
        :return: None
        """
        files = os.listdir(folder_path)
        index = len(files)
        print(files)
        file_path = os.path.join(folder_path, '{}.npy'.format(index))
        np.save(file_path, np.array([map_matrix, end_matrix]))

    def is_end(self):
        """
        判断是否已经将箱子全部推入
        :param: None
        :return: bool
        """
        return is_end(self.map, self.end, self.mode)

    def move(self, opera):
        """
        :param
            opera：移动操作
        :return: bool，判断移动是否合法
        """
        new_map, new_current_pos = move(self.map, self.current_pos, opera, self.width, self.height)
        if new_map is not None and new_current_pos is not None:
            self.map, self.current_pos = new_map, new_current_pos
            if self.mode == 'given_map':
                self.map, self.end = is_match(self.map, self.end)
            return True
        else:
            return False

    def solve(self):
        """
        调用A*算法求解路径
        """
        start_time = time.time()
        if self.mode == 'arbitrary_map':
            res, count = more_advanced_astar_search(self.map, self.end, self.current_pos, self.width, self.height)
            if res:
                end_time = time.time()
                print(end_time - start_time)
                return res, count
            else:
                res, count = trivial_astar_search(self.map, self.end, self.current_pos, self.width, self.height)
                if res:
                    end_time = time.time()
                    print(end_time - start_time)
                    return res, count
                else:
                    return None, None
        else:
            res, count = advanced_astar_search_mode2(self.map, self.end, self.current_pos, self.width, self.height)
            return res, count


if __name__ == "__main__":
    c = Controller()
    c.set_mode('given_map')
    c.initialize(given_file_path='given_map/0.npy')
    print(c.map)
    print(c.end)
    print(c.solve())