import numpy as np
import heapq
from enum import IntEnum


class Move(IntEnum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0  # 用于处理具有相同优先级的元素顺序问题

    def push(self, priority, item):
        heapq.heappush(self._queue, (priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]

    def empty(self):
        return self._queue == []

    def __len__(self):
        return len(self._queue)

    def __str__(self):
        return str(self._queue)


class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()

    def peek(self):
        if not self.is_empty():
            return self.items[-1]

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)

    def clear(self):
        self.items = []


def manhattan_distance(x, y):
    x1, x2 = x
    y1, y2 = y
    return abs(x1 - y1) + abs(x2 - y2)


def is_end(map, end, mode='arbitrary_map'):
    if mode == 'arbitrary_map':
        return np.all(np.multiply(map, end) == end * 2)
    else:
        return np.all(end == 0)


def is_match(map, end):
    current_map = map.copy()
    current_end = end.copy()
    match_mat = (current_map / 10).astype(int)
    same_indices = np.where((match_mat == current_end) & (current_end > 0))
    for x, y in zip(same_indices[0], same_indices[1]):
        current_map[x, y] = 0
        current_end[x, y] = 0
    return current_map, current_end


def move(map, current_pos, opera, width, height):
    """
    :param
        opera：移动操作
    :return: bool，判断移动是否合法
    """
    return_map = map.copy()
    return_current_pos = current_pos[:]
    if opera == Move.LEFT and current_pos[1] > 0:
        if map[current_pos[0], current_pos[1] - 1] == 0:
            return_map[current_pos] = 0
            return_current_pos = (current_pos[0], current_pos[1] - 1)
            return_map[return_current_pos] = 3
            return return_map, return_current_pos

        elif current_pos[1] >= 2 and map[current_pos[0], current_pos[1] - 1] % 10 == 2\
                and map[current_pos[0], current_pos[1] - 2] == 0:
            return_map[current_pos] = 0
            return_current_pos = (current_pos[0], current_pos[1] - 1)
            return_map[return_current_pos] = 3
            return_map[return_current_pos[0], return_current_pos[1] - 1] = map[return_current_pos]
            return return_map, return_current_pos

    elif opera == Move.RIGHT and current_pos[1] < width - 1:
        if map[current_pos[0], current_pos[1] + 1] == 0:
            return_map[current_pos] = 0
            return_current_pos = (current_pos[0], current_pos[1] + 1)
            return_map[return_current_pos] = 3
            return return_map, return_current_pos

        elif current_pos[1] < width - 2 and map[current_pos[0], current_pos[1] + 1] % 10 == 2\
                and map[current_pos[0], current_pos[1] + 2] == 0:
            return_map[current_pos] = 0
            return_current_pos = (current_pos[0], current_pos[1] + 1)
            return_map[return_current_pos] = 3
            return_map[return_current_pos[0], return_current_pos[1] + 1] = map[return_current_pos]
            return return_map, return_current_pos

    elif opera == Move.UP and current_pos[0] > 0:
        if map[current_pos[0] - 1, current_pos[1]] == 0:
            return_map[current_pos] = 0
            return_current_pos = (current_pos[0] - 1, current_pos[1])
            return_map[return_current_pos] = 3
            return return_map, return_current_pos

        if current_pos[0] >= 2 and map[current_pos[0] - 1, current_pos[1]] % 10 == 2\
                and map[current_pos[0] - 2, current_pos[1]] == 0:
            return_map[current_pos] = 0
            return_current_pos = (current_pos[0] - 1, current_pos[1])
            return_map[return_current_pos] = 3
            return_map[return_current_pos[0] - 1, return_current_pos[1]] = map[return_current_pos]
            return return_map, return_current_pos

    elif opera == Move.DOWN and current_pos[0] < height - 1:
        if map[current_pos[0] + 1, current_pos[1]] == 0:
            return_map[current_pos] = 0
            return_current_pos = (current_pos[0] + 1, current_pos[1])
            return_map[return_current_pos] = 3
            return return_map, return_current_pos

        if current_pos[0] < height - 2 and map[current_pos[0] + 1, current_pos[1]] % 10 == 2 \
                and map[current_pos[0] + 2, current_pos[1]] == 0:
            return_map[current_pos] = 0
            return_current_pos = (current_pos[0] + 1, current_pos[1])
            return_map[return_current_pos] = 3
            return_map[return_current_pos[0] + 1, return_current_pos[1]] = map[return_current_pos]
            return return_map, return_current_pos

    return None, None


def special_distance(boxes, ends):
    """
    这个只统计重合数量
    """
    result = 0
    set1 = set(boxes)
    set2 = set(ends)
    diff1 = set1.difference(set2)
    diff2 = set2.difference(set1)
    for ele1 in diff1:
        min_dist = float('inf')
        min_ele = None
        for ele2 in diff2:
            current_dist = manhattan_distance(ele1, ele2)
            if current_dist < min_dist:
                min_ele = ele2
                min_dist = current_dist
        if min_ele:
            if ele1[0] == min_ele[0] or ele1[1] == min_ele[1]:
                result += 1
            else:
                result += 2
    return result


def distance(boxes, ends):
    """
    计算当前的价值函数。原理是，计算每个箱子到所有目标点的最短曼哈顿距离，然后对所有箱子进行求和
    1. 对箱子到达指定位置进行奖励
    """
    result = 0
    for box in boxes:
        result += min([manhattan_distance(box, end) for end in ends])
    return result


def mat2list(mat, number):
    indices = np.where(mat == number)
    coordinates = list(zip(indices[0], indices[1]))
    return coordinates


def mat_distance(map, end):
    distance(mat2list(map, 2), mat2list(end, 1))


def if_pruning(current_boxes, current_map, end_list, width, height):
    for box in current_boxes:
        if not is_possible_point(current_map, box, end_list, width, height):
            return True
    else:
        return False


def trivial_astar_search(map, end, initial_pos, width, height):
    """
    直接搜索玩家的移动。这样效果并不好，因为大部分算力浪费在玩家跑图的过程中了
    """
    def do_hash(target):
        """
        将列表转化为字符串进行哈希查询
        """
        # 对x, y坐标进行进制转化
        pos = target[0]
        box = target[1]
        return tuple([pos[0] * width + pos[1]] + [x[0] * width + x[1] for x in box])

    operations = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
    end_list = mat2list(end, 1)
    q = PriorityQueue()

    initial_map_list = mat2list(map, 2)
    q.push(distance(initial_map_list, end_list), [map, initial_pos, []])
    closed = set()
    closed.add(do_hash([initial_pos, initial_map_list]))
    count = 0
    while not q.empty():
        count += 1
        print(count)
        current_map, current_pos, path = q.pop()
        if is_end(current_map, end):
            return path, count
        current_boxes = mat2list(current_map, 2)
        index = np.where(current_map == 3)
        tmp_current_map = current_map.copy()
        tmp_current_map[index[0], index[1]] = 0
        if if_pruning(current_boxes, tmp_current_map, end_list, width, height):
            continue
        for opera in operations:
            new_map, new_current_pos = move(current_map, current_pos, opera, width, height)
            if new_map is not None and new_current_pos is not None:
                map_list = mat2list(new_map, 2)
                state = [new_current_pos, map_list]
                if do_hash(state) not in closed:
                    closed.add(do_hash(state))
                    new_path = path + [opera]
                    q.push(len(new_path) + distance(map_list, end_list), [new_map, new_current_pos, new_path])
    return None, None

def simple_move_directions(pos, map, width, height):
    """
    简单去看周围的移动方向
    """
    operations = []
    succ_pos = []
    if pos[1] >= 1 and map[pos[0], pos[1] - 1] == 0:
        operations.append(Move.LEFT)
        succ_pos.append((pos[0], pos[1] - 1))
    if pos[1] < width - 1 and map[pos[0], pos[1] + 1] == 0:
        operations.append(Move.RIGHT)
        succ_pos.append((pos[0], pos[1] + 1))
    if pos[0] >= 1 and map[pos[0] - 1, pos[1]] == 0:
        operations.append(Move.UP)
        succ_pos.append((pos[0] - 1, pos[1]))
    if pos[0] < height - 1 and map[pos[0] + 1, pos[1]] == 0:
        operations.append(Move.DOWN)
        succ_pos.append((pos[0] + 1, pos[1]))
    return operations, succ_pos


def maze_search(map, initial_point, end_point, width, height):
    """
    简单的寻路，判断能不能到目标位置
    :return 寻路的路径
    """
    if initial_point == end_point:
        return [None]
    q = PriorityQueue()
    q.push(manhattan_distance(initial_point, end_point), (initial_point, []))
    closed = [initial_point]
    while not q.empty():
        current_pos, path = q.pop()
        if current_pos == end_point:
            return path
        directions, succ_pos = simple_move_directions(current_pos, map, width, height)
        for i in range(len(directions)):
            if succ_pos[i] not in closed:
                new_path = path + [directions[i]]
                new_cost = manhattan_distance(succ_pos[i], end_point) + len(new_path)
                q.push(new_cost, (succ_pos[i], new_path))
                closed.append(succ_pos[i])
    return None


def reverse_direction(d):
    """
    调转方向
    """
    if d == Move.RIGHT:
        return Move.LEFT
    elif d == Move.LEFT:
        return Move.RIGHT
    elif d == Move.DOWN:
        return Move.UP
    elif d == Move.UP:
        return Move.DOWN
    else:
        return None


def bidirection_maze_search(map, initial_point, end_point, width, height):
    """
    双向寻路，复杂度从之前球的体积4pi*r^3/3降低为pi*r^3/3
    :return 寻路的路径
    """
    def do_hash(target):
        return target[1] + target[0] * width

    if initial_point == end_point:
        return [None]
    q = PriorityQueue()
    q.push(manhattan_distance(initial_point, end_point), (initial_point, [], True))
    q.push(manhattan_distance(initial_point, end_point), (end_point, [], False))
    closed1 = dict()
    closed2 = dict()

    while not q.empty():
        current_pos, path, direction = q.pop()
        if direction:
            if bool(closed2) and do_hash(current_pos) in closed2.keys():
                new_path = closed2[do_hash(current_pos)][:]
                new_path.reverse()
                new_path = path + new_path
                return new_path
            if current_pos == end_point:
                return path
            directions, succ_pos = simple_move_directions(current_pos, map, width, height)
            for i in range(len(directions)):
                if do_hash(succ_pos[i]) not in closed1:
                    new_path = path + [directions[i]]
                    new_cost = manhattan_distance(succ_pos[i], end_point) + len(new_path)
                    q.push(new_cost, (succ_pos[i], new_path, True))
                    closed1[do_hash(succ_pos[i])] = new_path
        else:
            if bool(closed1) and do_hash(current_pos) in closed1.keys():
                new_path = path[:]
                new_path.reverse()
                new_path = closed1[do_hash(current_pos)] + new_path
                return new_path
            if current_pos == initial_point:
                new_path = path.reverse()
                return new_path
            directions, succ_pos = simple_move_directions(current_pos, map, width, height)
            for i in range(len(directions)):
                if do_hash(succ_pos[i]) not in closed2:
                    new_path = path + [reverse_direction(directions[i])]
                    new_cost = manhattan_distance(succ_pos[i], initial_point) + len(new_path)
                    q.push(new_cost, (succ_pos[i], new_path, False))
                    closed2[do_hash(succ_pos[i])] = new_path
    return None


def is_possible_point(map, given_box, end_list, width, height):
    """
    找出完全不可能的情况进行剪枝
    """
    if given_box in end_list:
        return True
    x = given_box[0]
    y = given_box[1]
    area = map[x-1: x+2, y-1: y+2]
    area = area.reshape((9, ))
    if np.all(np.multiply(area, np.array([0, 1, 0, 1, 1, 0, 0, 0, 0])) == np.array([0, 1, 0, 1, 2, 0, 0, 0, 0]))\
        or np.all(np.multiply(area, np.array([0, 1, 0, 0, 1, 1, 0, 0, 0])) == np.array([0, 1, 0, 0, 2, 1, 0, 0, 0]))\
        or np.all(np.multiply(area, np.array([0, 0, 0, 1, 1, 0, 0, 1, 0])) == np.array([0, 0, 0, 1, 2, 0, 0, 1, 0]))\
        or np.all(np.multiply(area, np.array([0, 0, 0, 0, 1, 1, 0, 1, 0])) == np.array([0, 0, 0, 0, 2, 1, 0, 1, 0]))\
        or np.all(np.multiply(area, np.array([0, 1, 0, 1, 1, 0, 0, 0, 0])) == np.array([0, 1, 0, 1, 2, 0, 0, 0, 0]))\
        or np.all(np.multiply(area, np.array([1, 1, 0, 1, 1, 0, 0, 0, 0])) >= np.array([1, 1, 0, 1, 1, 0, 0, 0, 0]))\
        or np.all(np.multiply(area, np.array([0, 1, 1, 0, 1, 1, 0, 0, 0])) >= np.array([0, 1, 1, 0, 1, 1, 0, 0, 0]))\
        or np.all(np.multiply(area, np.array([0, 0, 0, 0, 1, 1, 0, 1, 1])) >= np.array([0, 0, 0, 0, 1, 1, 0, 1, 1]))\
        or np.all(np.multiply(area, np.array([0, 0, 0, 1, 1, 0, 1, 1, 0])) >= np.array([0, 0, 0, 1, 1, 0, 1, 1, 0]))\
        or np.all(np.multiply(area, np.array([0, 1, 1, 1, 1, 0, 0, 1, 1])) == np.array([0, 2, 1, 1, 2, 0, 0, 2, 1]))\
        or np.all(np.multiply(area, np.array([1, 1, 0, 0, 1, 1, 1, 1, 0])) == np.array([1, 2, 0, 0, 2, 1, 1, 2, 0]))\
        or np.all(np.multiply(area, np.array([1, 0, 1, 1, 1, 1, 0, 1, 0])) == np.array([1, 0, 1, 2, 2, 2, 0, 1, 0]))\
        or np.all(np.multiply(area, np.array([0, 1, 0, 1, 1, 1, 1, 0, 1])) == np.array([0, 1, 0, 2, 2, 2, 1, 0, 1])):
        return False
    return True

def box_possible_point(map, box, width, height):
    """
    找出一个箱子可能推动的点位
    """
    possible_move = []
    succ_point = []
    if box[1] >= 1 and map[box[0], box[1] - 1] == 0 and box[1] < width - 1 and map[box[0], box[1] + 1] == 0:
        possible_move.append(Move.RIGHT)
        possible_move.append(Move.LEFT)
    if box[0] >= 1 and map[box[0] - 1, box[1]] == 0 and box[0] < height - 1 and map[box[0] + 1, box[1]] == 0:
        possible_move.append(Move.DOWN)
        possible_move.append(Move.UP)

    return possible_move


def left_detection(current_map, possible_move, box, end_list, width, height):
    count = 0
    if box[0] > 0:
        while box[0] > count + 1 and current_map[box[0] - count - 1, box[1]] == 0 and (box[0] - count - 1, box[1]) not in end_list:
            count += 1
        tmp_count = count
        if (box[0] - tmp_count - 1, box[1]) in end_list:
            index = end_list.index((box[0] - tmp_count - 1, box[1]))
            count += height * (index + 2)
        if current_map[box[0] - tmp_count - 1, box[1]] == 2:
            count += height
    return count


def right_detection(current_map, possible_move, box, end_list, width, height):
    count = 0
    if box[0] < height - 1:
        while box[0] + count + 1 < height - 1 and current_map[box[0] + count + 1, box[1]] == 0 and (box[0] + count + 1, box[1]) not in end_list:
            count += 1
        tmp_count = count
        if (box[0] + tmp_count + 1, box[1]) in end_list:
            index = end_list.index((box[0] + tmp_count + 1, box[1]))
            count += height * (index + 2)
        if current_map[box[0] + tmp_count + 1, box[1]] == 2:
            count += height
    return count


def up_detection(current_map, possible_move, box, end_list, width, height):
    count = 0
    if box[1] > 0:
        while box[1] > count + 1 and current_map[box[0], box[1] - count - 1] == 0 and (box[0], box[1] - count - 1) not in end_list:
            count += 1
        tmp_count = count
        if (box[0], box[1] - tmp_count - 1) in end_list:
            index = end_list.index((box[0], box[1] - tmp_count - 1))
            count += width * (index + 2)
        if current_map[box[0], box[1] - tmp_count - 1] == 2:
            count += width
    return count


def down_detection(current_map, possible_move, box, end_list, width, height):
    count = 0
    if box[1] < width - 1:
        while box[1] < width - count - 2 and current_map[box[0], box[1] + count + 1] == 0 and (box[0], box[1] + count + 1) not in end_list:
            count += 1
        tmp_count = count
        if (box[0], box[1] + tmp_count + 1) in end_list:
            index = end_list.index((box[0], box[1] + tmp_count + 1))
            count += width * (index + 2)
        if current_map[box[0], box[1] + tmp_count + 1] == 2:
            count += width
    return count


def cal_movable_mat(current_map, box_list, end_list, width, height):
    """
    返回状态表。算法中状态为箱子的可移动性
    """
    num = len(box_list)
    movable_tabel = np.zeros((num, 4))
    for i in range(num):
        box = box_list[i]
        # 计算每个方向能移动的距离
        k = 0
        for j in range(4):
            possible_move = Move(j)
            count = 0
            if possible_move == Move.UP and box[0] > 0:
                count = left_detection(current_map, possible_move, box, end_list, width, height)
            if possible_move == Move.DOWN and box[0] < height - 1:
                count = right_detection(current_map, possible_move, box, end_list, width, height)
            if possible_move == Move.LEFT and box[1] > 0:
                count = up_detection(current_map, possible_move, box, end_list, width, height)
            if possible_move == Move.RIGHT and box[1] < width - 1:
                count = down_detection(current_map, possible_move, box, end_list, width, height)

            movable_tabel[i, j] = count
    return movable_tabel


def box_move(pre_box, direction, width, height):
    if direction == Move.UP and pre_box[0] >= 1:
        return (pre_box[0] - 1, pre_box[1])
    if direction == Move.DOWN and pre_box[0] <= height - 2:
        return (pre_box[0] + 1, pre_box[1])
    if direction == Move.LEFT and pre_box[1] >= 1:
        return (pre_box[0], pre_box[1] - 1)
    if direction == Move.RIGHT and pre_box[1] <= width - 2:
        return (pre_box[0], pre_box[1] + 1)
    return None


def get_movable_point(box, direction):
    if direction == Move.UP:
        return (box[0] + 1, box[1])
    elif direction == Move.DOWN:
        return (box[0] - 1, box[1])
    elif direction == Move.LEFT:
        return (box[0], box[1] + 1)
    elif direction == Move.RIGHT:
        return (box[0], box[1] - 1)
    return None


def get_target_point(box, direction):
    if direction == Move.UP:
        return (box[0] - 1, box[1])
    elif direction == Move.DOWN:
        return (box[0] + 1, box[1])
    elif direction == Move.LEFT:
        return (box[0], box[1] - 1)
    elif direction == Move.RIGHT:
        return (box[0], box[1] + 1)
    return None


def new_keep_move(map, current_pos, boxes_list, end_list, current_box_index, direction, width, height):
    """
    只用找正交方向上的前进
    """
    box = boxes_list[current_box_index]
    new_box_list = boxes_list[:]
    tmp_new_box_list = boxes_list[:]

    new_path = []
    new_box = box[:]
    new_current_pos = current_pos[:]

    tmp_new_box = new_box[:]
    tmp_new_box_list = new_box_list[:]

    if direction == Move.UP or direction == Move.DOWN:
        info1 = left_detection(map, Move.LEFT, box, end_list, width, height)
        info2 = right_detection(map, Move.RIGHT, box, end_list, width, height)
    else:
        info1 = up_detection(map, Move.UP, boxes_list[current_box_index], end_list, width, height)
        info2 = down_detection(map, Move.DOWN, boxes_list[current_box_index], end_list, width, height)
    info = [info1, info2]
    tmp_info = info[:]
    count = 0
    while True:
        new_box = box_move(new_box, direction, width, height)
        new_box_list[current_box_index] = new_box
        if direction == Move.UP or direction == Move.DOWN:
            info1 = left_detection(map, Move.LEFT, new_box, end_list, width, height)
            info2 = right_detection(map, Move.RIGHT, new_box, end_list, width, height)
        else:
            info1 = up_detection(map, Move.UP, new_box, end_list, width, height)
            info2 = down_detection(map, Move.DOWN, new_box, end_list, width, height)
        info = [info1, info2]
        if not new_box:
            new_box = tmp_new_box[:]
            new_box_list = tmp_new_box_list[:]
            info = tmp_info[:]
            return new_box_list, new_current_pos, new_path, count
        if map[new_box] != 0:
            new_box = tmp_new_box[:]
            new_box_list = tmp_new_box_list[:]
            info = tmp_info[:]
            return new_box_list, new_current_pos, new_path, count
        if info != tmp_info and count >= 1:
            new_box = tmp_new_box[:]
            new_box_list = tmp_new_box_list[:]
            info = tmp_info[:]
            return new_box_list, new_current_pos, new_path, count
        if new_box in end_list and info == tmp_info:
            new_path.append(direction)
            new_current_pos = box_move(new_current_pos, direction, width, height)
            return new_box_list, new_current_pos, new_path, count + 1
        new_path.append(direction)
        new_current_pos = box_move(new_current_pos, direction, width, height)
        tmp_new_box = new_box[:]
        tmp_new_box_list = new_box_list[:]
        tmp_info = info[:]
        count += 1
    new_current_pos = box_move(new_current_pos, direction, width, height)


def keep_move(map, current_pos, boxes_list, end_list, current_box_index, direction, width, height):
    """
    一直移动直到状态矩阵发生变化
    """
    current_map = map.copy()
    box = boxes_list[current_box_index]
    current_map[box] = 0

    new_box_list = boxes_list[:]
    tmp_new_box_list = boxes_list[:]

    new_path = []
    new_box = box[:]
    new_current_pos = current_pos[:]

    tmp_new_box = new_box[:]
    tmp_new_box_list = new_box_list[:]
    origin_mat = cal_movable_mat(map, new_box_list, end_list, width, height)
    new_mat = origin_mat.copy()

    count = 0
    while True:
        new_box = box_move(new_box, direction, width, height)
        new_box_list[current_box_index] = new_box
        new_mat = cal_movable_mat(map, new_box_list, end_list, width, height)
        if not new_box:
            new_mat = origin_mat.copy()
            new_box = tmp_new_box[:]
            new_box_list = tmp_new_box_list[:]
            return new_box_list, new_current_pos, new_path, count
        if current_map[new_box] != 0:
            new_mat = origin_mat.copy()
            new_box = tmp_new_box[:]
            new_box_list = tmp_new_box_list[:]
            return new_box_list, new_current_pos, new_path, count
        if new_box in end_list:
            new_path.append(direction)
            new_current_pos = box_move(new_current_pos, direction, width, height)
            return new_box_list, new_current_pos, new_path, count + 1
        if np.sum(new_mat != origin_mat) > 2:
            new_path.append(direction)
            new_current_pos = box_move(new_current_pos, direction, width, height)
            return new_box_list, new_current_pos, new_path, count + 1
        new_path.append(direction)
        new_current_pos = box_move(new_current_pos, direction, width, height)
        origin_mat = new_mat.copy()
        tmp_new_box = new_box[:]
        tmp_new_box_list = new_box_list[:]
        count += 1

    return new_box_list, new_current_pos, new_path, count


perfect_move = [3, 0, 0, 3, 3, 1, 1, 3, 1, 2, 1, 2, 2, 0, 0, 3, 1, 2, 1, 1, 2, 2, 0, 0, 3, 0, 3, 1, 3, 3, 3, 0, 2, 0, 0, 2, 2, 1, 1, 3, 1, 3, 0, 2, 2, 2, 1, 3, 2, 2, 1, 1, 3, 3, 0, 0, 3, 0, 3, 3, 1, 2, 1, 2, 3, 0, 0, 2, 1, 2, 2, 0, 3, 1, 2, 2, 1, 1, 3, 3, 0, 1, 2, 2, 0, 0, 3, 0, 3, 3, 1, 3, 1, 2, 0, 0, 2, 2, 1, 3, 2, 2, 1, 1, 3, 3, 0, 3, 3, 0, 3, 0, 2, 1, 1, 2, 2, 1, 2, 2, 0, 0, 3, 0, 3, 1, 0, 0, 0, 3, 3, 1, 1, 2]
perfect_move = [Move(x) for x in perfect_move]


def more_advanced_astar_search(map, end, initial_pos, width, height):
    """
    改进算法，让箱子成为主体，考虑箱子的移动，人的移动直接使用联通算法即可
    """
    def do_hash(target):
        """
        将列表转化为字符串进行哈希查询
        """
        # 对x, y坐标进行进制转化
        return tuple([x[0] * width + x[1] for x in target])

    end_list = mat2list(end, 1)
    # 整体地图只保留墙
    wall_map = np.multiply(map, map <= 1)
    q = PriorityQueue()

    initial_map_list = mat2list(map, 2)
    q.push(special_distance(initial_map_list, end_list), [initial_map_list, initial_pos, [], 0])
    closed = set()
    closed.add(do_hash(initial_map_list))
    count = 0
    max_count = 0
    while not q.empty():
        current_boxes, current_pos, path, pre_cost = q.pop()
        if set(current_boxes) == set(end_list):
            return path, count
        if count > 10000:
            return -1, 10000
        current_map = wall_map.copy()
        # 从list还原出整体地图
        for box in current_boxes:
            current_map[box] = 2
        # 首先检索可以连通的箱子和节点
        print(count)
        num = 0
        if if_pruning(current_boxes, current_map, end_list, width, height):
            # 放入闭节点表
            num += 1
            continue
        count += 1
        for j in range(len(current_boxes)):
            box = current_boxes[j]
            possible_moves = box_possible_point(current_map, box, width, height)
            for possible_move in possible_moves:
                possible_point = get_movable_point(box, possible_move)
                box_path = bidirection_maze_search(current_map, current_pos, possible_point, width, height)
                if box_path:
                    # 移动到可移动性矩阵不再改变为止
                    new_boxes = current_boxes[:]
                    new_boxes, new_current_pos, new_path, move_count = keep_move(current_map, possible_point, current_boxes, end_list,\
                                                    j, possible_move, width, height)
                    if move_count >= 2 or move_count == 0:
                        pass
                    if box != new_current_pos:
                        pass
                    if do_hash(new_boxes) not in closed:
                        if box_path == [None]:
                            box_path = []
                        new_path = path + box_path + new_path

                        new_cost = special_distance(new_boxes, end_list) + pre_cost + 1
                        q.push(new_cost, [new_boxes, new_current_pos, new_path, pre_cost + 1])
                        closed.add(do_hash(new_boxes))
    print(num)
    return None, None


def advanced_astar_search(map, end, initial_pos, width, height):
    """
    改进算法，让箱子成为主体，考虑箱子的移动，人的移动直接使用联通算法即可
    """
    def do_hash(target):
        """
        将列表转化为字符串进行哈希查询
        """
        # 对x, y坐标进行进制转化
        return tuple([x[0] * width + x[1] for x in target])

    end_list = mat2list(end, 1)
    # 整体地图只保留墙
    wall_map = np.multiply(map, map <= 1)
    q = PriorityQueue()

    initial_map_list = mat2list(map, 2)
    q.push(distance(initial_map_list, end_list), [initial_map_list, initial_pos, []])
    closed = set()
    closed.add(do_hash(initial_map_list))
    count = 0
    while not q.empty():
        current_boxes, current_pos, path = q.pop()
        if set(current_boxes) == set(end_list):
            return path, count
        if count > 1000000:
            return -1, 1000000
        current_map = wall_map.copy()
        # 从list还原出整体地图
        for box in current_boxes:
            current_map[box] = 2
        # 首先检索可以连通的箱子和节点
        print(count)
        if if_pruning(current_boxes, current_map, end_list, width, height):
            # 放入闭节点表
            continue
        count += 1
        for j in range(len(current_boxes)):
            box = current_boxes[j]
            possible_moves = box_possible_point(current_map, box, width, height)
            for possible_move in possible_moves:
                possible_point = get_movable_point(box, possible_move)
                box_path = bidirection_maze_search(current_map, current_pos, possible_point, width, height)
                if box_path:
                    # 移动到可移动性矩阵不再改变为止
                    new_boxes = current_boxes[:]
                    new_boxes.remove(box)
                    new_boxes.append((2 * box[0] - possible_point[0], 2 * box[1] - possible_point[1]))
                    new_current_pos = box
                    if do_hash(new_boxes) not in closed:
                        if box_path == [None]:
                            box_path = []
                        new_path = path + box_path + [possible_move]
                        new_cost = distance(new_boxes, end_list) + len(new_path)
                        q.push(new_cost, [new_boxes, new_current_pos, new_path])
                        closed.add(do_hash(new_boxes))
    return None, None


def mat2list_end(end):
    height, width = end.shape
    end_list = []
    index = []
    for i in range(height):
        for j in range(width):
            if end[i, j] > 0:
                end_list.append((i, j))
                index.append(end[i, j])
    combined = list(zip(end_list, index))
    sorted_combined = sorted(combined, key=lambda x: x[1])
    return [item[0] for item in sorted_combined]


def mat2list_box(end):
    height, width = end.shape
    end_list = []
    index = []
    for i in range(height):
        for j in range(width):
            if end[i, j] > 10:
                end_list.append((i, j))
                index.append(int(end[i, j] / 10))
    combined = list(zip(end_list, index))
    sorted_combined = sorted(combined, key=lambda x: x[1])
    return [item[0] for item in sorted_combined]


def mode2_distance(box_list, end_list):
    num = len(end_list)
    dist = 0
    for i in range(num):
        dist += manhattan_distance(box_list[i], end_list[i])
    return dist

def advanced_astar_search_mode2(map, end, initial_pos, width, height):
    """
    改进算法，让箱子成为主体，考虑箱子的移动，人的移动直接使用联通算法即可
    """
    def do_hash(target):
        """
        将列表转化为字符串进行哈希查询
        """
        # 对x, y坐标进行进制转化
        return tuple([x[0] * width + x[1] for x in target])

    end_list = mat2list_end(end)
    # 整体地图只保留墙
    wall_map = np.multiply(map, map <= 1)
    q = PriorityQueue()

    initial_map_list = mat2list_box(map)
    current_end_list = end_list[:]
    q.push(mode2_distance(initial_map_list, end_list), [initial_map_list, current_end_list, initial_pos, []])
    closed = set()
    closed.add(do_hash(initial_map_list))
    count = 0
    while not q.empty():
        current_boxes, current_end_list, current_pos, path = q.pop()
        current_pointer = 0
        new_boxes = current_boxes[:]
        new_end_list = current_end_list[:]
        for i in range(len(new_end_list)):
            if new_end_list[current_pointer] == new_boxes[current_pointer]:
                new_end_list.pop(current_pointer)
                new_boxes.pop(current_pointer)
            else:
                current_pointer += 1
        if not new_end_list:
            return path, count
        if count > 100000:
            return -1, 100000
        current_map = wall_map.copy()
        # 从list还原出整体地图
        for box in new_boxes:
            current_map[box] = 2
        # 首先检索可以连通的箱子和节点
        print(count)
        if if_pruning(new_boxes, current_map, new_end_list, width, height):
            # 放入闭节点表
            continue
        count += 1
        for j in range(len(new_boxes)):
            box = new_boxes[j]
            possible_moves = box_possible_point(current_map, box, width, height)
            for possible_move in possible_moves:
                possible_point = get_movable_point(box, possible_move)
                box_path = bidirection_maze_search(current_map, current_pos, possible_point, width, height)
                if box_path:
                    brand_new_boxes = new_boxes[:]
                    brand_new_boxes[brand_new_boxes.index(box)] = (2 * box[0] - possible_point[0], 2 * box[1] - possible_point[1])
                    new_current_pos = box
                    if do_hash(brand_new_boxes) not in closed:
                        if box_path == [None]:
                            box_path = []
                        new_path = path + box_path + [possible_move]
                        new_cost = mode2_distance(brand_new_boxes, new_end_list) + len(new_path)
                        q.push(new_cost, [brand_new_boxes, new_end_list, new_current_pos, new_path])
                        closed.add(do_hash(new_boxes))
    return None, None