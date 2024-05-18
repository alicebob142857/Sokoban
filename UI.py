import pygame
import sys
import json
import os
import numpy as np
from controller import Controller, Move
import time
from enum import Enum
from utility import *


class Interface(Enum):
    WELCOME = 0
    PLAYER = 1
    AI = 2
    MVP = 3
    LEVELCHOOSE = 4
    MODECHOOSE = 5

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# 导入设置
with open('settings.json', 'r') as file:
    settings = json.load(file)

interface_settings = settings['interface']
size = width, height = interface_settings['windows width'], interface_settings['windows height']
display_speed = interface_settings['display speed']

pygame.init()
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Hollow Knight Sokoban")
background_image = pygame.image.load(os.path.join('image', 'welcome_background.jpg'))
player_image = pygame.image.load(os.path.join('image', 'player.jpg'))
wall_image = pygame.image.load(os.path.join('image', 'wall.png'))
true_box_image = pygame.image.load(os.path.join('image', 'true_box.png'))
false_box_image = pygame.image.load(os.path.join('image', 'false_box.png'))
point_image = pygame.image.load(os.path.join('image', 'point.png'))
air_image = pygame.image.load(os.path.join('image', 'air.png'))

controller = Controller()
controller.initialize()


def draw_textbox(x, y, box_width, box_height, texts, bg_color=WHITE, text_color=BLACK, character_size=36):
    font = pygame.font.Font(None, character_size)
    pygame.draw.rect(screen, bg_color, (x, y, box_width, box_height))  # 绘制矩形边框

    text_num = len(texts)
    for i in range(text_num):
        text_surface = font.render(texts[i], True, text_color)  # 渲染文本
        text_rect = text_surface.get_rect(center=(x + box_width // 2, y + box_height * (i + 1) // (text_num + 1)))  # 获取文本框的位置
        screen.blit(text_surface, text_rect)  # 绘制文本


def draw_background():
    """
    绘制初始界面
    """
    screen.blit(background_image, (0, 0))
    box_width = 200
    box_height = 120
    text = ['Press wsda to Control', 'Press q to return to', 'the last interface', 'Press e to show holes', 'alicebob142857@github.com', 'COPY LEFT']
    draw_textbox(width - box_width, height - box_height, box_width, box_height, text, character_size=20)


def draw_box_index(index, color, position, box_length):
    """
    绘制箱子和洞的对应编号
    """
    font = pygame.font.Font(None, 40)
    text_surface = font.render("{}".format(index), True, color)
    text_rect = text_surface.get_rect()
    text_rect.center = (position[0] + box_length / 2, position[1] + box_length / 2)
    screen.blit(text_surface, text_rect.topleft)


def draw_game(maps, ends, mode, show_flag=True):
    """
    绘制游戏棋盘
    """
    board_height, board_width = maps.shape
    box_length = round(min(width / (board_width + 2), height / (board_height + 2)))  # 每一个小块的大小
    initial_x = (width - box_length * board_width) / 2
    initial_y = (height - box_length * board_height) / 2

    # 对图片进行缩放
    scaled_player = pygame.transform.scale(player_image, (box_length, box_length))
    scaled_wall = pygame.transform.scale(wall_image, (box_length, box_length))
    scaled_air = pygame.transform.scale(air_image, (box_length, box_length))
    scaled_true_box = pygame.transform.scale(true_box_image, (box_length, box_length))
    scaled_false_box = pygame.transform.scale(false_box_image, (box_length, box_length))
    scaled_point = pygame.transform.scale(point_image, (box_length, box_length))

    for i in range(board_height):
        for j in range(board_width):
            position = (initial_x + j * box_length, initial_y + i * box_length)
            if maps[i, j] == 3 and show_flag:
                screen.blit(scaled_player, position)
            elif maps[i, j] == 1:
                screen.blit(scaled_wall, position)
            elif mode == 'arbitrary_map' and maps[i, j] == 2 and show_flag:
                if ends[i, j] == 1:
                    screen.blit(scaled_true_box, position)
                else:
                    screen.blit(scaled_false_box, position)
            elif mode == 'given_map' and maps[i, j] > 10 and show_flag:
                screen.blit(scaled_false_box, position)
                draw_box_index(int(maps[i, j] / 10), BLACK, position, box_length)
            elif ends[i, j] != 0:
                screen.blit(scaled_point, position)
                if mode == 'given_map':
                    draw_box_index(int(ends[i, j]), RED, position, box_length)
            else:
                pygame.draw.rect(screen, WHITE, (position[0], position[1], box_length, box_length))
                pygame.draw.rect(screen, WHITE, (position[0], position[1], box_length, box_length), 2)


def ai_show_result(controller_obj, results):
    """
    用于展示AI算法求解的结果
    """
    res, count = results
    if res == -1:
        draw_textbox(300, 225, 360, 150, ['Count limit: {}!'.format(count)], character_size=50)
        pygame.display.flip()
        time.sleep(1)
    elif res:
        for opera in res:
            controller_obj.move(opera)
            draw_game(controller_obj.map, controller_obj.end, controller_obj.mode)
            pygame.display.flip()
            time.sleep(display_speed)
        draw_textbox(330, 225, 300, 150, ['Counts cost: {}'.format(count)], character_size=50)
        pygame.display.flip()
        time.sleep(0.5)
    else:
        draw_textbox(300, 225, 360, 150, ['No Solution'], character_size=50)
        pygame.display.flip()
        time.sleep(1)
    pygame.event.clear()


# 执行死循环，确保窗口一直显示
game_state = Interface.WELCOME
# 使用栈控制状态
s = Stack()
while True:
    draw_background()
    if game_state == Interface.WELCOME:
        draw_background()
        button_player = pygame.Rect(390, 175, 180, 50)
        button_ai = pygame.Rect(390, 375, 180, 50)
        draw_textbox(button_player.x, button_player.y, button_player.width, button_player.height,
                     ['Human Player'], character_size=36)
        draw_textbox(button_ai.x, button_ai.y, button_ai.width, button_ai.height,
                     ['AI Player'], character_size=36)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # 如果鼠标点击了按钮，输出点击按钮
                if button_player.collidepoint(event.pos):
                    s.push(Interface.PLAYER)
                    s.push(Interface.LEVELCHOOSE)
                    s.push(Interface.MODECHOOSE)
                    game_state = s.pop()
                if button_ai.collidepoint(event.pos):
                    s.push(Interface.AI)
                    s.push(Interface.LEVELCHOOSE)
                    s.push(Interface.MODECHOOSE)
                    game_state = s.pop()

    elif game_state == Interface.MODECHOOSE:
        button_arbitrary = pygame.Rect(390, 175, 180, 50)
        button_given = pygame.Rect(390, 375, 180, 50)
        draw_textbox(button_arbitrary.x, button_arbitrary.y, button_arbitrary.width, button_arbitrary.height,
                     ['arbitrary holes'], character_size=36)
        draw_textbox(button_given.x, button_given.y, button_given.width, button_given.height,
                     ['given holes'], character_size=36)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    s.clear()
                    game_state = Interface.WELCOME
                    controller.initialize()
                    pygame.event.clear()
                    break
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # 如果鼠标点击了按钮，输出点击按钮
                if button_arbitrary.collidepoint(event.pos):
                    game_state = s.pop()
                    controller.set_mode('arbitrary_map')
                    break
                elif button_given.collidepoint(event.pos):
                    game_state = s.pop()
                    controller.set_mode('given_map')
                    break

    elif game_state == Interface.LEVELCHOOSE:
        box_width = 100
        box_height = 50
        initial_x = width / 8 - box_width / 2
        initial_y = height / 4 - box_height / 2
        files_num = len(os.listdir(controller.mode))
        button_player = []
        for j in range(2):
            for i in range(4):
                number = 4 * j + i
                box_x = initial_x + i * width / 4
                box_y = initial_y + j * height / 2
                if (number < files_num):
                    draw_textbox(box_x, box_y, box_width, box_height, '{}'.format(number), character_size=36)
                    button_player.append(pygame.Rect(box_x, box_y, box_width, box_height))
                else:
                    draw_textbox(box_x, box_y, box_width, box_height, '-', character_size=20)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    game_state = Interface.MODECHOOSE
                    s.push(Interface.LEVELCHOOSE)
                    controller.initialize()
                    pygame.event.clear()
                    break
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # 如果鼠标点击了按钮，输出点击按钮
                for i in range(files_num):
                    if button_player[i].collidepoint(event.pos):
                        game_state = s.pop()
                        controller.initialize(given_file_path=os.path.join(controller.mode, '{}.npy'.format(i)))
                        break

    elif game_state == Interface.PLAYER:
        keys = pygame.key.get_pressed()
        show_flag = True
        if keys[pygame.K_e]:
            show_flag = False
        draw_game(controller.map, controller.end, controller.mode, show_flag)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    game_state = Interface.WELCOME
                    controller.initialize()
                    pygame.event.clear()
                    break
                elif event.key == pygame.K_w:
                    controller.move(Move.UP)
                elif event.key == pygame.K_s:
                    controller.move(Move.DOWN)
                elif event.key == pygame.K_a:
                    controller.move(Move.LEFT)
                elif event.key == pygame.K_d:
                    controller.move(Move.RIGHT)
        if controller.is_end():
            pygame.event.clear()
            game_state = Interface.MVP

    elif game_state == Interface.MVP:
        draw_game(controller.map, controller.end, controller.mode)
        draw_textbox(330, 225, 300, 150, ['Winner!'], character_size=50)
        pygame.display.flip()
        time.sleep(0.5)
        controller.initialize()
        pygame.event.clear()
        game_state = Interface.WELCOME

    elif game_state == Interface.AI:
        draw_game(controller.map, controller.end, controller.mode)
        draw_textbox(300, 225, 360, 150, ['Solving...'], character_size=50)
        pygame.display.flip()
        results = controller.solve()
        ai_show_result(controller, results)
        game_state = Interface.WELCOME
        controller.initialize()

    pygame.display.flip()
pygame.quit()

