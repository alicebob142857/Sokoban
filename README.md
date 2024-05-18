# Hollow Knight Sokoban人工智能原理大作业

- UI.py

  程序入口，负责渲染游戏界面并控制游戏模式的选取
- utility.py

  定义了移动Move枚举类，优先级队列和栈等数据结构，并包含运行智能体的各种函数
  主要的搜索算法函数为
    - bidirection_maze_search
        
        双向路径搜索算法，用于搜索地图两点之间的最短路径
    - trivial_astar_search
    
        基础的a*算法，以箱子位置为状态，以玩家的运动为动作
    - advanced_astar_search
    
        改进的a*算法，以箱子位置为状态，以箱子的运动为动作
    - more_advanced_astar_search

        进一步改进的a*算法，以箱子和其他元素的交互为状态，以箱子的运动为动作
    - advanced_astar_search_mode2
  
        完成任务2的算法，基于advanced_astar_search微调而来
- controller.py 

  controller类，负责有游戏的逻辑进行控制，对接界面和智能体函数
- make_map.py

  地图制作模块，负责制作自动生成地图
- settings.json

  游戏设置，包含窗口大小设置和AI播放速度设置
- image

  游戏图片素材
- arbitrary_map

  箱子进入洞口不会消失，且箱子和洞口不一一对应的地图
- given_map

  箱子进入洞口立刻消失，且箱子和洞口一一对应的地图