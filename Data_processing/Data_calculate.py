# -*- coding: UTF-8 -*-
'''
@Project ：code 
@File    ：Data_calculate.py , 计算生猪运动距离、运动速度、运动加速度、静止时间
@Author  ：leeqingshui
@Date    ：2022/6/21 2:37 
'''

import math
import numpy as np

# 每秒帧率，为30帧
FPS = 10
time = FPS/30
'''
@ 函数功能                          ：输出固定序号生猪的质心运动列表
@ 入口参数 {list}  move_data_list   ：运动轨迹列表,数据格式为:
@                                    [[{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1],...]
@ 返回参数 {list}  center_move_list ：生猪质心运动列表，数据格式：
@                                    [[{frame},{x_center},{y_center}],...]
'''
def Get_CenterMoveData_List(move_data_list):
    # 生猪质心运动列表
    center_move_list = []

    for move_data in move_data_list:
        # move_data格式为[{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1]
        # 其中x, y, w, h 表示了框的左上角坐标 + 长 + 宽

        temp_center_line_data = []

        # 提取原数据
        frame = move_data[0]
        x_left_up = move_data[2]
        y_left_up = move_data[3]
        w = move_data[4]
        h = move_data[5]
        print('原数据：', 'frame:' + str(frame), '左上角横坐标：' + str(x_left_up), '左上角纵坐标：' + str(y_left_up), '宽度：' + str(w),
              '高度：' + str(h))

        # 坐标数据格式转换 , //代表整除
        x_center = x_left_up + w // 2
        y_center = y_left_up + h // 2
        print('转换数据：', 'frame:' + str(frame), 'x_center:' + str(x_center), 'y_center：' + str(y_center))

        temp_center_line_data.append(frame)
        temp_center_line_data.append(x_center)
        temp_center_line_data.append(y_center)

        center_move_list.append(temp_center_line_data)

    print(center_move_list)
    return center_move_list

'''
@ 函数功能                            ：输出固定序号生猪每一帧的质心位移列表
@ 入口参数 {list}  center_move_list   ：运动轨迹列表,数据格式为:
@                                      [[{frame},{x_center},{y_center}],...]
@ 返回参数 {list}  move_distance_list ：生猪质心运动列表，数据格式：
@                                      [[{Next_frame},{move_distance}],...]
'''
def Get_MoveDistance_List(center_move_list):
    move_distance_list = []

    # 计数变量
    i = -1

    for data in center_move_list:

        temp_1d_list = []
        distance_list = []
        i = i + 1

        if i == 0:
            # 跳过第一帧
            continue

        # 提取每s的运动距离
        elif i%FPS ==0:
            frame_next = data[0]

            # 计算像素距离
            x_last = center_move_list[i - 1][1]
            y_last = center_move_list[i - 1][2]

            x_next = center_move_list[i][1]
            y_next = center_move_list[i][2]

            distance = math.sqrt((x_next - x_last) * (x_next - x_last) + (y_next - y_last) * (y_next - y_last))

            # 根据像素距离计算实际距离
            distance = Get_true_distance(distance)

            temp_1d_list.append(frame_next)
            temp_1d_list.append(distance)

            move_distance_list.append(temp_1d_list)
        else:
            continue

    return move_distance_list

'''
@ 函数功能                         ：根据像素距离计算实际距离
@ 入口参数 {float}  Pixel_distance ：像素距离
@ 返回参数 {float}  true_distance  ：实际距离
'''
def Get_true_distance(Pixel_distance):

    true_distance = (Pixel_distance * 1.9) / 720

    return true_distance

'''
@ 函数功能                             ：计算总移动距离
@ 入口参数 {list}  move_distance_list  ：距离列表 ，计算相邻两帧间移动距离
@                                       [[{Next_frame},{move_distance}],...]
@ 返回参数 {float}  total_distance     ：总距离
'''
def Get_total_distance(move_distance_list):
    total_distance = 0

    for x in move_distance_list:
        total_distance += x[1]

    return total_distance

'''
@ 函数功能                            ：输出固定序号生猪相邻两帧的瞬时速度
@ 入口参数 {list}  move_distance_list ：生猪质心运动列表，数据格式：
@                                      [[{Next_frame},{move_distance}],...]
@ 返回参数 {list}  move_speed_list    ：生猪运动速度列表，数据格式：
@                                      [[{Next_frame},{move_speed}],...]
'''
def Get_MoveSpeed_List(move_distance_list):
    move_speed_list = []
    time = FPS/30

    for data in move_distance_list:
        temp_1d_list = []

        frame = data[0]
        speed = data[1] / time

        temp_1d_list.append(frame)
        temp_1d_list.append(speed)

        move_speed_list.append(temp_1d_list)

    return move_speed_list

'''
@ 函数功能                            ：计算移动平均速度
@ 入口参数 {list}  move_distance_list ：距离列表 ，[[{Next_frame},{move_distance}],...]
@ 入口参数 {float} total_distance     ：运动总距离
@ 返回参数 {float}  average_speed     ：平均速度
'''
def Get_Average_Speed(move_distance_list, total_distance):
    average_speed = 0
    last_index = len(move_distance_list)

    init_frame = move_distance_list[0][0]
    last_frame = move_distance_list[last_index - 1][0]

    print('初始帧为：' + str(init_frame))
    print('结束帧为：' + str(last_frame))

    total_time = (1 / FPS) * (last_frame - init_frame)
    average_speed = total_distance / total_time

    return average_speed

'''
@ 函数功能                             ：输出固定序号生猪相邻两帧的加速度列表
@ 入口参数 {list}  move_speed_list     ：生猪运动速度列表，数据格式：
@                                       [[{Next_frame},{move_speed}],...]
@ 返回参数 {list}  move_acc_list       ：生猪运动加速度列表，数据格式：
@                                       [[{Next_frame},{move_acc}],...]
'''
def Get_Acc_List(move_speed_list):
    move_acc_list = []
    time = FPS/30

    # 计数变量
    i = -1

    for data in move_speed_list:

        temp_1d_list = []
        i = i + 1

        if i == 0:
            # 跳过第一帧
            continue
        else:
            frame_next = data[0]

            v_last = move_speed_list[i - 1][1]
            v_next = move_speed_list[i][1]

            acc_next = abs((v_last - v_next) / time)

            temp_1d_list.append(frame_next)
            temp_1d_list.append(acc_next)

            move_acc_list.append(temp_1d_list)

    return move_acc_list

'''
@ 函数功能                            ：计算移动平均加速度
@ 入口参数 {list}   move_acc_list     ：生猪运动加速度列表，数据格式：
@                                       [[{Next_frame},{move_acc}],...]
@ 返回参数 {float}  average_acc       ：平均加速度
'''
def Get_Average_Acc(move_acc_list):
    total_acc = 0

    last_index = len(move_acc_list)

    init_frame = move_acc_list[0][0]
    last_frame = move_acc_list[last_index - 1][0]

    print('初始帧为：' + str(init_frame))
    print('结束帧为：' + str(last_frame))

    for acc_data in move_acc_list:
        total_acc = total_acc + acc_data[1]

    total_time = (1 / FPS) * (last_frame - init_frame)
    average_acc = total_acc / total_time

    return average_acc

'''
@ 函数功能                         ：求最大值
@ 入口参数 {list}   move_info_list ：生猪运动加速度列表，数据格式：
@                                   [[{Next_frame},{move_info}],...]
@ 返回参数 {float}  max_value      ：平均加速度
'''
def Get_MaxValue(move_info_list):
    # 临时存值列表
    temp_info_value = []

    for data in move_info_list:
        temp_info_value.append(data[1])

    return(max(temp_info_value))





