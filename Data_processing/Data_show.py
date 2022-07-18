# -*- coding: UTF-8 -*-
'''
@Project ：code 
@File    ：Data_show.py , 对数据进行展示，包括：
@                         1. 单个（多个）猪质心点运动轨迹在二值化和RGB背景图展示
@                         2. 单个或者多个生猪运动轨迹、运动速度、运动加速度热力图展示
@                         3. 多个生猪运动平均速度、运动总距离、运动平均加速度柱状图
@Author  ：leeqingshui
@Date    ：2022/6/21 2:02 
'''

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 读取背景mask文件路径
rgb_maskimg_path    = 'F:\\pig_healthy\\code\\pig_data_processing\\mask_rgb.jpg'
binary_maskimg_path = 'F:\\pig_healthy\\code\\pig_data_processing\\mask_binary.jpg'

# =======================================运动轨迹=========================================================
'''
@ 函数功能                          ：单个猪质心点运动轨迹在二值化和RGB背景图展示，
@                                    同时将标注后图片保存到指定文件夹
@ 入口参数 {list}  center_move_list ：生猪质心运动列表，数据格式：
@                                    [[{frame},{x_center},{y_center}],...]
@ 入口参数 {int}   pig_id           ：生猪序号
@ 入口参数 {list}  center_move_list ：生猪质心运动列表，数据格式：[{frame},{x_center},{y_center}]
@ 入口参数 {str}   img_save_path    ：标注后图片的保存路径
@ 入口参数 {tuple} color            ：标注颜色
'''
def Show_move_in_maskimg(center_move_list, img_save_path = '', pig_id = 0, color = (0, 0, 255)):

    print('================================显示背景RGB图像=======================================')
    # 读取图片
    rgb_mask_img = cv.imread(rgb_maskimg_path,1)
    # 显示图片
    cv.imshow('rgb mask img',rgb_mask_img)
    cv.waitKey(5)
    cv.destroyAllWindows()

    print('================================显示背景二值化图像=======================================')
    # 读取图片
    binary_mask_img = cv.imread(binary_maskimg_path,0)
    # 显示图片
    cv.imshow('binary mask img',binary_mask_img)
    cv.waitKey(5)
    cv.destroyAllWindows()

    for center_data in center_move_list:
        # center_data为 [{frame},{x_center},{y_center}]
        x_center = center_data[1]
        y_center = center_data[2]
        cv.circle(rgb_mask_img, (x_center, y_center), 5, color, -1)
        cv.circle(binary_mask_img, (x_center, y_center), 5, color, -1)

    # 显示图片
    cv.imshow(str(pig_id)+'_tag_in_rgbmaskimg',rgb_mask_img)
    cv.waitKey(5)
    cv.destroyAllWindows()

    # 显示图片
    cv.imshow(str(pig_id)+'_tag_in_binarymaskimg',binary_mask_img)
    cv.waitKey(5)
    cv.destroyAllWindows()

    cv.imwrite(img_save_path+'\\'+str(pig_id)+'_tag_in_rgbmaskimg.png', rgb_mask_img)
    cv.imwrite(img_save_path + '\\' +str(pig_id)+ '_tag_in_binarymaskimg.png', binary_mask_img)

# =======================================多只猪对比直方图=========================================================
'''
@ 函数功能                                  ：将八只生猪的总运动距离、运动平均速度、运动平均加速度柱状图对比
@ 入口参数 {list}  total_move_distance_list ：生猪的总运动距离列表
@ 入口参数 {list}  average_speed_list       ：生猪的平均运动速度列表
@ 入口参数 {list}  average_acc_list         ：生猪的平均运动加速度列表
@ 入口参数 {str}   img_save_path            ：标注后图片的保存路径
'''
def Show_Contrast_Histogram(total_move_distance_list, average_speed_list, average_acc_list, img_save_path = ''):

    # 解决中文显示乱码
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    id_list = [1,2,3,4,5,6,7,8]

    # 条形宽度
    width = 0.3

    # 距离条形图横坐标
    index_dis = np.arange(len(id_list))
    index_spd = index_dis + width
    index_arr = index_spd + width

    # 使用三次bar函数画出两组条形图
    plt.bar(index_dis, height=total_move_distance_list, width=width, color='b', label='move distance')
    plt.bar(index_spd, height=average_speed_list,       width=width, color='g', label='move speed')
    plt.bar(index_arr, height=average_acc_list,         width=width, color='r', label='move arr')

    # 显示图例
    plt.legend()
    # 纵坐标轴标题
    plt.ylabel('Pig movement data')
    # 图形标题
    plt.title('Histogram of pig exercise data comparison')
    # 保存直方图
    plt.savefig(img_save_path+"\\"+"Histogram_of_pig_exercise_data_comparison.jpg")

    plt.show()

# =======================================单只猪运动折线图=========================================================

'''
@ 函数功能                                ：绘出固定序列号单只生猪速度-时间折线图
@ 入口参数 {list}   average_speed_list    ：生猪的平均运动速度列表，格式为[[{Next_frame},{move_speed}],...]
@ 入口参数 {str}    img_save_path         ：标注后图片的保存路径
'''
def Show_Speed_Plot(average_speed_list, img_save_path, color = 'r',obj_id = 1):

    # 原速度列表为[[{Next_frame},{move_speed}],...]
    # 取其中速度和帧组成新的列表
    frame_list = []
    speed_list = []
    for data_1d_list in average_speed_list:
        temp_frame = data_1d_list[0]
        frame_list.append(temp_frame)
        temp_speed = data_1d_list[1]
        speed_list.append(temp_speed)

    # 创建画图窗口
    fig = plt.figure()
    # 将画图窗口分成1行1列，选择第一块区域作子图
    ax = fig.add_subplot(1, 1, 1)
    # 设置标题
    ax.set_title('line plot of pig movement velocity ( pig id:'+str(obj_id)+')')
    # 设置横坐标名称
    ax.set_xlabel('frame serial number')
    # 设置纵坐标名称
    ax.set_ylabel('velocity of movement(m/s)')
    # 列出折线图
    ax.plot(frame_list, speed_list, c = color)

    # 保存图片
    plt.savefig(img_save_path+'\\'+'line_plot_of_pig_movement_velocity_'+str(obj_id) +'.jpg', dpi=300)
    # 显示图像
    plt.show()

'''
@ 函数功能                                ：绘出固定序列号单只生猪加速度-时间折线图
@ 入口参数 {list}   average_speed_list    ：生猪的平均运动加速度列表，格式为[[{Next_frame},{move_acc}],...]
@ 入口参数 {str}    img_save_path         ：标注后图片的保存路径
'''
def Show_Acc_Plot(move_acc_list, img_save_path, color = 'b',obj_id = 1):

    # 原速度列表为[[{Next_frame},{move_speed}],...]
    # 取其中速度和帧组成新的列表
    frame_list = []
    acc_list = []
    for data_1d_list in move_acc_list:
        temp_frame = data_1d_list[0]
        frame_list.append(temp_frame)
        temp_speed = data_1d_list[1]
        acc_list.append(temp_speed)

    # 创建画图窗口
    fig = plt.figure()
    # 将画图窗口分成1行1列，选择第一块区域作子图
    ax = fig.add_subplot(1, 1, 1)
    # 设置标题
    ax.set_title('line plot of pig movement accelerated speed ( pig id:'+str(obj_id)+')')
    # 设置横坐标名称
    ax.set_xlabel('frame serial number')
    # 设置纵坐标名称
    ax.set_ylabel('accelerated speed of movement(m/s)')
    # 列出折线图
    ax.plot(frame_list, acc_list, c = color)

    # 保存图片
    plt.savefig(img_save_path+'\\'+'line_plot_of_pig_movement_accelerated_speed_'+str(obj_id) +'.jpg', dpi=300)
    # 显示图像
    plt.show()

'''
@ 函数功能                                   ：绘出多只生猪运动数据柱状图
@ 入口参数 {list}   average_motioninfo_list  ：生猪运动数据一维列表
@ 入口参数 {str}    img_save_path            ：标注后图片的保存路径
@ 入口参数 {str}    info_type                ：运动信息种类，包括：
@                                             距离       ：'distance'
@                                             加速度     ：'accelerated_speed'
@                                             平均速度   ：'average_speed'
@                                             最大速度   ：'max_speed'
@                                             最大加速度 ：'max_acc'
'''
def Show_Motion_Conv(average_motioninfo_list,img_save_path,info_type):

    # 横轴数据，生猪序号
    x_data = [1,2,3,4,5,6,7,8]
    y_data = average_motioninfo_list

    # 画图，plt.bar()可以画柱状图
    for i in range(len(x_data)):
        plt.bar(x_data[i], y_data[i])

    # 设置图片名称
    plt.title("Histogram of "+info_type+" comparison of multiple target pigs")
    # 设置x轴标签名
    plt.xlabel("pig id")
    # 设置y轴标签名
    plt.ylabel(info_type)
    # 保存图片
    plt.savefig(img_save_path+'\\'+"Histogram_of_"+info_type+"_comparison_of_multiple_target_pigs" +'.jpg', dpi=300)
    # 显示
    plt.show()














