# -*- coding: UTF-8 -*-
'''
@Project ：code 
@File    ：Data_save.py , 数据保存为csv文件
@Author  ：leeqingshui
@Date    ：2022/7/16 15:29 
'''

import pandas as pd

'''
@ 函数功能                                 ：将三个列表合并后保存到csv文件中
@ 入口参数 {list}   average_distance_list  ：生猪运动距离数据列表
@ 入口参数 {list}   average_speed_list     ：生猪运动速度数据列表
@ 入口参数 {list}   average_acc_list       ：生猪运动加速度数据列表
@ 入口参数 {str}    save_path              ：csv文件保存地址
'''
def dataset_save(average_distance_list , average_speed_list, average_acc_list, save_path):
    # 合并列表
    save_list = []
    # 遍历合并
    for i in range(len(average_distance_list)):
        # 行列表
        temp_save_list = []
        # 添加数据
        temp_save_list.append(average_distance_list[i])
        temp_save_list.append(average_speed_list[i])
        temp_save_list.append(average_acc_list[i])

        save_list.append(temp_save_list)

    tet = pd.DataFrame(data=save_list)
    tet.to_csv(save_path)
    print(len(temp_save_list))





