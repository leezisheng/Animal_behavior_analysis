# -*- coding: UTF-8 -*-
'''
@Project ：code ，对Data_calculate.py、Data_parsing.py、Data_show.py中函数进行测试
@File    ：single_test.py
@Author  ：leeqingshui
@Date    ：2022/6/21 2:44 
'''

from Data_parsing import openreadtxt,Data_format_trans,Get_move_data
from Data_calculate import Get_CenterMoveData_List, Get_MoveDistance_List, Get_total_distance,\
     Get_MoveSpeed_List, Get_Average_Speed, Get_Acc_List,Get_Average_Acc
from Data_show import Show_move_in_maskimg,Show_Speed_Plot,Show_Acc_Plot

# =============================================全局变量=========================================================
# 测试txt文件路径
test_txt_path = "F:\\pig_healthy\\code\\output\\results.txt"
# 选定要看的目标生猪序号
pig_id = 7
# 结果图片保存地址
mask_result_img_save_path = 'F:\\pig_healthy\\code\\pig_data_processing\\result\\center_move_in_mask'
single_pig_motion_plot_save_path = 'F:\\pig_healthy\\code\\pig_data_processing\\result\\single_pig_motion_plot'

# ========================================从txt文件中解析数据===================================================
print('==============================================================================')
print('开始解析文件：'+test_txt_path+'中数据')

try:
    # 读取txt文件中每一行数据，保存到line_data_list中
    line_data_list = openreadtxt(test_txt_path)
    # 对行数据二维列表中元素从字符串格式到int格式进行转换
    line_data_list = Data_format_trans(line_data_list)
    # 选出id号代表的某头猪的运动轨迹数据列表
    move_data_list = Get_move_data(line_data_list , pig_id)
    print('解析文件中数据成功')
except:
    print('解析文件中数据失败')

# ========================================计算运动距离、速度、加速度=============================================

# 输出固定序号生猪的质心运动列表
# 数据格式：[[{frame},{x_center},{y_center}],...]
center_move_list = Get_CenterMoveData_List(move_data_list)

# 显示并保存标注轨迹点后背景图片
Show_move_in_maskimg(center_move_list,
                     img_save_path = mask_result_img_save_path,
                     pig_id = pig_id,
                     color = (0, 0, 255))
print(center_move_list)

# 计算移动距离，获得每一帧下移动距离的列表
move_distance_list = Get_MoveDistance_List(center_move_list)
# print("生猪每一帧移动距离：")
# print(move_distance_list)

# 计算总运动距离
total_distance = Get_total_distance(move_distance_list)
print('序号为：'+str(pig_id)+'的生猪总移动距离为：'+str(total_distance)+'(m)')

# 计算瞬时速度列表
move_speed_list = Get_MoveSpeed_List(move_distance_list)
# print("生猪每一帧瞬时速度：")
# print(move_speed_list)

# 计算总平均速度
average_speed = Get_Average_Speed(move_distance_list, total_distance)
print('序号为：'+str(pig_id)+'的生猪总移动速度为：'+str(average_speed)+'(m/s)')

# 计算瞬时加速度列表
move_acc_list = Get_Acc_List(move_speed_list)
# print("生猪每一帧瞬时加速度：")
# print(move_acc_list)

# 计算总平均加速度
average_acc = Get_Average_Acc(move_acc_list)
print('序号为：'+str(pig_id)+'的生猪总移动加速度为：'+str(average_acc)+'(m/s2)')

# ============================================数据展示=======================================================

# 显示固定id生猪的速度-时间折线图
Show_Speed_Plot(average_speed_list = move_speed_list,
                img_save_path = single_pig_motion_plot_save_path,
                obj_id = pig_id)

# 显示固定id生猪的加速度-时间折线图
Show_Acc_Plot(move_acc_list = move_acc_list,
              img_save_path = single_pig_motion_plot_save_path,
              obj_id = pig_id)

