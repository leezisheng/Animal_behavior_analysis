# -*- coding: UTF-8 -*-
'''
@Project ：code 
@File    ：multi_test.py , 对Data_calculate.py、Data_parsing.py、Data_show.py中函数进行测试
@Author  ：leeqingshui
@Date    ：2022/6/25 0:53 
'''
from Data_parsing import openreadtxt,Data_format_trans,Get_move_data
from Data_calculate import Get_CenterMoveData_List, Get_MoveDistance_List, Get_total_distance,\
     Get_MoveSpeed_List, Get_Average_Speed, Get_Acc_List,Get_Average_Acc, Get_MaxValue
from Data_show import Show_move_in_maskimg,Show_Speed_Plot,Show_Acc_Plot,Show_Motion_Conv

# =============================================全局变量=========================================================
# 测试txt文件路径
test_txt_path = "F:\\pig_healthy\\code\\output\\results.txt"
# 结果图片保存地址
mask_result_img_save_path          = 'F:\\pig_healthy\\code\\pig_data_processing\\result\\center_move_in_mask'
single_pig_motion_plot_save_path   = 'F:\\pig_healthy\\code\\pig_data_processing\\result\\single_pig_motion_plot'
multi_pig_conv_histogram_save_path = 'F:\\pig_healthy\\code\\pig_data_processing\\result\\motion_information_Histogram'

# 八只猪 平均距离列表
average_distance_list = []
# 八只猪 平均加速度列表
average_acc_list      = []
# 八只猪 平均速度列表
average_speed_list    = []
# 八只猪 最大速度列表
max_speed_list        = []
# 八只猪 最大加速度列表
max_acc_list          = []

# 数据txt文件保存名字
save_txt_name = '03.txt'
# 数据txt文件保存地址
save_path     = 'F:\\pig_healthy\\code\\pig_data_processing\\multi_result'+'\\'+ save_txt_name

# ============================================数据处理===========================================================

for pig_id in range(1,9):
    # ========================================从txt文件中解析数据===================================================
    print('==============================================================================')
    print('开始解析文件：' + test_txt_path + '中数据')

    try:
        # 读取txt文件中每一行数据，保存到line_data_list中
        line_data_list = openreadtxt(test_txt_path)
        # 对行数据二维列表中元素从字符串格式到int格式进行转换
        line_data_list = Data_format_trans(line_data_list)
        # 选出id号代表的某头猪的运动轨迹数据列表
        move_data_list = Get_move_data(line_data_list, pig_id)
        print('解析文件中数据成功')
    except:
        print('解析文件中数据失败')

    # ========================================计算运动距离、速度、加速度=============================================

    # 输出固定序号生猪的质心运动列表
    # 数据格式：[[{frame},{x_center},{y_center}],...]
    center_move_list = Get_CenterMoveData_List(move_data_list)

    # 显示并保存标注轨迹点后背景图片
    Show_move_in_maskimg(center_move_list,
                         img_save_path=mask_result_img_save_path,
                         pig_id=pig_id,
                         color=(0, 0, 255))
    print(center_move_list)

    # 计算移动距离，获得每一帧下移动距离的列表
    move_distance_list = Get_MoveDistance_List(center_move_list)
    # print("生猪每一帧移动距离：")
    # print(move_distance_list)

    # 计算总运动距离
    total_distance = Get_total_distance(move_distance_list)
    print('序号为：' + str(pig_id) + '的生猪总移动距离为：' + str(total_distance) + '(m)')

    # 计算瞬时速度列表
    move_speed_list = Get_MoveSpeed_List(move_distance_list)
    # print("生猪每一帧瞬时速度：")
    # print(move_speed_list)

    # 计算总平均速度
    average_speed = Get_Average_Speed(move_distance_list, total_distance)
    print('序号为：' + str(pig_id) + '的生猪总移动速度为：' + str(average_speed) + '(m/s)')

    # 计算瞬时加速度列表
    move_acc_list = Get_Acc_List(move_speed_list)
    # print("生猪每一帧瞬时加速度：")
    # print(move_acc_list)

    # 计算总平均加速度
    average_acc = Get_Average_Acc(move_acc_list)
    print('序号为：' + str(pig_id) + '的生猪总移动加速度为：' + str(average_acc) + '(m/s2)')

    # 计算最大速度
    max_speed = Get_MaxValue(move_speed_list)
    print('序号为：' + str(pig_id) + '的生猪最大移动速度为：' + str(max_speed) + '(m/s)')

    # 计算最小速度
    max_acc   = Get_MaxValue(move_acc_list)
    print('序号为：' + str(pig_id) + '的生猪最大移动加速度为：' + str(max_acc) + '(m/s2)')

    # ============================================数据展示=======================================================

    # 显示固定id生猪的速度-时间折线图
    Show_Speed_Plot(average_speed_list=move_speed_list,
                    img_save_path=single_pig_motion_plot_save_path,
                    obj_id=pig_id)

    # 显示固定id生猪的加速度-时间折线图
    Show_Acc_Plot(move_acc_list=move_acc_list,
                  img_save_path=single_pig_motion_plot_save_path,
                  obj_id=pig_id)

    # 存入列表，后续对比保存
    average_distance_list.append(total_distance)
    average_speed_list.append(average_speed)
    average_acc_list.append(average_acc)
    max_speed_list.append(max_speed)
    max_acc_list.append(max_acc)

# ============================================数据统计===========================================================

# 打印数据
print("=============================================================================")
print("==================================打印统计数据================================")
print("平均距离列表：")
print(average_distance_list)
print("平均速度列表：")
print(average_speed_list)
print("平均加速度列表：")
print(average_acc_list)
print("最大速度列表：")
print(max_speed_list)
print("最大加速度列表：")
print(max_acc_list)

# 多只生猪运动数据柱状图对比
# 距离
Show_Motion_Conv(average_motioninfo_list= average_distance_list,
                 img_save_path = multi_pig_conv_histogram_save_path,
                 info_type = 'distance'
                 )
# 速度
Show_Motion_Conv(average_motioninfo_list= average_speed_list,
                 img_save_path = multi_pig_conv_histogram_save_path,
                 info_type = 'average_speed'
                 )
# 加速度
Show_Motion_Conv(average_motioninfo_list= average_acc_list,
                 img_save_path = multi_pig_conv_histogram_save_path,
                 info_type = 'accelerated_speed'
                 )

# 最大速度
Show_Motion_Conv(average_motioninfo_list= max_speed_list,
                 img_save_path = multi_pig_conv_histogram_save_path,
                 info_type = 'max_speed'
                 )

# 最大加速度
Show_Motion_Conv(average_motioninfo_list= max_acc_list,
                 img_save_path = multi_pig_conv_histogram_save_path,
                 info_type = 'max_acc'
                 )

# ============================================数据保存===========================================================
# 生猪运动数据列表
# 数据格式：
# [[{id},{average_distance},{average_speed},{average_acc},{max_speed},{max_acc}]
pig_mov_info_list = []

for i in range(1,9):
    # 生猪运动数据列表
    temp_pig_mov_info_list = []

    # 添加生猪序号
    temp_pig_mov_info_list.append(i)
    # 添加运动平均距离
    temp_pig_mov_info_list.append(average_distance_list[i])
    # 添加运动平均速度
    temp_pig_mov_info_list.append(average_speed_list[i])
    # 添加运动平均加速度
    temp_pig_mov_info_list.append(average_acc_list[i])
    # 添加运动最大速度
    temp_pig_mov_info_list.append(max_speed_list[i])
    # 添加运动最大加速度
    temp_pig_mov_info_list.append(max_acc_list[i])

    pig_mov_info_list.append(temp_pig_mov_info_list)

print(pig_mov_info_list)

# 写入txt文件
f = open(save_path, 'a')

for i in range(len(pig_mov_info_list)):
    for j in range(len(pig_mov_info_list[i])):
        # write函数不能写int类型的参数，所以使用str()转化
        f.write(str(pig_mov_info_list[i][j]))
        # 相当于Tab一下，换一个单元格
        f.write('\t')
        # 写完一行立马换行
    f.write('\n')
f.close()

print("保存文件成功")
