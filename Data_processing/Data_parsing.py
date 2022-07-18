# -*- coding: UTF-8 -*-
'''
@Project ：code 
@File    ：Data_parsing.py , 对output文件夹下保存生猪运动点的txt文件进行解析
@Author  ：leeqingshui
@Date    ：2022/6/21 2:36 
'''

# 测试txt文件路径
test_txt_path = "F:\\pig_healthy\\code\\output\\results.txt"

'''
@ 函数功能                  ：读取txt文件中每一行的数据
@ 入口参数 {str}  file_name ：txt文件路径
@ 返回参数 {list} data      ：存放每一行数据的列表，为二维列表，列表中元素为字符串
'''
def openreadtxt(file_name):
    data = []
    # 打开文件
    file = open(file_name,'r')
    # 读取所有行
    file_data = file.readlines()
    for row in file_data:
        # 去掉列表中每一个元素的换行符
        row = row.strip('\n')
        # 按‘，’切分每行的数据
        tmp_list = row.split(' ')
        # 将每行数据插入data列表中
        data.append(tmp_list)
    return data

'''
@ 函数功能                        ： 将原行数据列表中元素，从字符串格式转换为一维列表格式
@                                   原格式：[[''],[''],[''],[''],[''],[''],[''],['']]
@                                   转换后格式：[[int,int,int,....],................]
@ 入口参数 {list}  line_list      ： 每一行的数据列表，列表格式为字符串
@                                   数据格式为'{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1'
@ 返回参数 {list}  line_data_list ： 存放每一行数据的二维列表，列表中元素为一维列表
@                                   格式[[int,int,int,....],................]
'''
def Data_format_trans(line_list):

    # 存放每一行数据的二维列表，列表中元素为二维列表
    line_data_list = []

    # data和line_list均为列表
    for data in line_list:

        temp_data_list = []

        # 将列表中元素提取，提取后data为字符串
        data = data[0]

        # 对拆解后的字符列表生成元素索引序列
        # 将索引序列转换为索引列表
        temp_line_list = list(enumerate(data.split(",")))

        # 开始转换
        for i, element in temp_line_list:
            temp_data_list.append(int(element))

        line_data_list.append(temp_data_list)

    return line_data_list

'''
@ 函数功能                         : 查看id号代表的某头猪的运动轨迹列表
@ 入口参数 {list}  line_data_list  : 存放每一行数据的二维列表，列表中元素为一维列表
@                                   数据格式为[[{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1],...]
@ 返回参数 {list}  move_data_list  : 存放对应id号生猪的运动轨迹数据的列表
'''
def Get_move_data(line_data_list , id):
    # 存放运动轨迹数据的列表
    move_data_list = []

    # data为一维列表，line_data_list为二维列表
    for data in line_data_list:
        # print(data)
        if data[1] == id:
            print(data)
            move_data_list.append(data)

    return move_data_list

if __name__=="__main__":
    # 读取txt文件中每一行数据
    # 保存到line_data_list中
    line_data_list = openreadtxt(test_txt_path)
    # print(line_data_list)

    # 对行数据格式进行转换
    line_data_list = Data_format_trans(line_data_list)
    # print(line_data_list)

    # 查看id号代表的某头猪的运动轨迹数据列表
    move_data_list = Get_move_data(line_data_list , 1)
