# -*- coding: UTF-8 -*-
'''
@Project ：code
@File    ：feature_correlation_calculation.py ， 计算特征之间相关系数部分
@Author  ：leeqingshui
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''
@ 函数功能                    ：分析相关系数，并对相关系数矩阵进行作热力图
@ 入口参数 {str}   csv_path   ：数据集导入位置
@ 入口参数 {str}   save_path  ：热力图保存位置
'''
def correlation_calculation(csv_path, save_path):
    # 读取数据
    data = pd.read_csv(csv_path,usecols=[1,2,3])
    # 提取需要计算相关系数的列
    a = data.corr()
    # 作图
    plt.subplots(figsize=(3, 3))
    sns.heatmap(a, square=True)
    plt.savefig(save_path)
    plt.show()
