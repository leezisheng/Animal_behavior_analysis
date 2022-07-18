# -*- coding: UTF-8 -*-
'''
@Project ：code
@File    ：Data_PCA.py , 数据降维
@Author  ：leeqingshui
'''

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

'''
@ 函数功能                                 ：PCA降维度
@ 入口参数 {list}   average_distance_list  ：生猪运动距离数据列表
@ 入口参数 {list}   average_speed_list     ：生猪运动速度数据列表
@ 入口参数 {list}   average_acc_list       ：生猪运动加速度数据列表
@ 返回参数 {list}   X_list                 ：降维后列表
'''
def Data_PCA_Operation(average_distance_list , average_speed_list, average_acc_list):
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

    # 对数据进行标准化处理
    X_std = StandardScaler().fit_transform(save_list)
    # 实例化PCA
    pca = PCA(n_components=2)
    # 训练数据
    pca.fit(X_std)

    # 使用PCA的属性查看特征值
    print(pca.singular_values_)
    # 使用PCA的属性查看特征值对应的特征向量
    print(pca.components_)

    # 对原始的数据集进行转换 , X_list为转换后列表
    X_list = np.array(save_list).dot(pca.components_.T)

    return X_list


