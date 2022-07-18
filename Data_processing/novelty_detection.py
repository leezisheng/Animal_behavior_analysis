# -*- coding: UTF-8 -*-
'''
@Project ：code 
@File    ：novelty_detection.py ， 异常检测部分
@Author  ：leeqingshui
@Date    ：2022/7/7 3:13 
'''

import pickle
import numpy as np
from math import ceil
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# 数据预处理
def pre_scaler(dataset, type_str="std"):
    if type_str == "minmax":
        scaler = MinMaxScaler()
    elif type_str == "std":
        scaler = StandardScaler()
    else:
        return None
    scaler.fit(dataset)
    return scaler, scaler.transform(dataset)

# 数据集拆分
def train_test_split(dataset, test_ratio=0.1, seed=42):
    if seed:
        np.random.seed(seed)
    shuffle_index = np.random.permutation(len(dataset))
    test_size = ceil(len(dataset) * test_ratio)
    test_index = shuffle_index[:test_size]
    train_index = shuffle_index[test_size:]
    dataset_train = dataset[train_index]
    dataset_test = dataset[test_index]
    return dataset_train, dataset_test

class One_SVM:
    '''
    @ 函数作用                      ：初始化
    @ 输入参数 {list} train_dataset : 数据集列表
    @ 输入参数 {str}  model         : 训练后模型
    '''

    def __init__(self, train_dataset, model=''):
        # 数据集列表
        self.train_dataset = train_dataset
        # 模型
        self.model   = model

    '''
    @ 函数作用                      ：模型训练
    @ 输入参数 {str} save_model_dir : 模型保存位置
    @ 返回参数 {pkl} model          : 训练好的模型
    '''
    def train_model(self, save_model_dir=''):

        self.model = svm.OneClassSVM(nu=0.05, kernel="rbf", gamma="auto")

        print("==============================================开始训练==============================================")
        self.model.fit(self.train_dataset)

        print("==============================================保存模型==============================================")
        try:
            output = open(save_model_dir + 'novelty_detect_model.pkl', 'wb')
            pickle.dump(self.model, output)
            print("保存模型成功，保存位置：" + save_model_dir + 'novelty_detect_model.pkl')
        except:
            print("保存模型失败")

        return self.model

    '''
    @ 函数作用                          : 模型加载和模型预测
    @ 输入参数 {list} predict_x_dataset : 数据集列表
    @ 返回参数 {list} predict_y_list    : 预测数据列表
    '''
    def predict_model(self, predict_x_dataset, load_model_dir=''):

        if load_model_dir != '':
            print("==========================================加载模型==========================================")
            try:
                model_file = open(str(load_model_dir) + '\\' + 'novelty_detect_model.pkl', 'rb')
                self.model = pickle.load(model_file)
                print("==========================================加载模型成功==========================================")
                predict_y_list = self.model.predict(predict_x_dataset)
            except:
                print("==========================================加载模型失败==========================================")

        else:
            print("==========================================使用文件模型==========================================")
            try:
                predict_y_list = self.model.predict(predict_x_dataset)
            except:
                print("==========================================文件模型错误==========================================")

        return predict_y_list

