import cv2
import os
import glob
import numpy as np
import requests
import matplotlib
import matplotlib.pyplot as plt
import statistics

matplotlib.use('TkAgg')

# 图片序号从1开始 

# 图像数据集地址
dir_path="G:\\pig_healthy\\dataset\\original_dataset\\test_dataset"+"\\"
images_path = glob.glob(os.path.join(dir_path + '*.jpg')) #*.jpg中的*，表示能匹配多个字符

# 统计图像哈希值的列表
hash_list=[]
# 统计图像汉明距离的列表
value_list=[]
# 汉明距离平均值
average_value=0

# 利用感知哈希算法 比较图片相似度
# 计算哈希值
def pHash(img):
    # 感知哈希算法
    # 缩放32*32
    img = cv2.resize(img, (32, 32))   # , interpolation=cv2.INTER_CUBIC
 
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 将灰度图转为浮点型，再进行dct变换
    dct = cv2.dct(np.float32(gray))
    # opencv实现的掩码操作
    dct_roi = dct[0:8, 0:8]
 
    hash = []
    avreage = np.mean(dct_roi)
    for i in range(dct_roi.shape[0]):
        for j in range(dct_roi.shape[1]):
            if dct_roi[i, j] > avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash

def cmpHash(hash1, hash2):
    # Hash值对比
    # 算法中1和0顺序组合起来的即是图片的指纹hash。顺序不固定，但是比较的时候必须是相同的顺序。
    # 对比两幅图的指纹，计算汉明距离，即两个64位的hash值有多少是不一样的，不同的位数越小，图片越相似
    # 汉明距离：一组二进制数据变成另一组数据所需要的步骤，可以衡量两图的差异，汉明距离越小，则相似度越高。汉明距离为0，即两张图片完全一样
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n + 1
    return n

# 统计汉明距离和图像相似比例的关系
def static(value_list,num):
    value_list=value_list.copy()
    for i in range(len(value_list)):
        if value_list[i] < num :
            value_list[i] = 0
        else:
            value_list[i] = 1
    temp=value_list.count(0)/len(value_list)
    return temp



#########################################################################
#########################################################################
# 求解平均汉明距离，作为衡量相似度的阈值

# 求所有图片哈希值
for i in images_path:
    img_temp=cv2.imread(i)
    hash_temp=pHash(img_temp)
    hash_list.append(hash_temp)

# print(len(hash_list))

# 挨个对比，求平均汉明距离
for i in range(len(hash_list)) :
    if i>0:
        for j in range(len(hash_list)-i) :
            if j>0:
                temp_value=cmpHash(hash_list[i],hash_list[i+j])
                value_list.append(temp_value)

# 汉明距离平均值结果为24
average_value=statistics.mean(value_list)

print(average_value)
print(len(value_list))
temp_list=[]

# 统计汉明距离和图像相似比例的关系
# 24是距离变化最快的点
for num in range(64):
    temp=static(value_list,num)
    print(temp)
    temp_list.append(temp)

plt.plot(temp_list)
plt.show()
print(value_list.count(0)/len(value_list))

#########################################################################
#########################################################################



#########################################################################
#########################################################################
# 开始去除汉明距离在阈值下的图片，认定为过分相似
# 对于最后一张 没有办法 手动判断

# # 求所有图片哈希值
# for i in images_path:
#     img_temp=cv2.imread(i)
#     hash_temp=pHash(img_temp)
#     hash_list.append(hash_temp)

# # 挨个对比，求汉明距离
# for i in range(len(hash_list)+1) :
#     if i>0:
#         print("picture index:",i)
#         for j in range(len(hash_list)-i) :
#             if j>0:  
#                 temp_value=cmpHash(hash_list[i],hash_list[i+j])
#                 if temp_value < 10:
#                     print("same picture index:",i+j)
#                     # 删除图片
#                     remove_path=dir_path+str(i+j).zfill(4)+".jpg" # zfill补零
#                     # print(remove_path)
#                     if os.path.exists(remove_path): # 如果文件存在
#                         os.remove(remove_path)
#                     else:
#                         print('no such file:%s'%remove_path)

#########################################################################
#########################################################################