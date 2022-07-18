# 参考博客https://blog.csdn.net/leo0308/article/details/84960248
# 修改xml文件中的<path>，与实际数据路径对应
# coding=utf-8
import os
import os.path
import xml.dom.minidom

file_path = "F:\\pig_healthy\\code\\pig_detect\\YOLOV4\\VOCdevkit\\VOC2007\\Annotations"
files = os.listdir(file_path)  # 得到文件夹下所有文件名称
s = []
for xmlFile in files:  # 遍历文件夹
    if not os.path.isdir(xmlFile):  # 判断是否是文件夹,不是文件夹才打开
        print(xmlFile)
        # xml文件读取操作
        # 将获取的xml文件名送入到dom解析
        # 最核心的部分,路径拼接,输入的是具体路径
        dom = xml.dom.minidom.parse(os.path.join(file_path, xmlFile))
        root = dom.documentElement
        # 获取标签对path之间的值
        original_path = root.getElementsByTagName('path')
        # 原始信息
        p0=original_path[0]
        # 原始路径
        path0=p0.firstChild.data
        print(path0)
        # 修改
        # 获取图片名
        jpg_name=path0.split('\\')[-1]
        # 修改后path
        modify_path=file_path+'\\'+jpg_name
        p0.firstChild.data=modify_path
        print(modify_path)

        # 保存修改到xml文件中
        with open(os.path.join(file_path, xmlFile), 'w') as fh:
            dom.writexml(fh)
            print('修改path OK!')


