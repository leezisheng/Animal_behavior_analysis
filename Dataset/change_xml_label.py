# -*- coding: UTF-8 -*-
# 参考博客https://blog.csdn.net/DD_PP_JJ/article/details/102772793
# 修改xml文件中的<name>，原标签值为 “标签值+编号”
# 注意仅适用于单类别目标检测和目标追踪

'''
@Project ：Animal_behavior_analysis 
@File    ：change_xml_label.py ， 修改xml文件中的<name>，原标签值为 “标签值+编号”
@Author  ：leeqingshui
@Date    ：2022/7/30 1:18 
'''

import os
import os.path
from xml.etree.ElementTree import parse, Element

def changeAll(xml_fold,new_name):
    '''
    xml_fold: xml存放文件夹
    new_name: 需要改成的正确的名字，在上个例子中就是cow
    '''
    files = os.listdir(xml_fold)
    cnt = 0
    for xmlFile in files:
        file_path = os.path.join(xml_fold, xmlFile)
        dom = parse(file_path)
        root = dom.getroot()
        for obj in root.iter('object'):
            #获取object节点中的name子节点
            tmp_name = obj.find('name').text
            obj.find('name').text = new_name
            print("change %s to %s." % (tmp_name, new_name))
            cnt += 1
        dom.write(file_path, xml_declaration=True)
        #保存到指定文件
    print("有%d个文件被成功修改。" % cnt)

if __name__ == '__main__':
    # xml文件所在的目录
    path = r"F:\\Animal_behavior_analysis\\Dataset\\img_dataset\\Aug_Annotations"
    # 要修改的标签名字
    label_names = "mouse"
    changeAll(path, label_names)



