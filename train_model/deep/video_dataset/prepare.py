# 通过标注文件将目标检测数据集中待检测目标扣出
# 扣出后图像作为表观特征模型训练数据集

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import xml.dom.minidom
import argparse

def main():
    img_path = 'H:\\temp\\2019-11-22--11_20_15\\000001\\JPEGImages'+'\\'
    anno_path = 'H:\\temp\\2019-11-22--11_20_15\\000001\\Annotations'+'\\'
    cut_path = 'H:\\temp\\2019-11-22--11_20_15\\000001\\dataset'+'\\'
    
    if not os.path.exists(cut_path):
        os.makedirs(cut_path)
    imagelist = os.listdir(img_path)
    for image in imagelist:
        image_pre, ext = os.path.splitext(image)
        img_file = img_path + image
        img = cv2.imread(img_file)
        xml_file = anno_path + image_pre + '.xml'

        # 避免因为中间有的图片没有xml文件而停止
        try:
            tree = ET.parse(xml_file)
        except:
            print("图片%s不存在xml文件"%image)
            continue

        root = tree.getroot()
        obj_i = 0

        for obj in root.iter('object'):
            obj_i += 1
            cls = obj.find('name').text
            xmlbox = obj.find('bndbox')
            b = [int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
                 int(float(xmlbox.find('xmax').text)),
                 int(float(xmlbox.find('ymax').text))]
            img_cut = img[b[1]:b[3], b[0]:b[2], :]
            path = os.path.join(cut_path, cls)
            mkdirlambda = lambda x: os.makedirs(x) if not os.path.exists(x) else True
            mkdirlambda(path)
            cv2.imwrite(os.path.join(cut_path, cls, '{}_{:0>2d}.jpg'.format(image_pre, obj_i)), img_cut)
            print("&&&&")

if __name__ == '__main__':
    main()
