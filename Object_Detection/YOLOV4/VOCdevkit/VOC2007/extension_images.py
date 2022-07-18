# 使用imgaug库进行数据增强
# 参考博客 https://blog.csdn.net/pw1623/article/details/88683270
import xml.etree.ElementTree as ET
import pickle
import os
from os import getcwd
import numpy as np
from PIL import Image
import shutil
import matplotlib.pyplot as plt

import imgaug as ia
from imgaug import augmenters as iaa

ia.seed(1)

def read_xml_annotation(root, image_id):
    in_file = open(os.path.join(root, image_id), encoding='UTF-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    bndboxlist = []

    for object in root.findall('object'):  # 找到root节点下的所有country节点
        bndbox = object.find('bndbox')  # 子节点下节点rank的值

        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        # print(xmin,ymin,xmax,ymax)
        bndboxlist.append([xmin, ymin, xmax, ymax])
        # print(bndboxlist)

    # ndbox = root.find('object').find('bndbox')
    return bndboxlist


# (506.0000, 330.0000, 528.0000, 348.0000) -> (520.4747, 381.5080, 540.5596, 398.6603)
def change_xml_annotation(root, image_id, new_target):
    new_xmin = new_target[0]
    new_ymin = new_target[1]
    new_xmax = new_target[2]
    new_ymax = new_target[3]

    in_file = open(os.path.join(root, str(image_id) + '.xml'), encoding='UTF-8')  # 这里root分别由两个意思
    tree = ET.parse(in_file)
    xmlroot = tree.getroot()
    object = xmlroot.find('object')
    bndbox = object.find('bndbox')
    xmin = bndbox.find('xmin')
    xmin.text = str(new_xmin)
    ymin = bndbox.find('ymin')
    ymin.text = str(new_ymin)
    xmax = bndbox.find('xmax')
    xmax.text = str(new_xmax)
    ymax = bndbox.find('ymax')
    ymax.text = str(new_ymax)
    tree.write(os.path.join(root, str("%06d" % (str(id) + '.xml'))))


def change_xml_list_annotation(root, image_id, new_target, saveroot, id):
    in_file = open(os.path.join(root, str(image_id) + '.xml'), encoding='UTF-8')  # 这里root分别由两个意思
    tree = ET.parse(in_file)
    elem = tree.find('filename')
    elem.text = (str("%06d" % int(id)) + '.jpg')
    xmlroot = tree.getroot()
    index = 0

    for object in xmlroot.findall('object'):  # 找到root节点下的所有country节点
        bndbox = object.find('bndbox')  # 子节点下节点rank的值

        # xmin = int(bndbox.find('xmin').text)
        # xmax = int(bndbox.find('xmax').text)
        # ymin = int(bndbox.find('ymin').text)
        # ymax = int(bndbox.find('ymax').text)

        new_xmin = new_target[index][0]
        new_ymin = new_target[index][1]
        new_xmax = new_target[index][2]
        new_ymax = new_target[index][3]

        xmin = bndbox.find('xmin')
        xmin.text = str(new_xmin)
        ymin = bndbox.find('ymin')
        ymin.text = str(new_ymin)
        xmax = bndbox.find('xmax')
        xmax.text = str(new_xmax)
        ymax = bndbox.find('ymax')
        ymax.text = str(new_ymax)

        index += 1

        print("index=", index)

    tree.write(os.path.join(saveroot, str("%06d" % int(id)) + '.xml'))


def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False


if __name__ == "__main__":

    IMG_DIR = "F:\\pig_healthy\\数据集\\目标检测数据集\\image_dataset\\JPEGImages"
    XML_DIR = "F:\\pig_healthy\\数据集\\目标检测数据集\\image_dataset\\Annotations"

    AUG_XML_DIR = "F:\\pig_healthy\\数据集\\目标检测数据集\\image_dataset\\AUG_Annotations"  # 存储增强后的XML文件夹路径
    try:
        shutil.rmtree(AUG_XML_DIR)
    except FileNotFoundError as e:
        a = 1
    mkdir(AUG_XML_DIR)

    AUG_IMG_DIR = "F:\\pig_healthy\\数据集\\目标检测数据集\\image_dataset\\AUG_JPEGImages"  # 存储增强后的影像文件夹路径
    try:
        shutil.rmtree(AUG_IMG_DIR)
    except FileNotFoundError as e:
        a = 1
    mkdir(AUG_IMG_DIR)

    AUGLOOP = 4  # 每张影像增强的数量

    boxes_img_aug_list = []
    new_bndbox = []
    new_bndbox_list = []

    # 影像增强
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # 镜像，50%的照片做镜像处理
        iaa.Flipud(0.5),
        iaa.ContrastNormalization((0.75, 1.5), per_channel=True),  ####0.75-1.5随机数值为alpha，对图像进行对比度增强，该alpha应用于每个通道
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1 * 255), per_channel=0.5),
        #### loc 噪声均值，scale噪声方差，50%的概率，对图片进行添加白噪声并应用于每个通道


        iaa.Multiply((0.8, 1.2), per_channel=0.2),  ####20%的图片像素值乘以0.8-1.2中间的数值,用以增加图片明亮度或改变颜色

    ])

    for root, sub_folders, files in os.walk(XML_DIR):

        for name in files:

            bndbox = read_xml_annotation(XML_DIR, name)
            shutil.copy(os.path.join(XML_DIR, name), AUG_XML_DIR)
            shutil.copy(os.path.join(IMG_DIR, name[:-4] + '.jpg'), AUG_IMG_DIR)
            print(os.path.join(IMG_DIR, name[:-4] + '.jpg'))

            for epoch in range(AUGLOOP):
                seq_det = seq.to_deterministic()  # 保持坐标和图像同步改变，而不是随机
                # 读取图片
                img = Image.open(os.path.join(IMG_DIR, name[:-4] + '.jpg'))
                # sp = img.size
                img = np.asarray(img)
                # bndbox 坐标增强
                for i in range(len(bndbox)):
                    bbs = ia.BoundingBoxesOnImage([
                        ia.BoundingBox(x1=bndbox[i][0], y1=bndbox[i][1], x2=bndbox[i][2], y2=bndbox[i][3]),
                    ], shape=img.shape)

                    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
                    boxes_img_aug_list.append(bbs_aug)

                    # new_bndbox_list:[[x1,y1,x2,y2],...[],[]]
                    n_x1 = int(max(1, min(img.shape[1], bbs_aug.bounding_boxes[0].x1)))
                    n_y1 = int(max(1, min(img.shape[0], bbs_aug.bounding_boxes[0].y1)))
                    n_x2 = int(max(1, min(img.shape[1], bbs_aug.bounding_boxes[0].x2)))
                    n_y2 = int(max(1, min(img.shape[0], bbs_aug.bounding_boxes[0].y2)))
                    if n_x1 == 1 and n_x1 == n_x2:
                        n_x2 += 1
                    if n_y1 == 1 and n_y2 == n_y1:
                        n_y2 += 1
                    if n_x1 >= n_x2 or n_y1 >= n_y2:
                        print('error', name)
                    new_bndbox_list.append([n_x1, n_y1, n_x2, n_y2])

                    # 存储变化后的图片
                    image_aug = seq_det.augment_images([img])[0]
                    path = os.path.join(AUG_IMG_DIR, str("%06d" % (len(files) + int(name[:-4]) + epoch * 250)) + '.jpg')

                    image_auged = bbs.draw_on_image(image_aug, thickness=0)  ####################################

                    Image.fromarray(image_auged).save(path)

                # 存储变化后的XML
                change_xml_list_annotation(XML_DIR, name[:-4], new_bndbox_list, AUG_XML_DIR,
                                           len(files) + int(name[:-4]) + epoch * 250)
                print(str("%06d" % (len(files) + int(name[:-4]) + epoch * 250)) + '.jpg')
                new_bndbox_list = []

