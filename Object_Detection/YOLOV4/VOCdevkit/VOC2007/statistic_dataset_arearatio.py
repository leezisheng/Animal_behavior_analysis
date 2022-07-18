# -*- coding:utf-8 -*-
# 参考博客：https://blog.csdn.net/Vertira/article/details/121792075
# 根据xml文件统计目标的平均长度、宽度、面积以及每一个目标在原图中的占比

#统计
# 计算每一个目标在原图中的占比
# 计算目标的平均长度、
# 计算平均宽度，
# 计算平均面积、
# 计算目标平均占比
 
import os
import xml.etree.ElementTree as ET
import numpy as np
 
np.set_printoptions(suppress=True, threshold=10000000)  #10,000,000
import matplotlib
from PIL import Image
 
def parse_obj(xml_path, filename):
    tree = ET.parse(xml_path + filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)
    return objects
 
def read_image(image_path, filename):
    im = Image.open(image_path + filename)
    W = im.size[0]
    H = im.size[1]
    area = W * H
    im_info = [W, H, area]
    return im_info
 
if __name__ == '__main__':
    image_path = 'F:\\pig_healthy\\code\\pig_detect\\YOLOV4\\VOCdevkit\\VOC2007\\JPEGImages'+'\\'
    xml_path   = 'F:\\pig_healthy\\code\\pig_detect\\YOLOV4\\VOCdevkit\\VOC2007\\Annotations'+'\\'
    filenamess = os.listdir(xml_path)
    filenames = []
    for name in filenamess:
        name = name.replace('.xml', '')
        filenames.append(name)
    print(filenames)
    recs = {}
    ims_info = {}
    obs_shape = {}
    classnames = []
    num_objs={}
    obj_avg = {}
    for i, name in enumerate(filenames):
        print('正在处理 {}.xml '.format(name))
        recs[name] = parse_obj(xml_path, name + '.xml')
        print('正在处理 {}.jpg '.format(name))
        ims_info[name] = read_image(image_path, name + '.jpg')
    print('所有信息收集完毕。')
    print('正在处理信息......')
    for name in filenames:
        im_w = ims_info[name][0]
        im_h = ims_info[name][1]
        im_area = ims_info[name][2]
        for object in recs[name]:
            if object['name'] not in num_objs.keys():
                num_objs[object['name']] = 1
            else:
                num_objs[object['name']] += 1
            #num_objs += 1
            ob_w = object['bbox'][2] - object['bbox'][0]
            ob_h = object['bbox'][3] - object['bbox'][1]
            ob_area = ob_w * ob_h
            w_rate = ob_w / im_w
            h_rate = ob_h / im_h
            area_rate = ob_area / im_area
            if not object['name'] in obs_shape.keys():
                obs_shape[object['name']] = ([[ob_w,
                                               ob_h,
                                               ob_area,
                                               w_rate,
                                               h_rate,
                                               area_rate]])
            else:
                obs_shape[object['name']].append([ob_w,
                                                  ob_h,
                                                  ob_area,
                                                  w_rate,
                                                  h_rate,
                                                  area_rate])
        if object['name'] not in classnames:
            classnames.append(object['name'])  # 求平均
 
    for name in classnames:
        obj_avg[name] = (np.array(obs_shape[name]).sum(axis=0)) / num_objs[name]
        print('{}的情况如下：*******\n'.format(name))
        print('  目标平均W={}'.format(obj_avg[name][0]))
        print('  目标平均H={}'.format(obj_avg[name][1]))
        print('  目标平均area={}'.format(obj_avg[name][2]))
        print('  目标平均与原图的W比例={}'.format(obj_avg[name][3]))
        print('  目标平均与原图的H比例={}'.format(obj_avg[name][4]))
        print('  目标平均原图面积占比={}\n'.format(obj_avg[name][5]))
    print('信息统计计算完毕。')