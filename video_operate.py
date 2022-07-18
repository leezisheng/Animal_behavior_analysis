# -*- coding: UTF-8 -*-
'''
@Project ：code 
@File    ：video_operate.py ，对视频图像与背景进行逻辑与运算
@Author  ：leeqingshui
@Date    ：2022/6/21 6:06 
'''

import cv2 as cv
import numpy as np

# 读取背景mask文件和video路径
binary_maskimg_path = 'F:\\pig_healthy\\code\\pig_data_processing\\mask_binary.jpg'
video_path           = 'F:\\pig_healthy\\code\\color.mp4'

# 读取图片
binary_mask_img = cv.imread(binary_maskimg_path,1)

video = cv.VideoCapture()
video.open(video_path)

# 忽略报错
# OpenCV: FFMPEG: tag 0x00000021/'!???' is not found (format 'mp4 / MP4 (MPEG-4 Part 14)')'
fourcc = 0x00000021

# 创建保存视频的对象，设置编码格式，帧率，图像的宽高等
out = cv.VideoWriter('F:\\pig_healthy\\code\\out_color.mp4', fourcc, 30, (1280, 720))

#判断是否成功创建视频流
while video.isOpened():

    ret,frame = video.read()
    print(ret)

    frame = cv.bitwise_and(frame, binary_mask_img)
    frame = cv.medianBlur(frame,5)

    out.write(frame)

    cv.imshow('video',frame)

    # 设置播放速速
    cv.waitKey(int(1000//video.get(cv.CAP_PROP_FPS)))

    #按下q键退出
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    #输出相关参数信息
    print('视频中的图像宽度{}'.format(video.get(cv.CAP_PROP_FRAME_WIDTH)))
    print('视频中的图像高度{}'.format(video.get(cv.CAP_PROP_FRAME_HEIGHT)))
    print('视频帧率{}'.format(video.get(cv.CAP_PROP_FPS)))
    print('视频帧数{}'.format(video.get(cv.CAP_PROP_FRAME_COUNT)))

#释放资源并关闭窗口
video.release()
out.release()
cv.destroyAllWindows()

