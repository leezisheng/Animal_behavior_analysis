# -*- coding: UTF-8 -*-
'''
@Project ：Animal_behavior_analysis , 调用摄像头采集视频数据并保存到文件夹中
@File    ：Data_Collection.py
@Author  ：leeqingshui
'''

import cv2
import numpy as np

# 创建VideoCapture的对象cap。传入的参数可以是设备索引1，也可以是自己本地的视频
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率

# 生成fourcc code
# 忽略报错
# OpenCV: FFMPEG: tag 0x00000021/'!???' is not found (format 'mp4 / MP4 (MPEG-4 Part 14)')'
fourcc = 0x00000021
# 需要保存，创建VideoWriter对象out
out = cv2.VideoWriter('F:\\Animal_behavior_analysis\\Dataset\\video_dataset\\01.avi', fourcc, 20.0, (640, 480), isColor=True)

# 确保cap打开了
if not cap.isOpened():
    print("cap is not opened, open the cap")
    cap.open()
    exit()
else:
    print('cap is opened, read the video stream...')

# 使用一个While循环不间断地对usb摄像头进行读取，一直到遇到键盘终止事件时break掉
while cap.isOpened():
    # 使用cap.read()从摄像头读取一帧
    ret, frame = cap.read()
    # 用read()返回的布尔值ret判断有没有正确读取到
    if not ret:
        print(' cannot receive frames(stream end?). Exiting...')
        break
    # frame = cv.flip()

    # 写入对象out调用write()写入这一帧
    out.write(frame)

    # 同时，把我们写入视频的这一帧显示出来，这样能实时看到我们处理和保存的内容
    cv2.imshow('frame', frame)
    # 等待1ms按键事件,如果未在规定时间按键，返回-1.如果在规定时间按键，返回所按键的ascII码值
    if cv2.waitKey(1) == ord('q'):
        break

    # 输出相关参数信息
    print('视频中的图像宽度{}'.format(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print('视频中的图像高度{}'.format(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print('视频帧率{}'.format(cap.get(cv2.CAP_PROP_FPS)))
    print('视频帧数{}'.format(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

# release你的cap对象和out对象
cap.release()
out.release()

# 销毁所有打开的HighGUI窗口。
cv2.destroyAllWindows()
