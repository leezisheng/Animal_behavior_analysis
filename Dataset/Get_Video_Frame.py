# -*- coding: UTF-8 -*-
'''
@Project ：Animal_behavior_analysis
@File    ：Get_Video_Frame.py , 视频抽帧
@Author  ：leeqingshui
'''
import cv2
from PIL import Image
import numpy as np

cap = cv2.VideoCapture("F:\\Animal_behavior_analysis\\Dataset\\video_dataset\\03\\03.mp4")  # 获取视频对象
isOpened = cap.isOpened  # 判断是否打开
# 视频信息获取
fps = cap.get(cv2.CAP_PROP_FPS)

imageNum = 1141
sum=0
#隔10帧保存一张图片
timef=10

img_save_path = "F:\\Animal_behavior_analysis\\Dataset\\img_dataset\\JPEGImages"+"\\"

while (isOpened):

    sum+=1

    (frameState, frame) = cap.read()  # 记录每帧及获取状态

    if frameState == True and (sum % timef==0):

        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame = Image.fromarray(np.uint8(frame))

        frame = np.array(frame)

        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        imageNum = imageNum + 1
        fileName = img_save_path + str(imageNum) + '.jpg'  # 存储路径
        cv2.imwrite(fileName, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        print(fileName + " successfully write in")  # 输出存储状态

    elif frameState == False:
        break

print('finish!')
cap.release()



