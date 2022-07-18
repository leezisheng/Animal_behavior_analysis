# 把 xml 中的目标框在原图上绘制出来 ，并显示标签，并且将附带标注的图片保存到指定文件夹中
# 标注绘制部分参考博客      ：  https://blog.csdn.net/qq_36758461/article/details/103947168
# PIL库文件保存部分参考博客 ：  https://blog.csdn.net/m0_59821641/article/details/119857379
# 读取xml。
import xml.etree.ElementTree as ET
import os
from PIL import Image,ImageDraw,ImageFont
import matplotlib.pyplot as plt

def parse_rec(filename):
    tree = ET.parse(filename)  # 解析读取xml函数
    objects = []
    img_dir = []
    for xml_name in tree.findall('filename'):
        img_path = os.path.join(pic_path, xml_name.text)
        img_dir.append(img_path)
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects, img_dir

# 可视化
def visualise_gt(objects, img_dir , filename):
    for id, img_path in enumerate(img_dir):
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        for a in objects:
            xmin = int(a['bbox'][0])
            ymin = int(a['bbox'][1])
            xmax = int(a['bbox'][2])
            ymax = int(a['bbox'][3])
            label = a['name']
            draw.rectangle((xmin, ymin, xmax, ymax), fill=None, outline=(0, 255, 0), width=2)
            draw.text((xmin - 10, ymin - 15), label, fill=(0, 255, 0), font=font)  # 利用ImageDraw的内置函数，在图片上写入文字
        # 实时显示图片，一张打开后手动关闭显示下一张图片
        # img.show()
        save_img_path = save_root + "\\" + str(os.path.splitext(str(filename))[0])+".jpg"
        # print(str(os.path.splitext(str(filename))[0]))
        try:
            img.save(save_img_path, quality=95)
            print('图片保存成功，保存在目录：' + save_img_path + ".jpg" + "\n")
        except:
            print('图片:'+str(os.path.splitext(str(filename))[0])+'保存失败' + "\n")

# 字体路径
fontPath = "C:\\Windows\\Fonts\\Consolas\\consola.ttf"  
# 图片img和标注文件的根目录
root = 'F:\\pig_healthy\\code\\pig_detect\\YOLOV4\\VOCdevkit\\VOC2007'
# xml文件所在路径
ann_path = os.path.join(root, 'Annotations')  
# 样本图片路径
pic_path = os.path.join(root, 'JPEGImages')
# Image库选用字体
font = ImageFont.truetype(fontPath, 16)       
# 带标注图片保存文件夹路径
save_root = "F:\\pig_healthy\\code\\pig_detect\\YOLOV4\\VOCdevkit\\VOC2007\\vision_xml"

for filename in os.listdir(ann_path):
    xml_path = os.path.join(ann_path, filename)
    object, img_dir = parse_rec(xml_path)
    visualise_gt(object, img_dir , filename)
