from Object_Detection.YOLOV4 import yolo
from Object_Detection.YOLOV4.yolo import YOLO

# python模块中的__all__，用于模块导入时限制，如：from module import *
# 此时被导入模块若定义了__all__属性，则只有__all__内指定的属性、方法、类可被导入；
# 若没定义，则导入模块内的所有公有属性，方法和类。
# __all__ 的限制后，外部使用import *导入的时候，只有‘build_detector'方法可以被引用
__all__ = ['build_detector']

def build_detector():
    return YOLO()
