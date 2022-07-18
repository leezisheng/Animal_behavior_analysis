# 批量修改文件编码，例如从ansi转为utf-8
# 使用时需注意39行 extension修改为需要转码的文件的后缀
# 如本文件中转换xml文件
# if not (extension == '.xml'):
# 参考博客：https://www.cnblogs.com/grooovvve/p/14881969.html 

import os
import sys
import codecs
import chardet
 
def get_file_extension(file):
    (filepath, filename) = os.path.split(file)
    (shortname, extension) = os.path.splitext(filename)
    return extension
 
def get_file_encode(filename):
    with open(filename, 'rb') as f:
        data = f.read()
        encoding_type = chardet.detect(data)
        # print(encoding_type)
 
    return encoding_type
 
def process_dir(root_path):
    for path, dirs, files in os.walk(root_path):
        for file in files:
            file_path = os.path.join(path, file)
            process_file(file_path, file_path)
 
def process_file(filename_in, filename_out):
    """
    filename_in :输入文件(全路径+文件名)
    filename_out :保存文件(全路径+文件名)
    文件编码类型: 'windows-1251','UTF-8-SIG'
    """
    extension = get_file_extension(filename_in).lower()
    if not (extension == '.xml'):
        return
 
    # 输出文件的编码类型
    dest_file_encode = 'utf-8'
    encoding_type = get_file_encode(filename_in)
    src_file_encode = encoding_type['encoding']
    if src_file_encode == 'utf-8':
        return
    elif src_file_encode is None:
        src_file_encode = 'windows-1251'
 
    print("[Convert]File:" + filename_in + " from:" + encoding_type['encoding'] + " to:UTF-8")
 
    try:
        with codecs.open(filename=filename_in, mode='r', encoding=src_file_encode) as fi:
            data = fi.read()
            with open(filename_out, mode='w', encoding=dest_file_encode) as fo:
                fo.write(data)
                fo.close()
 
        with open(filename_out, 'rb') as f:
            data = f.read()
            print(chardet.detect(data))
    except Exception as e:
        print(e)
 
def dump_file_encode(root_path):
    for path, dirs, files in os.walk(root_path):
        for file in files:
            filename = os.path.join(path, file)
            with open(filename, 'rb') as f:
                data = f.read()
                encoding_type = chardet.detect(data)
                print("FILE:" + file + " ENCODE:" + str(encoding_type))
 
def convert(path):
    """
    批量转换文件编码格式
    path :输入文件或文件夹
    """
    # sys.argv[1], sys.argv[2]
    if os.path.isfile(path):
        process_file(path, path)
    elif os.path.isdir(path):
        process_dir(path)
 
if __name__ == '__main__':
    convert(r"F:\\pig_healthy\\code\\pig_detect\\YOLOV4\\VOCdevkit\\VOC2007\\Annotations")
    dump_file_encode(r'C:\\Users\\Administrator\\Desktop\\cc')