# 对划分好了数据集，生成label
# -*-coding:utf-8-*-
import os
import os.path

def write_txt(content, filename, mode='w'):
    """保存txt数据
    :param content:需要保存的数据,type->list
    :param filename:文件名
    """
    with open(filename, mode) as f:
        for line in content:
            str_line = ""
            for col, data in enumerate(line):
                if not col == len(line) - 1:
                    # 以空格作为分隔符
                    str_line = str_line + str(data) + " "
                else:
                    # 每行最后一个数据用换行符“\n”
                    str_line = str_line + str(data) + "\n"
            f.write(str_line)

def get_files_list(dir):
    '''
    实现遍历dir目录下,所有文件(包含子文件夹的文件)
    :param dir:指定文件夹目录
    :return:包含所有文件的列表->list
    '''
    # parent:父目录, filenames:该目录下所有文件夹,filenames:该目录下的文件名
    # 写入文件的数据
    files_list = []
    for parent, dirnames, filenames in os.walk(dir):
        for filename in filenames:

            print("parent is: " + parent)
            print("filename is: " + filename)
            # 输出rootdir路径下所有文件（包含子文件）信息
            print(os.path.join(parent, filename).replace('\\','/'))

            # 获取正在遍历的文件夹名（也就是类名）
            curr_file = parent.split(os.sep)[-1]

            #根据class名确定labels
            if curr_file == "pig0":
                labels = 0
            elif curr_file == "pig1":
                labels = 1
            elif curr_file == "pig2":
                labels = 2
            elif curr_file == "pig3":
                labels = 3
            elif curr_file == "pig4":
                labels = 4
            elif curr_file == "pig5":
                labels = 5
            elif curr_file == "pig6":
                labels = 6
            elif curr_file == "pig7":
                labels = 7

            dir_path = parent.replace('\\','/').split('/')[-2]

            curr_file = os.path.join(dir_path, curr_file)

            # 绝对路径
            curr_file = os.path.join(root_path, curr_file)

            # 相对路径+label
            files_list.append([os.path.join(curr_file, filename).replace('\\','/'), labels])

    return files_list

if __name__ == '__main__':

    root_path='F:/pig_healthy_temp/DeepSORT-YOLOv4-master/pytorch-yolov4-deepsort-main/deep_sort/deep/data'+'/'

    train_dir = './data/train'
    train_txt = './data/train.txt'
    train_data = get_files_list(train_dir)
    write_txt(train_data, train_txt, mode='w')

    test_dir = './data/test'
    test_txt = './data/test.txt'
    test_data = get_files_list(test_dir)
    write_txt(test_data, test_txt, mode='w')


