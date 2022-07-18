# python断言相关，用于程序调试
from os import environ

# 判断文件是否在指定的文件夹中
def assert_in(file, files_to_check):
    if file not in files_to_check:
        raise AssertionError("{} does not exist in the list".format(str(file)))
    return True

# 对编程环境中，每一个文件夹中应有的文件进行检查
def assert_in_env(check_list: list):
    for item in check_list:
        assert_in(item, environ.keys())
    return True
