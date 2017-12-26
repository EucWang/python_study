from os import listdir

import numpy as np
import kNN_test.kNN as knn


def img2vector(filename):
    """读取单个文件,将图片信息,转换成1行32*32 列的矩阵数据数据"""
    try:
        with open(filename) as f:

            result_vect = np.zeros((1, 1024))
            for i in range(32):
                line_str = f.readline()
                for j in range(32):
                    result_vect[0, 32 * i + j] = int(line_str[j])
    except BaseException as e:
        print(e)
    else:
        return result_vect


def load_digits_file_data(digits_dir):
    """从数据集中加载数据
        返回三个字段
        @:return mat:    数据集, nums行1024列的矩阵
        @:return labels: 数据对应的标签
        @:return nums:   数据集的数量
    """
    # training_dir = 'trainingDigit'

    # 加载目录,获取目录下的文件列表
    training_files = listdir(digits_dir)
    # 获取文件的数量
    file_nums = len(training_files)
    # 生成一个样本数据集的矩阵, 每一条数据有1024个特征
    digits_result_mat = np.zeros((file_nums, 1024))

    hw_labels = []

    for i in range(file_nums):
        # 遍历, 获取文件名称
        file_name_str = training_files[i]

        # int((file_name_str.split('.')[0]).split('_')[0])
        # 获取文件名称的第一个字符,这个就是这个文件里保存的图像数据对应的数字
        class_num_str = int(file_name_str[0])

        # 获取第i个数据的标签
        hw_labels.append(class_num_str)
        # 获取第i个数据的数据矩阵
        digits_result_mat[i, :] = img2vector(digits_dir + "/" + file_name_str)

    return digits_result_mat, hw_labels, file_nums


def hw_test():
    error_count = 0.0
    training_mat, training_labels, training_num = load_digits_file_data('trainingDigits')
    test_mat, test_labels, test_num = load_digits_file_data('testDigits')

    for i in range(test_num):
        classifier_result = knn.classify0(test_mat[i], training_mat, training_labels, 3)
        print('the classifier came back with : %d, the real answer is %d'%(classifier_result, test_labels[i]))
        if classifier_result != test_labels[i]:
            error_count += 1.0

    print('\nthe total number of errors is : %d'% error_count)
    print('\nthe total error rate is %f'%(error_count/float(test_num)))

hw_test()