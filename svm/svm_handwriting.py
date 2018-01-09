import numpy as np
import os

from svm.opt_struct import OptStruct
from svm.platt_smo import platt_smo


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


def load_images(dir_name):
    hw_labels = []
    training_file_list = os.listdir(dir_name)   # 目录下的文件列表
    m = len(training_file_list)                 # 目录下的文件数量
    training_mat = np.zeros((m, 1024))
    for i in range(m):                          # 遍历每一个文件,构建样本数据举证
        file_name_str = training_file_list[i]   # 获取文件名
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])   # 获取该文件对应的分类标签
        if class_num_str == 9:               # 如果是9 , 分类 -1
            hw_labels.append(-1)
        else:
            hw_labels.append(1)              # 不是9, 分类 1
        training_mat[i, :] = img2vector('%s/%s' % (dir_name, file_name_str))   # 获取数据
    print('m is ', m)
    return training_mat, hw_labels


def test_digits(k_tup=('rbf', 10)):
    data_arr, label_arr = load_images('../kNN_test/trainingDigits')
    b, alphas = platt_smo(data_arr, label_arr, 200, 0.0001, 10000, k_tup)
    dat_mat = np.mat(data_arr)
    label_mat = np.mat(label_arr).transpose()
    sv_i_nd = np.nonzero(alphas.A > 0)[0]
    s_v_s = dat_mat[sv_i_nd]
    label_s_v = label_mat[sv_i_nd]
    print('there are %d support vectors' % np.shape(s_v_s)[0])

    m, n = np.shape(dat_mat)
    err_count = 0
    for i in range(m):
        kernel_eval = OptStruct.kernel_trans(s_v_s, dat_mat[i, :], k_tup)
        predict = kernel_eval.T * np.multiply(label_s_v, alphas[sv_i_nd]) + b
        if np.sign(predict) != np.sign(label_arr[i]):
            err_count += 1
    print('the training error rate is: %f' % (float(err_count)/m))

    data_arr, label_arr = load_images('../kNN_test/testDigits')
    err_count = 0
    data_mat = np.mat(data_arr)
    label_mat = np.mat(label_arr).transpose()
    m, n = np.shape(data_mat)
    for i in range(m):
        kernel_eval = OptStruct.kernel_trans(s_v_s, dat_mat[i, :], k_tup)
        predict = kernel_eval.T * np.multiply(label_s_v, alphas[sv_i_nd]) + b
        if np.sign(predict) != np.sign(label_arr[i]):
            err_count += 1
    print('the test error rate is : %f' % (float(err_count/m)))


test_digits(('rbf', 5))
