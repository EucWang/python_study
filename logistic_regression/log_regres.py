import math
import numpy as np


def load_data_set():
    data_mat = []
    label_mat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([1.0, float(line_arr[0])], float(line_arr[1]))
        label_mat.append(int(line_arr[2]))

    return data_mat, label_mat


def sigmoid(in_x):
    """
    Sigmoid算法, 将一个大数转换到 0~1 之间, 通过判断其是否大于0.5 确定其值要么为0, 要么为1
    :param in_x:
    :return:
    """
    return 1.0/(1 + np.exp(-in_x))


def grad_ascent(data_mat_in, class_labels):
    """

    这里将第0维特征的值设置为1.0

    Logistic回归梯度上升优化算法
    :param data_mat_in: 一个2维Numpy数组, 每列代表不同的特征, 每行代表每个训练样本
    :param class_labels: 数据对应的类别标签
    :return:
    """

    #
    data_matrix = np.mat(data_mat_in)

    label_mat = np.mat(class_labels).transpose()  # 转置,  转换成列向量

    m, n = np.shape(data_matrix)    # 获取矩阵的 维数, m行, n列

    alpha = 0.001                   # 向目标移动的步长

    max_cycles = 500                # 迭代的次数

    weights = np.ones((n, 1))       #  n行, 1 列, 列向量 , 全部数据都是1

    for k in range(max_cycles):     # 迭代500次
        # 首先让data_matrix 与weights做矩阵相乘, 获得一个列向量
        h = sigmoid(data_matrix * weights)      # 通过sigmoid算法计算得到一个初始的值, 这里的h依然是一个列向量
        error = (label_mat - h)                 # 和标签结果比较,获取误差值,  这里error也成了一个列向量了

        # 这里alpha * data_matrix.T ,会乘到矩阵上的每一个元素上
        # alpha * data_matrix.T * error 依然会得到一个列向量
        # 然后 + weights, 依然是列向量, 元素和元素想加, 结果依然是列向量
        weights = weights + alpha * data_matrix.transpose() * error

    return weights
