from numpy import linalg

import numpy as np
import common.load_data_from_file as load_data
import matplotlib.pyplot as plt


def stand_regress(x_arr, y_arr):
    """
    计算最佳拟合直线
    使用线性回归
    :param x_arr:
    :param y_arr:
    :return:   回归系数
    """
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr).T
    # 计算 X.T * X
    x_t_x = x_mat.T * x_mat
    # 判断上一步计算结果的行列式是否为零, 如果为零,那么计算逆矩阵的时候会出现错误,终止程序
    # numpy.linalg.det(mat) : 可以计算矩阵的行列式
    if linalg.det(x_t_x) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return

    # 行列式非0, 则
    # x_t_x.I  矩阵的逆矩阵
    # x_mat.T 矩阵的转置矩阵
    ws = x_t_x.I * (x_mat.T * y_mat)
    return ws


def test_and_draw():
    x_arr, y_arr = load_data.load_data_set('ex0.txt')
    # 获取到回归系数
    ws_ = stand_regress(x_arr, y_arr)
    print('回归系数\n', ws_)

    # 创建图像
    fig = plt.figure()
    # 在图像上创建子图
    axis = fig.add_subplot(111)
    # 转换成 矩阵数据格式
    x_mat_ = np.mat(x_arr)
    # 转换成 矩阵数据格式
    y_mat_ = np.mat(y_arr)

    # x_mat_[:, 1] 将第二特征全部取出来
    # flatten() 函数将矩阵所有的元素全部转成 1行m列的举证
    # .A[0]  将第0行的矩阵元素全部取出来组成一个数组, ndarray格式
    xs = x_mat_[:, 1].flatten().A[0]
    # 将 y_mat_数据取出来转成 ndarray格式
    ys = y_mat_.T[:, 0].flatten().A[0]
    axis.scatter(xs, ys, s=5, c='red')

    # 赋值x_mat_矩阵一份到x_copy
    x_copy = x_mat_.copy()
    # 对x_copy的列元素在列中排序, 这里传0, 如果需要按行元素在行中排序,传1,或者不传值
    x_copy.sort(0)
    # 计算 x_copy矩阵 和 回归系数 相乘得到的 拟合的直线的y值
    y_hat = x_copy * ws_
    # 取 x_copy的第二个特征, 取y_hat 画出拟合直线图形
    axis.plot(x_copy[:, 1], y_hat)
    plt.show()

    # 计算预测值和真实值的相关性
    corrcoef = np.corrcoef((x_mat_ * ws_).T, y_mat_)
    print('预测值和真实值的相关性\n', corrcoef)


# test_and_draw()