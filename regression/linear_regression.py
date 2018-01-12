import json
import random
import urllib
from time import sleep

import numpy as np
from numpy import linalg


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


def lwlr(test_point, x_arr, y_arr, k=1.0):
    """
    局部加权的线性回归函数
    使用 '核' 为高斯核
    :param test_point:
    :param x_arr:
    :param y_arr:
    :param k:
    :return:  对单个点的估值
    """
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr).T
    # m是特征数
    m = np.shape(x_mat)[0]
    # np.eye((m)) : 生成m行m列的对角矩阵, 返回的是ndarray格式
    weights = np.mat(np.eye(m))
    # 高斯核公式: w(i,i) = exp(|xi - x|/(-2 * k**2))
    for j in range(m):
        # 测试点的所有特征和 第j行的训练数据所有特征相减,得到特征差值, 矩阵相减
        diff_mat = test_point - x_mat[j, :]
        # 利用高斯核 计算第j个权重
        # 这样就可以构建一个只含对角元素的权重矩阵w
        # 并且点 x 与点x(i)越近, w(j,j) 将会越大
        weights[j, j] = np.exp(diff_mat * diff_mat.T/(-2.0*k**2))

    # 局部加权线性回归公式: w` = (X.T * W * X).I * X.T * W * Y
    # 这里计算 上面括号中的内容, 这部分内容就是 x_mat的行列式的值
    x_t_x = x_mat.T * (weights * x_mat)

    # 判断x_mat的行列式的值如果为0, 那么在计算逆矩阵的时候讲出现错误
    if linalg.det(x_t_x) == 0.0:
        print('this matrix is singular, cannot do inverse')
        return

    # 完成 '局部加权线性回归公式' 剩下部分的计算
    ws = x_t_x.I * (x_mat.T * (weights * y_mat))
    return test_point * ws


def lwlrs(test_arr, x_arr, y_arr, k=1.0):
    """
    包装局部加权线性回归函数, 对多个点进行估值
    :param test_arr:  测试数据
    :param x_arr:     训练数据特征集合
    :param y_arr:     训练数据结果结合
    :param k:         优化参数
    :return:          对多个点的估值
    """
    m = np.shape(test_arr)[0]  # 特征数
    # 1行m列的数组ndarray, 对预期结果进行初始化
    y_hat = np.zeros(m)
    for i in range(m):
        y_hat[i] = lwlr(test_arr[i], x_arr, y_arr, k)
    return y_hat


def rss_error(y_arr, y_hat_arr):
    """
    计算预测值与真实值差值的平方和
    :param y_arr:
    :param y_hat_arr:
    :return:
    """
    return ((y_arr - y_hat_arr)**2).sum()


def calc_ridge_regress(x_mat, y_mat, lam=0.2):
    """
    岭回归
    公式:  w` = (X.T * X + lam * I).I * X.T * y
    :param x_mat:  特征矩阵
    :param y_mat:  结果值矩阵
    :param lam:   常量, 一个惩罚项, 同时也是为了防止  特征比样本数多时,不能计算
    :return:  返回 回归系数矩阵
    """
    x_t_x = x_mat.T * x_mat
    denom = x_t_x + np.eye(np.shape(x_mat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = denom.I * (x_mat.T * y_mat)
    return ws


def regularize_native_data(x_arr, y_arr):
    """
    首先对特征集做标准化处理
    使每一个特征具有相同的重要性(不考虑特征代表什么)
    公式:
        xi` = (xi - mean_x) / (var_x)
        yi` = (yi - mean_y)
    :param x_arr:
    :param y_arr:
    :return:    标准化处理之后的 训练特征矩阵和训练数据结果值矩阵
    """
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr).T
    # 计算 结果值的平均值
    y_mean = np.mean(y_mat, 0)
    # 和平均值的差值
    y_mat = y_mat - y_mean
    # 计算每一个特征的平均值
    x_means = np.mean(x_mat, 0)
    # 沿着垂直方向技计算总体的方差
    # 方差计算公式: 每一个元素和总体的平均值的差值的平方的和除上元素数量
    x_var = np.var(x_mat, 0)
    # 每一个样本的特征值与特征平均值的差值除上样本总体的方差
    x_mat = (x_mat - x_means) / x_var
    return x_mat, y_mat


def ridge_regression(x_arr, y_arr):
    """
    使用岭回归函数
    :param x_arr:  训练样本特征集
    :param y_arr:  训练样本结果集
    :return:  按照指数增长的惩罚系数改变的30个回归系数矩阵
    """
    x_mat, y_mat = regularize_native_data(x_arr, y_arr)
    num_test_pts = 30
    w_mat = np.zeros((num_test_pts, np.shape(x_mat)[1]))
    for i in range(num_test_pts):
        # 这里的惩罚系数lambda 按照指数级变化 , 从 e**(-10) --> e**(19)
        ws = calc_ridge_regress(x_mat, y_mat, np.exp(i - 10))
        w_mat[i, :] = ws.T

    return w_mat


def stage_wise(x_arr, y_arr, eps=0.01, num_it=100):
    """
    前向逐步线性回归
    :param x_arr:    样本数据特征集
    :param y_arr:    样本数据结果集
    :param eps:      每次迭代调整的步长
    :param num_it:   迭代的次数
    :return:         每次迭代产生的 回归系数的集合矩阵
    """
    # 对原始样本数据进行规范化
    x_mat, y_mat = regularize_native_data(x_arr, y_arr)
    # 获得样本的数量m, 特征数量n
    m, n = np.shape(x_mat)
    ret_mat = np.zeros((num_it, n))  # 初始化返回结果
    ws = np.zeros((n, 1))            # 初始化回归系数
    ws_test = ws.copy()
    ws_max = ws.copy()
    for i in range(num_it):       # 遍历次数,有函数外部确定,默认100次
        print(ws.T)
        lowest_err = np.inf       # 本次迭代中的最小误差 初始化
        for j in range(n):                           # 遍历每一种特征
            for sign in [-1, 1]:                     # 在 -1/1 之间切换 sign标志
                ws_test = ws.copy()
                ws_test[j] += eps * sign
                y_test = x_mat * ws_test              # 特征值与回归系数乘积 得到 预测的结果值
                rss_e = rss_error(y_mat.A, y_test.A)  # 判断预测的结果和真实的结果的误差
                if rss_e < lowest_err:                # 如果误差值比上一次的误差值小,
                    lowest_err = rss_e
                    ws_max = ws_test                  # 保存本次回归系数值为最佳回归系数
        ws = ws_max.copy()         # 遍历完所有特征之后, 获得最佳回归系数
        ret_mat[i, :] = ws.T       # 将本次最佳回归系数矩阵保存到 返回结果矩阵中
    return ret_mat


def split_train_test_data(x_arr, y_arr):
    """
    将数据集随机分隔成90%的数据作为训练数据,10%的数据作为测试数据
    :param index_list:
    :param m:
    :param x_arr:
    :param y_arr:
    :return:
    """
    train_x = []  # 训练数据集
    train_y = []
    test_x = []  # 测试数据集
    test_y = []

    # 样本数据点个数
    m = len(y_arr)
    # 索引列表
    index_list = range(m)
    # 对index_list集合中的顺序打乱
    random.shuffle(list(index_list))
    # 对样本数据点进行遍历
    for j in range(m):
        # 随机90%的数据放入到训练数据集中
        if j < m * 0.9:
            train_x.append(x_arr[index_list[j]])
            train_y.append(y_arr[index_list[j]])
        # 其他数据放入到 测试数据集中
        else:
            test_x.append(x_arr[index_list[j]])
            test_y.append(y_arr[index_list[j]])

    return test_x, test_y, train_x, train_y


def cross_validation(x_arr, y_arr, num_val=10):
    """
    交叉验证测试岭回归
    :param x_arr:
    :param y_arr:
    :param num_val: 交叉验证的次数
    :return:
    """
    #
    err_mat = np.zeros((num_val, 30))
    #
    for i in range(num_val):
        test_x, test_y, train_x, train_y = split_train_test_data(x_arr, y_arr)

        # 岭回归函数的所有回归系数
        w_mat = ridge_regression(train_x, train_y)
        # 遍历30次
        for k in range(30):
            # 测试数据集的矩阵
            mat_test_x = np.mat(test_x)
            # 训练数据集的矩阵
            mat_train_x = np.mat(train_x)
            # 训练数据集特征的平均值的矩阵
            mean_train = np.mean(mat_train_x, 0)
            # 训练数据集的 方差的矩阵
            var_train = np.var(mat_train_x, 0)
            #  使用训练数据集的平均值和方差获得规范之后的测试数据集矩阵
            mat_test_x = (mat_test_x - mean_train)/var_train
            # 通过公式  计算得到 测试数据集的预测结果矩阵
            # y` = X * W.T + mean_y
            y_est = mat_test_x * np.mat(w_mat[k, :]).T + np.mean(train_y)
            # 通过差的平方和计算误差,结果放入到err_mat矩阵中
            err_mat[i, k] = rss_error(y_est.T.A, np.array(test_y))

    # 平均误差值
    mean_err = np.mean(err_mat, 0)
    # 找到最小的均值误差
    min_mean = float(min(mean_err))
    # 将均值误差最小的lambda对应的回归系数作为最佳系数
    best_weights = w_mat[(np.nonzero(mean_err == min_mean))]

    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr).T
    mean_x = np.mean(x_mat, 0)
    var_x = np.var(x_mat, 0)
    un_reg = best_weights/var_x
    print('the best model from Ridge Regression is:\n', un_reg)
    print('with constrant term:', -1 * np.sum(np.multiply(mean_x, un_reg)) + np.mean(y_mat))



