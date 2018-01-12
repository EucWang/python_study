
import numpy as np
import common.load_data_from_file as load_data
import matplotlib.pyplot as plt
import regression.linear_regression as regress


def test_and_draw_stand_regress():
    x_arr, y_arr = load_data.load_data_set('ex0.txt')
    # 获取到回归系数
    ws_ = regress.stand_regress(x_arr, y_arr)
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


def test_and_draw_lwlr_test():

    x_arr, y_arr = load_data.load_data_set('ex0.txt')
    # 获取到回归系数
    y_hat0 = regress.lwlr(x_arr[0], x_arr, y_arr, 1.0)
    print('k=1, 对第一个点估值\t', y_hat0, '\t实际值:\t', y_arr[0])
    y_hat0 = regress.lwlr(x_arr[0], x_arr, y_arr, 0.1)
    print('修改k=0.1, 对第一个点估值\t', y_hat0, '\t实际值:\t', y_arr[0])
    y_hat0 = regress.lwlr(x_arr[0], x_arr, y_arr, 0.01)
    print('修改k=0.05, 对第一个点估值\t', y_hat0, '\t实际值:\t', y_arr[0])

    y_hat = regress.lwlrs(x_arr, x_arr, y_arr, 0.003)
    x_mat = np.mat(x_arr)
    srt_ind = x_mat[:, 1].argsort(0) # 对第二个特征所有值进行排序之后的结果的索引值保存到srt_ind中
    x_sort = x_mat[srt_ind][:, 0, :]

    # # 创建图像
    fig = plt.figure()
    # # 在图像上创建子图
    axis = fig.add_subplot(111)
    axis.plot(x_sort[:, 1], y_hat[srt_ind])
    axis.scatter(x_mat[:, 1].flatten().A[0], np.mat(y_arr).flatten().A[0], s=5, c='red')
    plt.show()
    #
    # # 计算预测值和真实值的相关性
    # corrcoef = np.corrcoef((x_mat_ * ws_).T, y_mat_)
    # print('预测值和真实值的相关性\n', corrcoef)


def test_abalone():
    """
    预测鲍鱼的年龄
    :return:
    """
    abalone_x, abalone_y = load_data.load_data_set('abalone.txt')
    y_hat01 = regress.lwlrs(abalone_x[0:99], abalone_x[0:99], abalone_y[0:99], 0.1)
    y_hat1 = regress.lwlrs(abalone_x[0:99], abalone_x[0:99], abalone_y[0:99], 1)
    y_hat10 = regress.lwlrs(abalone_x[0:99], abalone_x[0:99], abalone_y[0:99], 10)

    err01 = regress.rss_error(abalone_y[0:99], y_hat01.T)
    err1 = regress.rss_error(abalone_y[0:99], y_hat1.T)
    err10 = regress.rss_error(abalone_y[0:99], y_hat10.T)
    print('在训练数据上, k取值越小,拟合得越好')
    print('err01', err01, '\nerr1', err1, '\nerr10', err10)

    y_hat01 = regress.lwlrs(abalone_x[100:199], abalone_x[0:99], abalone_y[0:99], 0.1)
    y_hat1 = regress.lwlrs(abalone_x[100:199], abalone_x[0:99], abalone_y[0:99], 1)
    y_hat10 = regress.lwlrs(abalone_x[100:199], abalone_x[0:99], abalone_y[0:99], 10)

    err01 = regress.rss_error(abalone_y[100:199], y_hat01.T)
    err1 = regress.rss_error(abalone_y[100:199], y_hat1.T)
    err10 = regress.rss_error(abalone_y[100:199], y_hat10.T)
    print('在新的测试数据上, k取值却是越大越好')
    print('err01', err01, '\nerr1', err1, '\nerr10', err10)

    ws = regress.stand_regress(abalone_x[0:99], abalone_y[0:99]) # 简单的线性回归
    y_hat_simple = np.mat(abalone_x[100:199]) * ws
    hat_simple = regress.rss_error(abalone_y[100:199], y_hat_simple.T.A)
    print('简单线性回归,训练前100个,测试后100,得到的误差')
    print('hat_simple', hat_simple)


def test_ridge_regress():
    """
    测试 岭回归
    :return:
    """
    abalone_x, abalone_y = load_data.load_data_set('abalone.txt')
    ridge_weights = regress.ridge_regression(abalone_x, abalone_y)
    fig = plt.figure()
    subplot = fig.add_subplot(111)
    # 这里plot() 接受矩阵参数 矩阵m行n列
    # 将矩阵垂直方向的数据作为不同的线段上的数据
    # 将画出n条线段,
    subplot.plot(ridge_weights)
    plt.show()


def test_stage_wise():
    """
    测试前向逐步线性回归
    :return:
    """
    abalone_x, abalone_y = load_data.load_data_set('abalone.txt')
    # ridge_weights = regress.stage_wise(abalone_x, abalone_y)
    ridge_weights = regress.stage_wise(abalone_x, abalone_y, 0.001, 5000)
    fig = plt.figure()
    subplot = fig.add_subplot(111)
    # 这里plot() 接受矩阵参数 矩阵m行n列
    # 将矩阵垂直方向的数据作为不同的线段上的数据
    # 将画出n条线段,
    subplot.plot(ridge_weights)
    plt.show()


def test_cross_ridge_regress():
    abalone_x, abalone_y = load_data.load_data_set('abalone.txt')
    regress.cross_validation(abalone_x, abalone_y)


test_cross_ridge_regress()
# test_stage_wise()
# test_ridge_regress()
# test_and_draw()
#test_and_draw_lwlr_test()
# test_abalone()
