import numpy as np
import matplotlib.pyplot as plt


def load_simple_data():
    """

    :return: 测试数据
    """
    data_mat = np.matrix([[1, 2.1],
                          [2, 1.1],
                          [1.3, 1],
                          [1, 1],
                          [2, 1]])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_mat, class_labels


def draw_test_data():
    data_mat, class_labels = load_simple_data()

    x = data_mat[:, 0].getA()
    y = data_mat[:, 1].getA()

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for index in range(len(class_labels)):
        if class_labels[index] == 1.0:
            x1.append(x[index])
            y1.append(y[index])
        elif class_labels[index] == -1.0:
            x2.append(x[index])
            y2.append(y[index])

    plt.scatter(x1, y1, s=40, c='blue')
    plt.scatter(x2, y2, s=40, c='red')
    plt.show()


def stump_classify(data_mat, dimen, threshholds_val, threshholds_ineq):
    """
    通过阈值比较对数据进行分类
    所有阈值一边的数据会分到类别-1, 而另外一边的数据分到类别+1
    通过 numpy的 数组过滤的功能实现
    首先将返回数组的全部元素设置为1,
    然后将所有不满足不等式要求的元素设置为-1.
    可以基于数据集中的任一元素进行比较, 同时可以将不等号在大于,小于之前切换

    :param data_mat:
    :param dimen:
    :param threshholds_val:  阈值
    :param threshholds_ineq: 阈值判断条件, 是大于 还是小于
    :return: 根据阈值判断条件,获取对应data_mat数据集的一个分类矩阵, 不满足条件的置位-1, 满足条件的置位1
    """
    columns = np.shape(data_mat)[0]  # 列数
    ret_arr = np.ones((columns, 1))

    if threshholds_ineq == 'lt':
        ret_arr[data_mat[:, dimen] <= threshholds_val] = -1.0
    else:
        ret_arr[data_mat[:, dimen] > threshholds_val] = -1.0
    return ret_arr


def build_stump(data_arr, class_labels, D):
    """

    :param data_arr:
    :param class_labels:
    :param D:
    :return:
    """
    data_mat = np.mat(data_arr)
    label_mat = np.mat(class_labels).T
    m, n = np.shape(data_mat)

    # 用于在特征的所有可能值上进行遍历
    num_steps = 10.0

    # 这个字典用于存储给定权重向量D时所得到的最佳单层决策时的相关信息
    best_stump = {}

    best_class_est = np.mat(np.zeros((m, 1)))

    #  初始化最小错误值为无穷大, 用于寻找可能的最小错误率
    min_error = np.inf

    # 遍历每一列, 遍历数据集的所有特征
    for i in range(n):
        range_min = data_mat[:, i].min()   # 当前列里的最小值
        range_max = data_mat[:, i].max()   # 当前列里的最大值
        step_size = (range_max - range_min) / num_steps   # 根据列里的数值,确定平均步长
        # 将
        for j in range(-1, int(num_steps) + 1):
            # 在大于和小于之间切换不等式
            for inequal in ['lt', 'gt']:
                thresh_val = (range_min + float(j) * step_size)
                # 遍历特征行, 遍历 阈值, 遍历 阈值判断式,
                # 然后根据分类函数,确定 分类结果
                predicted_vals = stump_classify(data_mat, i, thresh_val, inequal)

                err_arr = np.mat(np.ones((m, 1)))         # 判断上面的分类方式的错误情况的矩阵,初始为1
                err_arr[predicted_vals == label_mat] = 0  # 如果分类正确的行,则设置为0,

                # 将错误向量err_arr和权重向量D 的相应元素相乘并求和,
                # 就得到数值weighted_err
                # 我们是基于权重向量D而不是其他错误计算指标来评价分类器的
                weighted_err = D.T * err_arr
                # print('split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f' % (i, thresh_val, inequal, weighted_err))
                if weighted_err < min_error:
                    # 最小的错误率
                    min_error = weighted_err
                    # 最好的预测分类结果
                    best_class_est = predicted_vals.copy()

                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal

    return best_stump, min_error, best_class_est


def adaboost_train_ds(data_arr, class_labels, num_it=40):
    """
    基于单层决策树的AdaBoost训练过程

    函数名称尾部的'ds' 代表 '单层决策树'(decision stump)
    单层决策树是AdaBoost中最流行的弱分类器, 但是并非唯一可用的弱分类器
    本函数建立在单层决策树基础之上, 任何分类器都可以作为基础分类器

    :param data_arr:       数据集
    :param class_labels:   类别标签
    :param num_it:         迭代次数
    :return:
    """
    weak_class_arr = []
    # 数据集的特征维度, 列数, 特征数量
    m = np.shape(data_arr)[0]

    # 权重向量,包含每一个数据点的权重
    # 初始赋予相同的值
    # D 是一个概率分布向量, 其所有元素之和为1.0
    D = np.mat(np.ones((m, 1))/m)
    # 列向量, 记录每个数据点的类别估计累计值
    agg_class_est = np.mat(np.zeros((m, 1)))

    for i in range(num_it):
        best_stump, min_err, best_class_est = build_stump(data_arr, class_labels, D)

        min_err = min_err.sum()
        print("D:", D.T, '\nmin_err', min_err)
        # 计算分类器的alpha权重值
        # np.max(min_err, np.le-16)  用于确保在没有错误时(min_err值为0时)不会发生除零溢出
        # alpha = 0.5 * ln((1- err)/err)
        alpha = float(0.5 * np.log((1.0 - min_err)/max(min_err, 1e-16)))
        best_stump['alpha'] = alpha
        # 将分类器的alpha权重值和分类结果放入到统计数组中
        weak_class_arr.append(best_stump)

        # 为下一次迭代计算新的权重向量D
        print('class_est', best_class_est.T)
        # 当预测分类和实际分类相同时, 表示被正确的分类了,  计算结果为: expon = -alpha, 则正确的样本通过公式计算会被降低权重
        # 当预测分类和实际分类不同时, 表示被错误的分类了, 计算结果为: expon = alpha,   则错误的样本通过公式计算会被提高权重
        expon = np.multiply(-1 * alpha * np.mat(class_labels).T, best_class_est)
        # 公式: d_new = (d_old * np.exp( (+/-)alpha))/ sum(d_all)
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()

        # 记录每个数据点的类别估计累计值
        agg_class_est += alpha * best_class_est
        print('-------------------------------------\n'
              'agg_class_est', agg_class_est.T)

        # np.sign() 函数, 符号函数, 如果 数值小于0, 返回-1, 大于0, 返回1, 等于0, 则返回0
        # np.sign(agg_class_est) != np.mat(class_labels).T 获得一个bool值的矩阵,
        # 有(True==1,False==0)成立 式子判断预测结果和实际结果的的情况,
        # 这里是不相等的情况, 不相等时为True, 表示预测结果错误, 相等时为False,表示预测结果正确
        # 然后调用 np.multiply() 和全1矩阵计算对应的每个元素和1的乘积,
        # Ture*1 = 1, False*1 = 0
        agg_errs = np.multiply(np.sign(agg_class_est) != np.mat(class_labels).T, np.ones((m, 1)))
        # agg_errs.sum() 获得的就是 一个数据集中预测的错误数量,
        # agg_errs.sum()/m 错误率
        err_rate = agg_errs.sum()/m
        print('total err: ', err_rate, "\n")

        # 如果错误率为0, 终止循环
        if err_rate == 0.0:
            break

    return weak_class_arr


def ada_classify(dat_to_class, classifier_arr):
    """
    AdaBoost分类函数
    :param dat_to_class:
    :param classifier_arr:
    :return:
    """
    data_mat = np.mat(dat_to_class)
    # 特征数
    m = np.shape(data_mat)[0]
    agg_class_est = np.mat(np.zeros((m, 1)))
    for i in range(len(classifier_arr)):
        class_est = stump_classify(data_mat, classifier_arr[i]['dim'], classifier_arr[i]['thresh'], classifier_arr[i]['ineq'])
        agg_class_est += classifier_arr[i]['alpha'] * class_est
        print(agg_class_est)

    return np.sign(agg_class_est)


# draw_test_data()

# data_mat, class_labels = load_simple_data()
# D = np.mat(np.ones((5, 1))/5)
# best_stump, min_error, best_class_est = build_stump(data_mat, class_labels, D)
# print('best_stump', best_stump)
# print('min_err', min_error)
# print('best_class_est', best_class_est)

# data_mat, class_labels = load_simple_data()
# classifier_arr = adaboost_train_ds(data_mat, class_labels)
# classify = ada_classify([-1, -1], classifier_arr)
# print('classify', classify)
