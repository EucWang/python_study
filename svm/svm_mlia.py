"""
程序清单6-1 SMO算法中的辅助函数
"""
import numpy as np


def load_data_set(file_name):
    """
    从文件中读取数据,最后一个数据作为分类标签
    :param file_name:
    :return:
    """
    data_mat = []
    label_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        line_arr = line.strip().split('\t')
        items = [float(item) for item in line_arr]
        data_mat.append(items[:-1])
        label_mat.append(items[-1])

    return data_mat, label_mat


def select_j_rand(index_, m):
    """
    只要函数值不等于输入值i, 函数就会进行随机选择
    :param index_:  第一个alpha的下标
    :param m:  所有alpha的数目
    :return:
    """
    j = index_
    while j == index_:
        # 一直获取随机数,直到j != i, 跳出循环
        # 随机数的范围0~m
        j = int(np.random.uniform(0, m))

    return j


def clip_alpha(alpha_j, high, low):
    """
    用于调整大于high 或者小于low 的alpha值
    给定最大值,最小值,超过最大值,赋值为最大值,
    低于最小值,赋值为最小值
    :param alpha_j:
    :param high:
    :param low:
    :return:
    """
    if alpha_j > high:
        alpha_j = high
    if low > alpha_j:
        alpha_j = low

    return alpha_j


def smo_simple(data_list_in, class_labels, c, toler, max_iter):
    """
    简化版SMO算法
    smo算法的目标就是求出一系列alpha和b,
    一旦求出了这些alpha, 就很容易计算出权重向量w并获得分隔超平面
    SMO算法的工作原理是:
    每次循环中选择两个alpha进行优化处理
    一旦找到一对合适的alpha,那就就增大其中一个同时减小另一个

    :param data_list_in:  输入训练数据集, list
    :param class_labels:  输入训练数据分类标签, list
    :param c:             常数c, 惩罚因子?
    :param toler:         容错率, 松弛变量?
    :param max_iter:      取消前最大的循环次数
    :return:
    """
    # 数据集的matrix
    data_mat = np.mat(data_list_in)
    # 分类标签 转置之后的matrix,  m行1列
    label_mat = np.mat(class_labels).transpose()
    b = 0
    # 数据集的维度, m行n列
    m, n = np.shape(data_mat)
    # alpha矩阵初始化, m行1列全0矩阵, 拉格朗日乘子
    alphas = np.mat(np.zeros((m, 1)))
    # 当前迭代次数, max_iter是最大迭代次数
    index_iter = 0

    while index_iter < max_iter:
        # 标记位, 记录alpha在此次循环中, 有没有优化
        alpha_pairs_changed = 0
        for i in range(m):  # 第i个样本

            # 第i个样本的预测类别
            # alpha的每一个元素和label_mat的每一个元素相乘,得到一个m行1列的矩阵,然后转置为1行m列的矩阵
            # data_mat第i行的转置为m行1列的矩阵,然后和data_mat矩阵做矩阵乘积,得到一个m行1列的矩阵
            # 然后两个矩阵乘积得到1个数值,然后加上b
            fxi = float(np.multiply(alphas, label_mat).T * (data_mat * data_mat[i, :].T)) + b
            # 误差
            ei = fxi - float(label_mat[i])

            # 是否可以继续优化
            if ((label_mat[i] * ei < - toler) and (alphas[i] < c)) or \
                    ((label_mat[i] * ei > toler) and (alphas[i] > 0)):
                # 随机选择第j个样本
                j = select_j_rand(i, m)
                # 第j个样本的预测类别
                fxj = float(np.multiply(alphas, label_mat).T * (data_mat * data_mat[j, :].T)) + b
                # 第j个样本的误差
                ej = fxj - float(label_mat[j])
                # 分配新的内存, 存储alpha_i 和alpha_j 的值
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()

                if label_mat[i] != label_mat[j]:
                    low = max(0, alphas[j] - alphas[i])
                    high = min(c, c + alphas[j] - alphas[i])
                else:
                    low = max(0, alphas[j] + alphas[i] - c)
                    high = min(c, alphas[j] + alphas[i])

                if low == high:
                    print('low == high')
                    continue

                # eta 是alpha[j]的最优化修改量,
                eta = 2.0 * data_mat[i, :] * data_mat[j, :].T - \
                      data_mat[i, :] * data_mat[i, :].T - \
                      data_mat[j, :] * data_mat[j, :].T

                # 如果eta为0, 那就是说需要退出for循环的当前迭代过程.
                # 该过程对真实SMO算法进行了简化处理,如果eta为0, 那么计算新的alpha[j]比较麻烦
                if eta >= 0:
                    print('eta >= 0')
                    continue

                alphas[j] -= label_mat[j] * (ei - ej) / eta
                # 阻止对alpha_j的修改量过大
                alphas[j] = clip_alpha(alphas[j], high, low)

                # 如果修改量很微小, 跳过本轮循环
                if abs(alphas[j] - alpha_j_old) < 0.00001:
                    print('j not moving enough')
                    continue

                # alpha_i 的修改方向相反,但是和 alpha_j的改变的大小一样
                alphas[i] += label_mat[j] * label_mat[i] * (alpha_j_old - alphas[j])

                # 为两个alpha设置常数项b
                b1 = b - ei - label_mat[i] * (alphas[i] - alpha_i_old) * data_mat[i, :] * data_mat[i, :].T - \
                     label_mat[j] * (alphas[j] - alpha_j_old) * data_mat[i, :] * data_mat[j, :].T

                b2 = b - ej - label_mat[i] * (alphas[i] - alpha_i_old) * data_mat[i, :] * data_mat[j, :].T - \
                     label_mat[j] * (alphas[j] - alpha_j_old) * data_mat[j, :] * data_mat[j, :].T

                if 0 < alphas[i] < c:
                    b = b1
                elif 0 < alphas[j] < c:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                # 说明alpha已经发生了改变
                alpha_pairs_changed += 1
                print('iter: %d i : %d, pairs changed %d' % (index_iter, i, alpha_pairs_changed))

        # 如果没有更新, 那么继续迭代
        # 如果有更新,那么迭代次数归0, 继续优化
        if alpha_pairs_changed == 0:
            index_iter += 1
        else:
            index_iter = 0
        print('iteration number: %d' % index_iter)

    # 只有当某次优化更新到了最大迭代次数
    # 这个时候才返回优化之后的alpha 和 b
    return b, alphas


# data_arr, label_arr = load_data_set('testSet.txt')
# b_, alphas_ = smo_simple(data_arr, label_arr, 0.6, 0.001, 40)
# print('b', b_)
# # 对数据进行过滤, 只显示大于0的元素, 这个过滤只对numpy.ndarray类型有用
# print('alpha', alphas_[alphas_ > 0])
#
# print('那些数据点是支撑向量:')
# for i in range(100):
#     if alphas_[i] > 0.0:
#         print(data_arr[i], label_arr[i])
