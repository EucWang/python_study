import numpy as np
from svm.opt_struct import OptStruct
from svm.svm_mlia import select_j_rand, clip_alpha, load_data_set


def calc_ek(opt_s, k_index):
    """
    对于给定的alpha值, 本函数能够计算E值并返回
    :param opt_s:    OptStruct类型数据
    :param k_index:
    :return:         计算误差值
    """
    fxk = float(np.multiply(opt_s.alphas, opt_s.label_mat).T * (opt_s.x * opt_s.x[k_index, :].T)) + opt_s.b
    ek = fxk - float(opt_s.label_mat[k_index])
    return ek


def update_e_k(opt_s, k):
    """
    调用计算误差值的方法,然后把误差值存入到缓存中
    在对alpha值进行优化之后,会用到这个缓存值
    :param opt_s:
    :param k:
    :return:
    """
    ek = calc_ek(opt_s, k)
    opt_s.e_cache[k] = [1, ek]


def select_j(i_index, opt_s, ei):
    """
    用于选择内循环的第二个alpha值
    这里的目标是选择合适的第二个alpha值以保证在每次优化中采用最大步长

    :param i_index:
    :param opt_s:
    :param ei:
    :return:
    """
    max_k = -1
    max_delta_e = 0
    ej = 0
    opt_s.e_cache[i_index] = [1, ei]
    # 构建一个非0表, ndarray类型
    valid_e_cache_list = np.nonzero(opt_s.e_cache[:, 0].A)[0]
    if len(valid_e_cache_list) > 1:
        for k in valid_e_cache_list:
            if k == i_index:
                continue
            ek = calc_ek(opt_s, k)
            delta_e = abs(ei - ek)
            if delta_e > max_delta_e:
                max_k = k
                max_delta_e = delta_e
                ej = ek

        return max_k, ej
    else:
        j = select_j_rand(i_index, opt_s.m)
        ej = calc_ek(opt_s, j)
        return j, ej


def inner_l(i_index, opt_s):
    """

    :param i_index:
    :param opt_s:
    :return:
    """
    ei = calc_ek(opt_s, i_index)

    if ((opt_s.label_mat[i_index] * ei < -opt_s.tol) and (opt_s.alphas[i_index] < opt_s.c)) or \
            ((opt_s.label_mat[i_index] * ei > opt_s.tol) and (opt_s.alphas[i_index] > 0)):

        j_index, ej = select_j(i_index, opt_s, ei)

        alpha_i_old = opt_s.alphas[i_index].copy()
        alpha_j_old = opt_s.alphas[j_index].copy()

        if opt_s.label_mat[i_index] != opt_s.alphas[i_index]:
            low = max(0 , opt_s.alphas[j_index] - opt_s.alphas[i_index])
            high = min(opt_s.c, opt_s.c + opt_s.alphas[j_index] - opt_s.alphas[i_index])
        else:
            low = max(0, opt_s.alphas[j_index] + opt_s.alphas[i_index] - opt_s.c)
            high = min(opt_s.c, opt_s.alphas[j_index] + opt_s.alphas[i_index])

        if low == high:
            print('low == high')
            return 0

        eta = 2.0 * opt_s.x[i_index, :] * opt_s.x[j_index, :].T - \
              opt_s.x[i_index, :] * opt_s.x[i_index, :].T - \
              opt_s.x[j_index, :] * opt_s.x[j_index, :].T

        if eta >= 0:
            print('eta >= 0')
            return 0

        opt_s.alphas[j_index] -= opt_s.label_mat[j_index] * (ei - ej)/eta
        opt_s.alphas[j_index] = clip_alpha(opt_s.alphas[j_index], high, low)

        # 在alpha改变时, 更新e_cache
        update_e_k(opt_s, j_index)

        b1 = opt_s.b - ei - \
            opt_s.label_mat[i_index] * (opt_s.alphas[i_index] - alpha_i_old) * opt_s.x[i_index, :] * opt_s.x[i_index, :].T - \
            opt_s.label_mat[j_index] * (opt_s.alphas[j_index] - alpha_j_old) * opt_s.x[i_index, :] * opt_s.x[j_index, :].T

        b2 = opt_s.b - ej - \
            opt_s.label_mat[i_index] * (opt_s.alphas[i_index] - alpha_i_old) * opt_s.x[i_index, :] * opt_s.x[j_index, :].T - \
            opt_s.label_mat[j_index] * (opt_s.alphas[j_index] - alpha_j_old) * opt_s.x[j_index, :] * opt_s.x[j_index, :].T

        if (opt_s.alphas[i_index] > 0) and (opt_s.c > opt_s.alphas[i_index]):
            opt_s.b = b1
        elif (opt_s.alphas[j_index] > 0) and (opt_s.c > opt_s.alphas[j_index]):
            opt_s.b = b2
        else:
            opt_s.b = (b1 + b2)/2.0

        return 1
    else:
        return 0


def platt_smo(data_mat_in, class_labels, c, toler, max_iter, k_tup=('lin', 0)):
    """
    完整版的PlattSMO算法
    :param data_mat_in:
    :param class_labels:
    :param c:
    :param toler:
    :param max_iter:
    :param k_tup:
    :return:
    """
    opt_strt = OptStruct(np.mat(data_mat_in), np.mat(class_labels).transpose(), c, toler)
    iter = 0
    entire_set = True
    alpha_pairs_changed = 0
    while iter < max_iter and ((alpha_pairs_changed > 0) or entire_set):
        alpha_pairs_changed = 0
        if entire_set:
            # 在数据集上遍历任意可能的alpha
            for i in range(opt_strt.m):
                # 如果有任意一对alpha值发生改变,返回1
                alpha_pairs_changed += inner_l(i, opt_strt)
            print('fullSet, iter: % i:%d, pairs changed %d' % (iter, i, alpha_pairs_changed))
            iter += 1
        else:
            non_bound_is = np.nonzero((opt_strt.alphas.A > 0) * (opt_strt.alphas.A < c))[0]
            # 遍历所有的非边界alpha值,也就是不在边界0 或者c上的值
            for i in non_bound_is:
                alpha_pairs_changed += inner_l(i, opt_strt)
                print('non-bound, iter: %d i: %d, pairs changed %d' % (iter, i, alpha_pairs_changed))
            iter += 1

        if entire_set:
            entire_set = False
        elif alpha_pairs_changed == 0:
            entire_set = True
        print('iteration number: %d' % iter)

    return opt_strt.b, opt_strt.alphas


def calc_ws(alphas, data_arr, class_labels):
    """
    计算 w向量
    :param alphas:  拉格朗日乘子 向量
    :param data_arr:  数据
    :param class_labels:  数据的分类标签
    :return:
    """
    x = np.mat(data_arr)
    label_mat = np.mat(class_labels).transpose()
    m, n = np.shape(x)
    w = np.zeros((n, 1))
    for i in range(m):
        # alpha_i * yi 的矩阵乘方法
        w += np.multiply(alphas[i] * label_mat[i], x[i, :].T)

    return w


data_arr, label_arr = load_data_set('testSet.txt')
b, alphas = platt_smo(data_arr, label_arr, 0.6, 0.001, 40)
ws = calc_ws(alphas, data_arr, label_arr)
print("ws", ws)
for i in range(99):
    prediction = (np.mat(data_arr)[i] * np.mat(ws) + b).getA()[0][0]
    print('prediction:', prediction, "\tlabel%d:" % i, label_arr[i])