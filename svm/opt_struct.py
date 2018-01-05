import numpy as np


class OptStruct:

    def __init__(self, data_mat_in, class_labels, c, toler, k_tup):
        self.x = data_mat_in
        self.label_mat = class_labels
        self.c = c
        self.tol = toler
        self.m = np.shape(data_mat_in)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.e_cache = np.mat(np.zeros((self.m, 2)))

        self.k = np.mat(np.zeros((self.m, self.m))) # 先构建一个空矩阵
        for i in range(self.m):
            self.k[:, i] = self.kernel_trans(self.x, self.x[i, :], k_tup)

    @staticmethod
    def kernel_trans(x_val, a_val, k_tup):
        """
        核函数转换函数
        :param x_val:
        :param a_val:
        :param k_tup:  核函数的信息,元祖类型, 元祖的第一个参数描述所用的核函数类型的一个字符串 'lin'/'rbf'
        :return:
        """
        m, n = np.shape(x_val)
        # m行1列的全0矩阵, 列向量
        k = np.mat(np.zeros((m, 1)))
        if k_tup[0] == 'lin':
            k = x_val * a_val.T
        elif k_tup[0] == 'rbf':
            for j in range(m):
                delta_row = x_val[j, :] - a_val
                k[j] = delta_row * delta_row.T

            k = np.exp(k / (-1 * k_tup[1] ** 2))
        else:
            raise NameError('Houston We Have a Problem -- That Kernel is not recongized')
        return k
