import numpy as np


def load_data_set(filename):
    """
    从文件中读取数据,将数据转换成浮点数
    :param filename:
    :return:
    """
    data_mat = []
    fr = open(filename)
    for line in fr.readlines():
        cur_line = line.strip().split('\t')
        flt_line = [float(item) for item in cur_line]
        data_mat.append(flt_line)

    return data_mat


def bin_split_data_set(data_set, feature_index, split_value):
    """
    将data_set进行分割,
    按照第feature_index列的特征,特征值为split_value来进行分割
    特征值大于split_value的数据行放到第一个矩阵中
    特征值小于等于split_value的数据行放到第二个矩阵中

    :param data_set:        训练数据矩阵
    :param feature_index:   特征在矩阵中的索引号
    :param split_value:     给定特征值
    :return:
    """
    # 获得指定索引的全部特征值的矩阵
    mat_the_feature = data_set[:, feature_index]
    # nonzero() 方法返回非0元素的索引值, 返回元祖类型
    nonzero1 = np.nonzero(mat_the_feature > split_value)
    value1_ = nonzero1[0]
    mat0 = data_set[value1_, :]

    nonzero2 = np.nonzero(mat_the_feature <= split_value)
    value2_ = nonzero2[0]
    mat1 = data_set[value2_, :]
    return mat0, mat1


def reg_leaf(data_set):
    """
    数据集最后一列的数据的平均值
    :param data_set:
    :return:
    """
    return np.mean(data_set[:, -1])


def reg_err(data_set):
    """
    data_set最后一列的数据集的方差 与 data_set的数据量 的乘积
    相当于只计算了最后一列每一个数据与最后一列平均值的差的平方和
    :param data_set:
    :return:
    """
    return np.var(data_set[:, -1]) * np.shape(data_set)[0]


def choose_best_split(data_set, leaf_type=reg_leaf, err_type=reg_err, ops={1,4}):
    """
    伪代码:
        对每个特征:
            对每个特征值:
                将数据集切分成两份
                计算切分的误差
                如果当期误差小于当前最小误差,那么将当期切分设定为最佳切分并更新最小误差
        放回最佳切分的特征和阈值
    :param data_set:
    :param leaf_type:
    :param err_type:
    :param opts:
    :return:
    """
    # 容许的误差下降值
    tol_s = ops[0]
    # 切分时的最少样本数
    tol_n = ops[1]
    # 将最后一列的数据转换成list类型的一维的一行数据
    result0 = data_set[:, -1].T.tolist()[0]
    # 去重之后, 如果长度为1, 表示所有值相等,则退出函数
    if len(set(result0)) == 1:
        return None, leaf_type(data_set)

    m, n = np.shape(data_set)
    # 计算总体的误差值
    sum_total_var = err_type(data_set)

    best_sum_var = np.inf
    best_index = 0
    best_value = 0

    # 遍历数据集的每一列
    for feat_index in range(n-1):
        # 去重之后, 遍历每一列中的每一个不同的值
        for split_val in set(data_set[:, feat_index].T.tolist()[0]):
            # 用不同列, 不同取值 对集合进行划分,得到 mat0, mat1
            mat0, mat1 = bin_split_data_set(data_set, feat_index, split_val)

            # 如果切分的结合的数据点数 小于设置的最小切分数,跳过本轮遍历
            if (np.shape(mat0)[0] < tol_n) or (np.shape(mat1)[0] < tol_n):
                continue
            # 计算切分之后的两个矩阵的总方差的和
            new_sum_var = err_type(mat0) + err_type(mat1)

            # 如果总方差的和最小,保存本轮遍历为最好结果值
            if new_sum_var < best_sum_var:
                best_index = feat_index
                best_value = split_val
                best_sum_var = new_sum_var

    # 如果数据集总体的总方差 与 最佳切分的 总发差 的比值小于 设置的最小误差, 退出函数
    if (sum_total_var - best_sum_var) < tol_s:
        return None, leaf_type(data_set)

    # 使用最佳分隔特征,以及最佳分隔的特征值来切分数据集,
    mat0, mat1 = bin_split_data_set(data_set, best_index, best_value)

    # 判断最佳分隔的结果是否小于最小切分样本数
    if (np.shape(mat0)[0] < tol_n) or (np.shape(mat1)[0] < tol_n):
        return None, leaf_type(data_set)
    return best_index, best_value


def create_tree(data_set, leaf_type=reg_leaf, err_type=reg_err, ops=(1, 4)):
    feature, value = choose_best_split(data_set, leaf_type, err_type, ops)
    if feature == None:
        return value
    ret_tree = {'sp_ind': feature, 'sp_val': value}

    l_set, r_set = bin_split_data_set(data_set, feature, value)
    ret_tree['left'] = create_tree(l_set, leaf_type, err_type, ops)
    ret_tree['right'] = create_tree(r_set, leaf_type, err_type, ops)
    return ret_tree


data_arr = load_data_set('ex00.txt')
trees = create_tree(np.mat(data_arr))
print(trees)



# a, b = bin_split_data_set(np.mat(np.eye(4)), 1, 0.5)
# print('a', a)
# print('b', b)
