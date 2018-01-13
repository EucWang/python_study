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
    return np.mean(data_set[:, -1])


def reg_err(data_set):
    return np.var(data_set[:, -1]) * np.shape(data_set)[0]


def choose_best_split(data_set, leaf_type, err_type, opts):
    pass


# def create_tree(data_set, leaf_type=reg_leaf, err_type=reg_err, ops=(1, 4)):
#     feature, value = choose_best_split(data_set, leaf_type, err_type, ops)
#     if feature == None:
#         return value
#     ret_tree = {'sp_ind': feature, 'sp_val': value}
#
#     l_set, r_set = bin_split_data_set(data_set, feature, value)
#     ret_tree['left'] = create_tree(l_set, leaf_type, err_type, ops)
#     ret_tree['right'] = create_tree(r_set, leaf_type, err_type, ops)
#     return ret_tree

data_arr = load_data_set('ex0.txt')
print('0: ', data_arr[0], type(data_arr[0][0]))



# a, b = bin_split_data_set(np.mat(np.eye(4)), 1, 0.5)
# print('a', a)
# print('b', b)