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

    数据集最后一列的总方差
    :param data_set:
    :return:
    """
    return np.var(data_set[:, -1]) * np.shape(data_set)[0]


def choose_best_split(data_set, leaf_type=reg_leaf, err_type=reg_err, ops={1, 4}):
    """
    伪代码:
        对每个特征:
            对每个特征值:
                将数据集切分成两份
                计算切分的误差
                如果当期误差小于当前最小误差,那么将当期切分设定为最佳切分并更新最小误差
        放回最佳切分的特征和阈值
    :param data_set:  待切分的数据集矩阵
    :param leaf_type: 计算得到最后一列的平均值的函数
    :param err_type:  计算得到最后一列的总方差的函数
    :param ops:       一个set集合, 第一个值表示`容许的误差下降值`, 第二个值表示`切分时的最少样本数`
    :return:          两个值, 第一个表示最佳切分的特征索引, 第二个表示最佳切分的特征值
    """
    # 容许的误差下降值
    tol_s = ops[0]
    # 切分时的最少样本数
    tol_n = ops[1]
    # 将最后一列的数据转换成list类型的一维的一行数据
    result0 = data_set[:, -1].T.tolist()[0]
    # 去重之后, 如果长度为1, 表示所有值相等,则退出函数
    if len(set(result0)) == 1:
        print("数据最后一列所有值相等,退出函数")
        return None, leaf_type(data_set)
    # 数据集矩阵的 行数, 列数
    m, n = np.shape(data_set)
    # 计算总体数据集的总方差
    sum_total_var = err_type(data_set)

    # 默认值为无穷大
    best_sum_var = np.inf
    best_index = 0
    best_value = 0

    # 遍历数据集的每一列, 去掉最后一列
    for feat_index in range(n - 1):
        # 去重之后, 遍历每一列中的每一个不同的值
        for split_val in set(data_set[:, feat_index].T.tolist()[0]):
            # 用不同列, 不同取值 对集合进行划分,得到 mat0, mat1
            mat0, mat1 = bin_split_data_set(data_set, feat_index, split_val)

            # 如果切分的之后的任意一个矩阵的的数据量小于设置的最小切分数,跳过本轮遍历
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
        # print("最佳切分的两个矩阵的总方差的和与未切分的总发差的差值小于预期值,返回None")
        return None, leaf_type(data_set)

    # 使用最佳分隔特征,以及最佳分隔的特征值来切分数据集,
    mat0, mat1 = bin_split_data_set(data_set, best_index, best_value)

    # 判断最佳分隔的结果是否小于最小切分样本数
    if (np.shape(mat0)[0] < tol_n) or (np.shape(mat1)[0] < tol_n):
        # print("最佳切分的任意一个矩阵的的数据量小于预期的最小切分数,返回None")
        return None, leaf_type(data_set)
    return best_index, best_value


def create_tree(data_set, leaf_type=reg_leaf, err_type=reg_err, ops=(1, 4)):
    """
    创建树
    :param data_set:  带分隔的数据集
    :param leaf_type: 叶子类型,分隔函数
    :param err_type:  错误类型,
    :param ops:
    :return:
    """
    feature, value = choose_best_split(data_set, leaf_type, err_type, ops)
    if feature is None:
        print("切分之后,接受到的feature=None,返回子集的最后一列的平均值")
        return value

    ret_tree = {'split_index': feature, 'split_value': value}
    # print('切分数据集')
    l_set, r_set = bin_split_data_set(data_set, feature, value)
    # print('创建左子树')
    ret_tree['left'] = create_tree(l_set, leaf_type, err_type, ops)
    # print('创建右子树')
    ret_tree['right'] = create_tree(r_set, leaf_type, err_type, ops)
    return ret_tree


def is_tree(obj):
    """
    测试输人变量是否是一棵树，返回布尔类型的结果。
    换句话说 ，该函数用于判断当前处理的节点是否是叶节点。
    :param obj:
    :return:
    """
    return type(obj).__name__ == 'dict'


def get_mean(tree):
    """
    函数getMean()是一个递归函数，它从上往下遍历树直到叶节点为止。
    如果找到两个叶节点则计算它们的平均值。
    该函数对树进行塌陷处理（即返回树平均值)，在prune()函数中调用该函数时应明确这一点。
    :param tree:
    :return:
    """
    if is_tree(tree['right']):
        tree['right'] = get_mean(tree['right'])

    if is_tree(tree['left']):
        tree['left'] = get_mean(tree['left'])

    return (tree['left'] + tree['right']) / 2.0


def prune(tree, test_data):
    """
    回归树剪枝函数
    后剪枝技术
    递归调用函数对测试数据进行切分
    :param tree:
    :param test_data:
    :return:
    """
    # 如果数据为空,返回树的平均值
    if np.shape(test_data)[0] == 0:
        return get_mean(tree)

    # 如果左子树或者右子树不是叶子节点, 对test_data进行分割,得到2个子集
    if is_tree(tree['right']) or is_tree(tree['left']):
        left_set, right_set = bin_split_data_set(test_data, tree['split_index'], tree['split_value'])

    # 如果左子树不是叶子, 递归调用本方法
    if is_tree(tree['left']):
        tree['left'] = prune(tree['left'], left_set)

    # 如果右子树不是叶子, 递归调用本方法
    if is_tree(tree['right']):
        tree['right'] = prune(tree['right'], right_set)

    # 如果左子树和右子树都是叶子
    if not is_tree(tree['left']) and not is_tree(tree['right']):
        # 按照最佳分隔 获得 数据集的两个子集
        left_set, right_set = bin_split_data_set(test_data, tree['split_index'], tree['split_value'])
        # 得到两个子集总方差的和
        error_no_merge = np.sum(np.power(left_set[:, -1] - tree['left'], 2)) + \
                         np.sum(np.power(right_set[:, -1] - tree['right'], 2))
        # 得到两个子集的平均值的平均值
        tree_mean = (tree['left'] + tree['right']) / 2.0
        # 不分割的总方差
        error_merge = np.sum(np.power(test_data[: -1] - tree_mean, 2))
        # 如果不分割的总方差比分隔之后子集的总方差的和小,则返回 两个子集的平均值的平均值
        # 意义: 如果两个数据子集合并后的误差比不合并的误差小,则进行合并,否则不合并直接返回
        if error_merge < error_no_merge:
            print('merging')
            return tree_mean
        # 否则,返回当前子树
        else:
            return tree
    else:
        return tree


def linear_solve(data_set):
    """
    功能: 将数据集格式化成为目标变量Y和自变量X, 以及获得模型参数系数
    如果矩阵的逆不存在,会抛出异常
    :param data_set:
    :return:
    """
    # 获得行数m, 列数n
    m, n = np.shape(data_set)
    # 获得m行,n列的全1矩阵X
    x = np.mat(np.ones((m, n)))
    # 获得m行1列的全1矩阵Y
    y = np.mat(np.ones((m, 1)))
    # 将data_set的前面n-1列的数据复制到x的后面n-1列中
    # 也就是x第一列全1, 后面的n-1列的数据就是data_set的前n-1列的数据
    x[:, 1:n] = data_set[:, 0:n - 1]
    # data_set最后一列的数据保存到y中
    y = data_set[:, -1]

    xtx = x.T * x
    # np.linalg.det() 矩阵求行列式(标量)
    if np.linalg.det(xtx) == 0.0:
        raise NameError('This matrix is singular, '
                        'cannot do inverse, try increasing the second value of ops')
    # 计算模型系数
    ws = xtx.I * (x.T * y)
    return ws, x, y


def model_leaf(data_set):
    """
    模型树
    当数据不再需要切分的时候
	负责生成叶子节点
	在数据集合上调用linear_solve(), 返回回归系数ws
	:param data_set:
	:return:   当给定的数据集不可以再次切分时, 返回回归系数ws
	"""
    ws, x, y = linear_solve(data_set)
    return ws


def model_err(data_set):
    """
	在给定的数据集上计算误差
	:param data_set:  给定的数据集
	:return:  在给定的数据集上返回推测结果和实际结果的误差的平方和
	"""
    ws, x, y = linear_solve(data_set)
    # 计算得到推测的结果值y_hat
    y_hat = x * ws
    # 返回推测结果和实际结果的误差的平方和
    return np.sum(np.power(y - y_hat, 2))


def reg_tree_eval(model, in_dat):
    """
    :param model:
    :param in_dat:
    :return:
    """
    return float(model)


def model_tree_eval(model, in_dat):
    """

    :param model:
    :param in_dat:
    :return:
    """
    # 数据集的列数
    n = np.shape(in_dat)[1]
    # 构建一个一行n+1列的全1矩阵
    x = np.mat(np.ones((1, n+1)))
    # 将in_dat的数据放入x矩阵中,保留第一列为1
    x[:, 1:n+1] = in_dat
    #
    return float(x*model)


def tree_fore_cast(tree, in_data, model_eval=reg_tree_eval):
    """

    :param tree:
    :param in_data:
    :param model_eval:
    :return:
    """
    # 如果当前是树节点
    if not is_tree(tree):
        return model_eval(tree, in_data)
    if in_data[tree['split_index']] > tree['split_value']:
        if is_tree(tree['left']):
            return tree_fore_cast(tree['left'], in_data, model_eval)
        else:
            return model_eval(tree['left'], in_data)
    else:
        if is_tree(tree['right']):
            return tree_fore_cast(tree['right'], in_data, model_eval)
        else:
            return model_eval(tree['right'], in_data)


def create_fore_cast(tree, test_data, model_eval=reg_tree_eval):
    """

    :param tree:
    :param test_data:
    :param model_eval:
    :return:
    """
    # 得到数据量m
    m = len(test_data)
    # m行1列全0矩阵
    y_hat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        y_hat[i, 0] = tree_fore_cast(tree, np.mat(test_data[i]), model_eval)
    return y_hat