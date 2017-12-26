import numpy as np
import operator

def create_data_set():
    """创建样本数据"""
    group = np.array([[1.0, 1.1],
              [1.0, 1.0],
              [0, 0],
              [0, 0.1]])

    # 数据对应的标签
    labels = ['A', 'A', 'B', 'B']

    return group, labels

def classify0(in_x, data_set, labels, k):
    """ @:param in_x : 用于分类的输入向量
        @:param data_set: 输入的训练样本
        @:param labels: 训练样本对应的标签向量
        @:param k: 用于选择最近邻居的数目
    """
    # 数据的行数,就是表示数据的总量
    data_set_size = data_set.shape[0]

    # 将in_x扩展成一个矩阵,行数和训练样本相同
    in_xs = np.tile(in_x, (data_set_size, 1))

    # 将扩展之后的矩阵 和训练样本矩阵相减, 各个元素相减
    diff_mat = in_xs - data_set

    #  相减之后的结果 各个元素平方
    sq_diff_mat = diff_mat ** 2

    # 每一行的各个元素进行想加, 得到一个n行1列的矩阵
    sq_distances = sq_diff_mat.sum(axis=1)

    # 对这个n行1列的矩阵每一行元素进行开方, 得到 in_x 和每一行训练样本的距离
    distances = sq_distances ** 0.5

    # 对距离结果进行排序
    # argsort() : 函数返回的是数组值从小到大的索引值的一个数组
    sorted_dist_indicies = distances.argsort()

    class_count = {}  # 一个字典, 统计前k个距离里,对应的各个标签的数量

    # 对排序之后的结果,取前面k个元素, 获取k个元素的距离的一个标签统计信息
    for i in range(k):
        # 距离从小到大排序之后, 这个获取各个距离数值的在原矩阵中的索引值
        dist_i = sorted_dist_indicies[i]
        # 获取排序之前,计算得到的各个距离对应的标签
        vote_i_label = labels[dist_i]

        # 在字典中对这个标签+1
        # get(key, default=None), 这里的传值0 表示如果没有就返回0
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1

    # class_count.items() 获取字典的一个键值对
    # key=np.operator.itemgetter(1) 根据键值对的值来进行排序,也就是第二列进行排序
    # reverse=True 默认排序结果时从小到大排序,这里表示倒序, 排序为从大到小
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)

    # 返回字典排序之后的结果
    return sorted_class_count[0][0]

group, labels = create_data_set()
result = classify0([0, 0], group, labels, 3)
print(result)
