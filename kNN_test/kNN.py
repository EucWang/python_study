import numpy as np
import operator
import matplotlib.pyplot as plt


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


def file2matrix(filename):
    """从文本文件中读取内容,转换成matrix"""
    try:
        with open(filename) as fr:
            # 读取所有行
            array_o_lines = fr.readlines()
            # 行数
            number_of_lines = len(array_o_lines)

            # 构建一个全部为0 的矩阵
            return_mat = np.zeros((number_of_lines, 3))
            # 构建一个标签集
            class_label_vector = []
            # 索引
            index = 0

            # 遍历
            for line in array_o_lines:
                line = line.strip()  # 去掉文本中前后的空格
                list_from_line = line.split('\t')  # 按照\t来分隔文本内容
                return_mat[index, :] = list_from_line[0:3]  # 将前3个元素的内容传递给全为0的矩阵
                class_label_vector.append((int(list_from_line[-1])))  # 将行的最后一个内容传递给标签集
                index += 1
    except BaseException as e:
        print(e)
        return None, None
    else:
        return return_mat, class_label_vector


def show_scatter_diagram(mat_data1, mat_data2, labels, title='', x_title='', y_title=''):
    """显示散点图"""
    figure = plt.figure()
    # ax = figure.add_subplot(111)
    # ax.scatter(mat_data1, mat_data2)

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # ax.title = title
    # ax.x_title = x_title
    # ax.y_title = y_title
    labelArr = np.array(labels)

    plt.scatter(mat_data1, mat_data2, 15.0 * labelArr, 15.0 * labelArr)
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)

    # plt.plot(X, C, color="blue", linewidth=2.5, linestyle="-", label="cosine")
    plt.show()


def show_scatter_diagram2(mat_data1, mat_data2, labels, title='', x_title='', y_title=''):
    """优化之后的显示散点图函数"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure(figsize=(10, 6), dpi=80)
    axes = plt.subplot(111)

    type1_x = []
    type1_y = []

    type2_x = []
    type2_y = []

    type3_x = []
    type3_y = []

    for i in range(len(labels)):
        if labels[i] == 1:
            type1_x.append(mat_data1[i])
            type1_y.append(mat_data2[i])
        elif labels[i] == 2:
            type2_x.append(mat_data1[i])
            type2_y.append(mat_data2[i])
        elif labels[i] == 3:
            type3_x.append(mat_data1[i])
            type3_y.append(mat_data2[i])

    type1 = axes.scatter(type1_x, type1_y, s=20, c='red')
    type2 = axes.scatter(type2_x, type2_y, s=20, c='green')
    type3 = axes.scatter(type3_x, type3_y, s=20, c='blue')
    axes.legend((type1, type2, type3), (u'不喜欢', u'魅力一般', u'极具魅力'), loc=2)

    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)

    plt.show()


def auto_norm(data_set):
    """对矩阵数值进行归一化处理, 所有数值全部设置为0~1,或者 -1 ~ 1 之间"""

    # matrix.min(0)  根据列获取最小值
    min_vals = data_set.min(0)
    # matrix.max(0)  根据列获取最大值
    max_vals = data_set.max(0)

    # 特征相减之后的 范围值
    ranges = max_vals - min_vals

    # 生成一个和样本数据集大小相同的矩阵
    norm_data_set = np.zeros(np.shape(data_set))

    # 获取矩阵有多少行
    m = data_set.shape[0]

    # 生成一个m行1列的用min_valus填充的矩阵
    min_vals_mat = np.tile(min_vals, (m, 1))

    # 样本数据集的每一个元素减去min_values
    norm_data_set = data_set - min_vals_mat

    # 生成一个m行1列的用ranges填充的矩阵
    ranges_mat = np.tile(ranges, (m, 1))

    # 相除
    norm_data_set = norm_data_set / ranges_mat

    return norm_data_set, ranges, min_vals

# group, labels = create_data_set()
# result = classify0([0, 0], group, labels, 3)
# print(result)

mat, labels = file2matrix('datingTestSet2.txt')
show_scatter_diagram2(mat[:, 0], mat[:, 1], labels, title='约会对象分类', y_title='游戏时间百分比', x_title='飞行时长')
# show_scatter_diagram(mat[:, 1], mat[:, 2], labels, title='约会对象分类', x_title='游戏时间百分比', y_title='冰激凌消耗公升数')
