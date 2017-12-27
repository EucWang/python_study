from math import log
import operator


def calc_shannon_ent(data_set):
    """功能:
            计算所有类别所有可能值包含的信息期望值
            也就是计算数据集的香农熵
       目的:
            为了计算给定数据集的熵:entropy
        @:param data_set : 普通的python list集合,不是numpy的ndarray或者matrix
        """
    # 数据集的数据量
    num_entries = len(data_set)
    # 不同分类的标签字典统计
    label_counts = {}

    # 遍历, 统计所有类标签发生的次数
    for feat_vec in data_set:
        # 数据集中每一行最后一列元素就是标签
        current_label = feat_vec[-1]

        if current_label not in label_counts.keys():
            label_counts[current_label] = 0

        label_counts[current_label] += 1

    # 结果的香农熵
    shannon_ent = 0.0
    # 对每一种标签进行遍历, 计算所有类别所有可能值包含的信息期望值
    for key in label_counts:
        # 每一种类标签的概率值
        prob = float(label_counts[key]) / num_entries
        # 通过公式 获取该标签的信息期望值
        # 累加之后的结果就是香农熵值
        shannon_ent -= prob * log(prob, 2)

    return shannon_ent


def split_data_set(data_set, axis, value):
    """
    功能:
        划分数据集,
        为了度量划分数据集的熵, 以便判断当前是否正确的划分了数据集
    接受三个输入参数:
        @:param : data_set 待划分的数据集, python list数据
        @:param : axis 划分数据集的特征,   对应数据集的列索引值
        @:param : value 需要返回的特征的值 axis指定的列, 这一列中所有取值中的特定值
    """
    ret_data_set = []

    # 对数据集进行遍历
    for feat_vec in data_set:
        # 如果 给定的特征 符合要求的值,则将这一条数据添加到新的数据集中,
        # 新数据集中,不包含给定的特征
        if feat_vec[axis] == value:
            # 删减之后的这一行中的axis这一列的前面所有数据
            reduced_feat_vect = feat_vec[:axis]

            # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
            # 这一行中的axis这一列之后的所有数据
            reduced_feat_vect.extend(feat_vec[axis + 1:])

            # 将 这一列的值为value的其他特征值的数据加入到待返回的数据集中
            ret_data_set.append(reduced_feat_vect)

    return ret_data_set


def choose_best_feature_to_split(data_set):
    """选择最好的数据集划分方式    ID3算法
        @:param data_set:  参数必须满足要求:
            1. 数据必须是一种由列表元素组成的列表，而且所有的列表元素都要具有相同的数据长度
            2. 数据的最后一列或者每个实例的最后一个元素是当前实例的类别标签"""

    # 计算所有数据的香农熵
    base_entropy = calc_shannon_ent(data_set)
    # 最好的信息增益值,
    best_info_gain = 0.0
    # 最好用于划分的特征的索引值, 对应数据集的列的索引
    best_feature = -1

    for i in range(len(data_set[0]) - 1):
        # 除了 最后一列,遍历每一列, 最后一列是分类标签
        # 获取第i列的所有数据组成一个列表
        feature_list = [example[i] for example in data_set]

        # 对第i列数据去重,获取一个Set集合
        unique_vals = set(feature_list)
        # 新的数据集的熵
        new_entropy = 0.0

        # 遍历第i列所有去重之后的特征值的所有可能值
        for value in unique_vals:

            # 去掉第i列特征, 符合给定特征取值的子集合
            sub_data_set = split_data_set(data_set, i, value)

            # 计算子数据集在总数据集中的占比
            prob = len(sub_data_set) / float(len(data_set))
            # 计算特征值的期望值, 这就是子集合的熵
            new_entropy += prob * calc_shannon_ent(sub_data_set)

            # 总集合的熵减去自己和的熵, 得到信息增益
            info_gain = base_entropy - new_entropy

            # 比较信息增益, 直到某一个特征的某一个给定特征值能获取到最大信息增益
            # 信息增益越大,说明从子集合扩展到总集合的过程,越无序,
            # 也说明, 在总集合一定的情况下, 子集合的熵越小, 子集合越有序
            # 记录信息增益最大的特征的索引
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = i

    return best_feature


def majority_cnt(class_list):
    """功能:
            返回给定list集合中出现次数最多的分类名称
        @:param
            class_list 分类集合的字典
        """

    # 统计标签以及标签在集合中的出现的数量
    class_count = {}

    # 对集合元素进行遍历, 将集合元素作为key存储到class_count中
    for vote in class_list:
        # 如果不存在,初始设置为0
        if vote not in class_count.keys():
            class_count[vote] = 0
        # 存在,则自增1
        class_count[vote] += 1

    # 按照 从大到小的顺序, 对 数量进行排序
    sorted_class_count = sorted(class_count.items(), key=operator.itengetter(1), reverse=True)
    # 返回 数量最大的key,也就是标签
    return sorted_class_count[0][0]


def create_desicion_tree(data_set, labels):
    """创建决策树"""

    # 获取当前集合的分类标签的列表
    class_list = [example[-1] for example in data_set]

    # 类别完全相同,则停止继续划分
    # 这里比较分类标签的第一个元素在分类标签集合中出现的次数和分类标签的总数目如果相同
    # 表示这个集合中只有一个分类标签
    # list.count(x) : 计算x在list集合中出现的次数
    if class_list.count(class_list[0]) == len(class_list):
        # 返回这个标签
        return class_list[0]

    # 如果数据集合的列数为1 ,表示这个数据集合已经没有特征存在了,只剩下分类标签了
    if len(data_set[0]) == 1:
        # 返回分类标签出现次数最多的标签
        return majority_cnt(class_list)

    # 对这个集合进行划分比较,获取到最佳划分的特征的索引
    index_best_feature = choose_best_feature_to_split(data_set)

    # 得到最佳划分的特征的标签
    best_feature_label = labels[index_best_feature]

    # 构建 返回的决策树, 指定label作为key
    my_tree = {best_feature_label: {}}

    # 从labels中删除掉对应的最佳划分的特征的标签
    del (labels[index_best_feature])

    # 从集合中取出最佳划分的特征的所有唯一特征值
    feature_values = [example[index_best_feature] for example in data_set]
    unique_vals = set(feature_values)

    # 遍历 最佳划分的特征的所有唯一特征值
    for value in unique_vals:
        # 获取 删除掉最佳划分的特征的标签的 labels备份
        sub_lables = labels[:]

        # 获取给定特征以及给定特征值的子集合
        sub_data_set = split_data_set(data_set, index_best_feature, value)

        # 对这个子集合递归处理,将结果放入到 决策树的字典中
        my_tree[best_feature_label][value] = create_desicion_tree(sub_data_set, sub_lables)

    return my_tree


def classify(input_tree, feature_labels, test_vec):
    """
    使用决策树的分类函数
    :param input_tree:      决策树
    :param feature_labels:  特征标签集合,用于确定 特征标签 在样本数据集中的索引值,也就是特征的列索引值
    :param test_vec:        需要测试的数据
    :return:
    """
    # 决策树第一个决策点的label
    first_str = list(input_tree.keys())[0]
    # 对应这个label下的下一级字典
    second_dict = input_tree[first_str]
    # 获取这个 label 的索引
    feature_index = feature_labels.index(first_str)

    # 遍历这个字典
    for key in second_dict.keys():

        # 如果测试数据的对应特征值 和决策树当前的值相同
        if test_vec[feature_index] == key:
            if type(second_dict[key]).__name__ == 'dict':  # 如果下一级依然是一个字典
                # 递归调用分类方法
                class_label = classify(second_dict[key], feature_labels, test_vec)
            else:
                # 获取 决策树 叶子节点的值, 也就是通过决策树推断的测试数据的分类标签
                class_label = second_dict[key]

    return class_label


def store_decision_tree(input_tree, filename):
    """
    保存决策树数据到文件中
    :param input_tree:
    :param filename:
    :return:
    """
    import pickle
    try:
        with open(filename, 'w') as fw:
            pickle.dump(input_tree, fw)
    except BaseException as e:
        print(e)


def grab_decision_tree(filename):
    """
    从文件中读取决策树
    :param filename:
    :return:
    """
    import pickle
    try:
        fr = open(filename)
    except BaseException as e:
        print(e)
    else:
        return pickle.load(fr)


def test_decision_tree():
    from decision_tree_test.decision_tree_plotter import create_plot2
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]  # 样本数据
    lense_labels = ['age', 'prescript', 'astigmatic', 'tearRate']   # 样本数据的特征label集合
    lenses_tree = create_desicion_tree(lenses, lense_labels)        # 创建决策树
    create_plot2(lenses_tree)


test_decision_tree()