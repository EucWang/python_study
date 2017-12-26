from math import log
import operator

def calc_shannon_ent(data_set):
    """功能:
            计算所有类别所有可能值包含的信息期望值
       目的:
            为了计算给定数据集的熵:entropy
        """
    # 数据量
    num_entries = len(data_set)
    label_counts = {}

    # 遍历, 统计所有类标签发生的次数
    for feat_vec in data_set:
        # 数据集中每一行最后一列元素就是标签
        current_label = feat_vec[-1]

        if current_label not in label_counts.keys():
            label_counts[current_label] = 0

        label_counts[current_label] += 1

    shannon_ent = 0.0
    # 对每一种分类遍历, 计算所有类别所有可能值包含的信息期望值
    for key in label_counts:
        # 每一种类标签的概率值
        prob = float(label_counts[key]) / num_entries
        shannon_ent -= prob * log(prob, 2)

    return shannon_ent


def split_data_set(data_set, axis, value):
    """接受三个输入参数:
        @:param : data_set 待划分的数据集
        @:param : axis 划分数据集的特征
        @:param : value 需要返回的特征的值
    """
    ret_data_set = []

    # 对数据集进行遍历
    for feat_vec in data_set:
        # 如果 给定的特征 符合要求的值,则将这一条数据添加到新的数据集中,
        # 新数据集中,不包含给定的特征
        if feat_vec[axis] == value:
            reduced_feat_vect = feat_vec[:axis]

            # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
            reduced_feat_vect.extend(feat_vec[axis + 1:])

            ret_data_set.append(reduced_feat_vect)

    return ret_data_set


def choose_best_feature_to_split(data_set):
    """选择最好的数据集划分方式    ID3算法
        @:param data_set:  参数必须满足要求:
            1. 数据必须是一种由列表元素组成的列表，而且所有的列表元素都要具有相同的数据长度
            2. 数据的最后一列或者每个实例的最后一个元素是当前实例的类别标签"""

    # 计算所有数据的香农熵
    base_entropy = calc_shannon_ent(data_set)
    best_info_gain = 0.0
    best_feature = -1

    for i in range(len(data_set[0]) - 1):
        # 除了 最后一列,遍历每一列
        # 获取每一列数据
        feature_list = [example[i] for example in data_set]

        # 数据集中, 统一特性, 所有取值, 去重
        unique_vals = set(feature_list)
        new_entropy = 0.0

        # 遍历一个特征值的所有可能值
        for value in unique_vals:
            # 总的数据集中,当前特征不为给定的特征值的其他所有数据
            # 同时去掉了当前特征列
            sub_data_set = split_data_set(data_set, i, value)

            # 计算子数据集在总数据集中的占比
            prob = len(sub_data_set) / float(len(data_set))
            # 计算特征值的期望值
            new_entropy += prob * calc_shannon_ent(sub_data_set)

            # 得到信息增益
            info_gain = base_entropy - new_entropy

            # 比较信息增益, 直到某一个特征的某一个给定特征值能获取到最大信息增益
            # 记录信息增益最大的特征的索引
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = i

    return best_feature


def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0

        class_count[vote] += 1

    sorted_class_count = sorted(class_count.items(), key=operator.itengetter(1), reverse=True)
    return sorted_class_count