"""
朴素贝叶斯通常有两种实现方式:
    1. 基于贝努利模型实现
        并不考虑词在文档中出现的次数,只考虑出不出现
        因此在这个意义上相当于假设词是等权重的
    2. 基于多项式模型实现
         会考虑词在文档汇总出现的次数
"""

import numpy as np

def load_data_set():
    """

    :return: 返回一个实验样本
        返回的第一个变量: 进行词条切分后的文档集合
        返回的第二个变量: 一个类别标签的集合 , 这里有两类,侮辱性的和非侮辱性的
                        这些文本的类别由人工标注,这些标注信息用于训练程序以便自动检测侮辱性留言
    """
    posting_list = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'grabage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]

    class_vec = [0, 1, 0, 1, 0, 1] # 1 代表侮辱性文字, 0 代表正常言论
    return posting_list, class_vec


def create_vocab_list(data_set):
    """
    创建 一个包含在所有文档中出现的不重复词的列表
    :param data_set:
    :return:
    """
    vocab_set = set([])         # 创建一个空集
    for document in data_set:
        vocab_set = vocab_set | set(document)  # 创建两个集合的并集, 在数学上, 按位或操作(OR)与集合合并操作使用相同符号

    return list(vocab_set)


def set_of_words_2_vec(vocab_list, input_set):
    """
    词集模型(set-of-words-model) : 将每个词的出现与作为一个特征
        如果一个词在文档中出现次数不止一次, 这可能意味着该词是否出现在文档中所不能表达的某种信息, 这种信息在词集模型中不能展现

    根据函数create_vocab_list() 创建的词汇表,
    去获取词汇表中的单词是否在入的某个文档input_set中出现,
    返回一个文档向量, 用0/1 来表示是否出现

    :param vocab_list: 词汇表
    :param input_set:  输入的某个文档
    :return:  文档向量, 向量的每一个元素为1 或者 0 , 分别表示词汇表中的单词在输入文档中是否出现
    """
    return_vect = [0] * len(vocab_list)   # 创建一个其中所含元素都是0的向量,和词汇表相等长度
    for word in input_set:
        if word in vocab_list:
            return_vect[vocab_list.index(word)] = 1
        else:
            print('the word: %s is not in my vocabulary!'% word)

    return return_vect


def bag_of_words_2_vec(vocab_list, input_set):
    """
    词袋模型, 每个单词可以出现多次, 将每个词出现的次数作为特征
    :param vocab_list:
    :param input_set:
    :return:
    """
    return_vect = [0] * len(vocab_list)   # 创建一个其中所含元素都是0的向量,和词汇表相等长度
    for word in input_set:
        if word in vocab_list:
            return_vect[vocab_list.index(word)] += 1
        else:
            print('the word: %s is not in my vocabulary!'% word)

    return return_vect


def train_naive_bayes0(train_matrix, train_category):
    """
    朴素贝叶斯分类器的训练函数
    :param train_matrix:    文档矩阵向量, 存储的应该是 多个真实文档的词在词汇表中是否出现的集合
    :param train_category:  每篇文档类别标签所构成的向量
    :return:
    """
    num_train_docs = len(train_matrix)   #有多少文档

    num_words = len(train_matrix[0])     # 每篇文档对应的词汇表, 词汇表的长度

    # sum(train_category) 计算train_category中所有元素想加的结果, 0/1值, 则只结算1的值,就是侮辱性类别的数量
    p_abusive = sum(train_category) / float(num_train_docs)  # 侮辱性的类别的 占比  p(1)

    p_0_num = np.ones(num_words)

    p_1_num = np.ones(num_words)

    p_0_denom = 2.0    #  p(0) 分母项   非侮辱类别

    p_1_demon = 2.0    #  p(1)  分母项  侮辱类别

    for i in range(num_train_docs):
        if train_category[i] == 1:               # 侮辱类别
            p_1_num += train_matrix[i]           #
            p_1_demon += sum(train_matrix[i])    # 出现了多少词汇
        else:                                    # 非侮辱类别
            p_0_num += train_matrix[i]
            p_0_denom += sum(train_matrix[i])

    p_1_vect = np.log(p_1_num / p_1_demon)
    p_0_vect = np.log(p_0_num / p_0_denom)

    return p_0_vect, p_1_vect, p_abusive


def classify_native_bayes(vec_2_classfiy, p0_vec, p1_vec, p_class1):
    """
       朴素贝叶斯分类器
    :param vec_2_classfiy:
    :param p0_vec:
    :param p1_vec:
    :param p_class1:
    :return:
    """
    p1 = sum(vec_2_classfiy * p1_vec) + np.log(p_class1)
    p0 = sum(vec_2_classfiy * p0_vec) + np.log(1.0 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0


def text_parse(big_text, min_tok_len=1):
    """
    切分英文文档
    :param big_text:
    :param min_tok_len: 英文单词最小长度, 默认为0 , 可以指定英文单词长度小于这个值的单词不予统计到结果中
    :return:  将英文文档转换成全是英文单词的list, 全小写,去掉标点符号
    """
    import re
    reg_ex = re.compile('\\W{1, }')   # 只匹配字母
    list_of_tokens = reg_ex.split(big_text)
    ret_val = [tok.lower() for tok in list_of_tokens if len(tok) > min_tok_len]
    return ret_val


