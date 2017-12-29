from naive_bayes.bayes import *


def test():
    list_of_posts, list_classes = load_data_set()
    my_vocab_list = create_vocab_list(list_of_posts)
    print(my_vocab_list)
    # print(list_classes)
    words___vec = set_of_words_2_vec(my_vocab_list, list_of_posts[3])
    # print(words___vec)

    train_mat = []
    for postin_doc in list_of_posts:
        train_mat.append(set_of_words_2_vec(my_vocab_list, postin_doc))

    p_0_v, p_1_v, p_abusive = train_naive_bayes0(train_mat, list_classes)
    print('p0v', p_0_v)
    print('p1v', p_1_v)
    print('p_abusive', p_abusive)


def testing_native_bayes():
    """
    完整的测试 朴素贝叶斯 的 训练器函数 和 分类器函数
    :return:
    """
    list_o_posts, list_classes = load_data_set()     # 样本数据
    my_vocab_list = create_vocab_list(list_o_posts)  # 根据样本数据获取的词汇表
    train_mat = []
    for postin_doc in list_o_posts:
        words___vec = set_of_words_2_vec(my_vocab_list, postin_doc)  # 每一份文章对应到词汇表的向量 0/1
        train_mat.append(words___vec)                 # 作为一行数据加入到矩阵中

    # 通过贝叶斯训练器,
    # 获取已知不同分类下的, 各个词汇的比率
    # 以及 样本数据中指定分类的比率
    p0_v, p1_v, p_abusive = train_naive_bayes0(np.array(train_mat), np.array(list_classes))

    # 测试数据
    test_entry = ['love', 'my', 'dalmation']
    this_doc__vec = np.array(set_of_words_2_vec(my_vocab_list, test_entry)) # 获取测试样本到词汇表的向量 0/1
    # 通过分类器,对测试数据进行分类
    native_bayes_result = classify_native_bayes(this_doc__vec, p0_v, p1_v, p_abusive)
    print(test_entry, "classified as", native_bayes_result)

    # 测试数据2
    test_entry = ['stupid', 'garbage']
    this_doc__vec = np.array(set_of_words_2_vec(my_vocab_list, test_entry))
    native_bayes_result = classify_native_bayes(this_doc__vec, p0_v, p1_v, p_abusive)
    print(test_entry, "classified as", native_bayes_result)


#testing_native_bayes()