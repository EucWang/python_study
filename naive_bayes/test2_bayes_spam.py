from naive_bayes.bayes import *


def test_spam():
    """
    垃圾邮件分类测试
    :return:
    """
    doc_list = []
    class_list = []
    full_text = []
    email_dir = 'email/'
    spam_dir = 'spam/'
    ham_dir = 'ham/'

    for i in range(1, 26):
        txt_read = open(email_dir + spam_dir + '%d.txt' % i, encoding='gb18030', errors='ignore').read()
        word_list = text_parse(txt_read)
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)

        txt_read = open(email_dir + ham_dir + '%d.txt' % i, encoding='gb18030', errors='ignore').read()
        word_list = text_parse(txt_read)
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)

    vocab_list = create_vocab_list(doc_list)  # 根据文档获取词汇表
    training_set = [index for index in range(0, 50)]   # 训练数据集的索引
    test_set = []              # 测试数据集的索引
    for i in range(10):  # 随机从50个训练数据集的索引中取10个数据放入到测试数据集的索引中,训练数据集保留40个数据
        rand_index = int(np.random.uniform(0, len(training_set)))  # 从0 到50 随机取一个值
        test_set.append(training_set[rand_index])
        del(training_set[rand_index])

    train_mat = []
    train_classes = []

    for doc_index in training_set:
        # 对照词汇表,生成词汇向量,做为行数据放入到train_mat矩阵中
        train_mat.append(bag_of_words_2_vec(vocab_list, doc_list[doc_index]))
        train_classes.append(class_list[doc_index])

    # 使用训练数据生成 p0_v, p1_v, p_spam
    p0_v, p1_v, p_spam = train_naive_bayes0(np.array(train_mat), np.array(train_classes))
    err_count = 0

    # 使用测试数据进行测试
    # 查看朴素贝叶斯分类器的错误率
    for doc_index in test_set:
        word_vector = bag_of_words_2_vec(vocab_list, doc_list[doc_index])
        class_native_bayes = classify_native_bayes(np.array(word_vector), p0_v, p1_v, p_spam)
        if class_native_bayes != class_list[doc_index]:
            err_count += 1

    print('the error rate is :', float(err_count)/len(test_set))
    return float(err_count)/len(test_set)


errs = 0.0
for i in range(5000):
    errs += test_spam()
print(errs/5000.0)
# 执行5000次, 错误率在0.4110599999999955
# 错误率也蛮高的


