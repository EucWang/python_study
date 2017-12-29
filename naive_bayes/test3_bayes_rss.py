"""
示 例 ：
    使用朴素贝叶斯分类器从个人广告中获取区域倾向
"""

import feedparser
import operator
import numpy as np

from naive_bayes.bayes import *


def calc_most_freq(vocab_list, full_text, num_of_top=30):
    """
    :param vocab_list:  总词典
    :param full_text:   文本
    :return:  遍历词典, 统计每个词在文本中出现的次数, 返回出现次数最多的30个词
    """
    freq_dict = {}
    for token in vocab_list:
        freq_dict[token] = full_text.count(token)

    sorted_freq = sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_freq[:num_of_top]  # 返回前30个高频词


ENTRIES = 'entries'
SUMMARY = 'summary'


def local_words(feed1, feed0):
    doc_list = []
    class_list = []
    full_text = []

    min_len = min(len(feed1[ENTRIES]), len(feed0[ENTRIES]))  # 两个rss源 获取的entries 的最小长度
    print('min_len', min_len)

    for index in range(min_len):
        word_list = text_parse(feed1[ENTRIES][index][SUMMARY])  # 分拆文档为单词的list
        doc_list.append(word_list)         # 作为一行数据加入到doc_list
        full_text.extend(word_list)        # 将每一单词加入到full_text
        class_list.append(1)

        word_list = text_parse(feed0[ENTRIES][index][SUMMARY])  # 分拆文档为单词的list
        doc_list.append(word_list)         # 作为一行数据加入到doc_list
        full_text.extend(word_list)        # 将每一单词加入到full_text
        class_list.append(0)

    vocab_list = create_vocab_list(doc_list)  # 生成词典
    top30_words = calc_most_freq(vocab_list, full_text, 30)  # 高频30词, 这个值动态改变,可以发现对结果影响很大
    for pair_w in top30_words:     # 从词典中去掉高频的30个词
        if pair_w[0] in vocab_list:
            vocab_list.remove(pair_w[0])

    # 另一个常用的方法是不仅移除高频词,
    # 同时从某个预订词表中移除结构上的辅助词, 这样的词表 称为 停用词表(stop word list)
    # 目前可以找到很多停用词表

    train_set = [index for index in range(2 * min_len)]
    test_set = []

    for index in range(20):
        rand_index = int(np.random.uniform(0, len(train_set)))
        test_set.append(rand_index)
        del(train_set[rand_index])

    train_mat = []
    train_classes = []

    for doc_index in train_set:
        train_mat.append(bag_of_words_2_vec(vocab_list, doc_list[doc_index]))
        train_classes.append(class_list[doc_index])

    p0_v, p1_v, p_spam = train_naive_bayes0(np.array(train_mat), np.array(train_classes))

    err_count = 0
    for index in test_set:
        word_vector = bag_of_words_2_vec(vocab_list, doc_list[index])
        class_of_test_index = classify_native_bayes(np.array(word_vector), p0_v, p1_v, p_spam)
        if class_of_test_index != class_list[index]:
            err_count += 1

    err_rate = float(err_count) / len(test_set)
    print('the error rate is', )

    return vocab_list, p0_v, p1_v, err_rate


def get_top_words(need1, need0):
    vocab_list, p0_v, p1_v, err_rate = local_words(need1, need0)
    top_need1 = []
    top_need0 = []
    print(p0_v)
    print(p1_v)
    for i in range(len(p0_v)):
        if p0_v[i] > -6.0:
            top_need0.append((vocab_list[i], p0_v[i]))
        if p1_v[i] > -6.0:
            top_need1.append((vocab_list[i], p1_v[i]))

    sorted_sf = sorted(top_need0, key=lambda pair: pair[1], reverse=True)
    print('\nSF**SF**SF**SF**SF**SF**SF**SF**SF**SF**^')
    for item in sorted_sf:
        print(item[0])
    print('SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**_')

    sorted_ny = sorted(top_need1, key=lambda pair: pair[1], reverse=True)
    print('\nNY**NY**NY**NY**NY**NY**NY**NY**NY**NY**^')
    for item in sorted_ny:
        print(item[0])
    print('NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**_')


def test_local_words():
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')

    err_rate_ = 0.0
    times = 500
    for i in range(times):
        vocab_list, p0_v, p1_v, err_rate = local_words(ny, sf)
        err_rate_ += err_rate

    print('mean error rate of %f times :' % times, err_rate_/times)


def test_get_top_words():
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')

    get_top_words(ny, sf)


# test_local_words()
test_get_top_words()