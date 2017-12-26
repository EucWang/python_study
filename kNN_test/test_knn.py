from kNN_test.kNN import file2matrix
from kNN_test.kNN import auto_norm
from kNN_test.kNN import classify0


def dating_class_test():
    # 对数据集中的10%的数据作为测试数据
    ho_ratio = 0.10

    # 从文件读取数据集
    dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')

    # 对数据进行归一化处理
    # norm_mat :  归一化之后的数据集
    # ranges   :  归一化时, 数据的比值范围, 分子
    # min_vals :  归一化时, 各个特征的最小值
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)

    # 有多少行数据
    m = norm_mat.shape[0]

    # 获取测试数据的数量
    num_test_vecs = int(m * ho_ratio)

    # 测试数据错误的比率
    error_count = 0.0

    # 对测试数据进行遍历
    for i in range(num_test_vecs):
        #将第i行作为测试数据
        #将数据集第num_test_vecs行之后的数据作为样本数据集
        #取前3个距离最近数据作为参考
        classify_result = classify0(norm_mat[i, :], norm_mat[num_test_vecs:m, :], dating_labels[num_test_vecs:m], 10)
        print("the classifier came back with: %d, the real answer is: %d" % (classify_result, dating_labels[i]))
        if classify_result != dating_labels[i]:
            error_count += 1.0

    print("the total rate is :%f" % (error_count / float(num_test_vecs)))

dating_class_test()