import numpy as np
import logistic_regression.log_regres as log_regres


def colic_test():
    fr_train = open('horseColicTraining.txt')
    fr_test = open('horseColicTest.txt')
    training_set = []
    training_labels = []

    # 按行读取训练样本数据集
    for line in fr_train.readlines():
        current_line = line.strip().split('\t')
        line_arr = []
        # 将每一行的前21个数据特征存入到line_arrz中
        for i in range(21):
            line_arr.append(float(current_line[i]))

        training_set.append(line_arr)
        training_labels.append(float(current_line[21]))

    # 调用梯度上升算法 获得最优系数
    training_weights = log_regres.stoc_grad_ascent0(np.array(training_set), training_labels, 500)
    error_count = 0
    num_test_vec = 0.0

    for line in fr_test.readlines():
        num_test_vec += 1.0
        current_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(current_line[i]))

        # 通过sigmoid函数计算 当前测试数据的分类,
        # 判断分类是否和结果相同,不同,则表示分类出错
        if int(log_regres.classify_vector(np.array(line_arr), training_weights)) != int(current_line[21]):
            error_count += 1

    # 计算中的错误率
    error_rate = (float(error_count))/num_test_vec
    print('the error rate of this test is :%f' % error_rate)
    return error_rate


def muli_test():
    num_tests = 10
    error_sum = 0.0
    for k in range(num_tests):
        error_sum += colic_test()
    print('after %d iterations the aveage error rate is %f' % (num_tests, error_sum/float(num_tests)))


muli_test()