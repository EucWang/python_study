from kNN_test.kNN import *


def classify_person():
    """通过用户控制台输入信息,对指定的信息判断该信息进行分类"""
    result_list = ['not at all', 'in small doses', 'in large doses']

    percent_tats = float(input('percentage of time spent playing video games?'))
    ff_miles = float(input('frequent flier miles earned per year?'))
    ice_cream = float(input('liters of ice cream consumed per year?'))

    dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')

    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)

    in_arr = np.array([ff_miles, percent_tats, ice_cream])

    classifier_result = classify0(in_arr, norm_mat, dating_labels, 3)

    print('You will probably like this person:', result_list[classifier_result - 1])


#调用函数
classify_person()
