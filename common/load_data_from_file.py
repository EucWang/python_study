
def load_data_set(file_name):
    """

    :param file_name:
    :return:
    """
    file = open(file_name)
    if not file:
        return None

    one_line_splits = file.readline().split('\t')  # 先读取一行
    num_feat = len(one_line_splits)                # 获取特征数
    data_mat = []
    label_mat = []

    fr = open(file_name)
    for line in fr.readlines():                   # 遍历每一行
        line_arr = []
        cur_line = line.strip().split('\t')       # 获取每一行所有特征
        for i in range(num_feat - 1):
            line_arr.append(float(cur_line[i]))
        data_mat.append(line_arr)
        label_mat.append(float(cur_line[-1]))      # 设定最后一行就是 分类的类别标签

    return data_mat, label_mat