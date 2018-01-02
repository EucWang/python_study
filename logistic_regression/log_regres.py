import numpy as np
import matplotlib.pyplot as plt


def load_data_set():
    """
    返回测试数据集
    :return:
        data_mat 数据集合, 普通的list集合
        label_mat 对应数据集合的标签集合, list集合
    """
    data_mat_ = []
    label_mat_ = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        line_arr = line.strip().split()
        # x0 为1.0
        # x1 为文件中的第一列
        # x2 为文件中的第二列
        data_mat_.append([1.0, float(line_arr[0]), float(line_arr[1])])
        # x3 为文件中的第三列, 标签向量
        label_mat_.append(int(line_arr[2]))

    return data_mat_, label_mat_


def sigmoid(in_x):
    """
    Sigmoid算法, 将一个大数转换到 0~1 之间, 通过判断其是否大于0.5 确定其值要么为0, 要么为1
    0.5 作为分界点, 在作图时可以作为分界线的取值,确定两个特征之间的对应关系,如果可以的话
    :param in_x:
    :return:
    """
    return 1.0/(1 + np.exp(-in_x))


def classify_vector(in_x, weights):
    """
    Logistic回归分类函数
    :param in_x:    特征向量
    :param weights: 最优化系数
    :return:  返回分类类型, 0.0/1.0
    """
    prob = sigmoid(sum(in_x * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def grad_ascent(data_mat_in, class_labels):
    """
    梯度上升算法 ,
    这里将第0维特征的值设置为1.0

    Logistic回归梯度上升优化算法
    :param data_mat_in: 一个2维Numpy数组, 每列代表不同的特征, 每行代表每个训练样本
    :param class_labels: 数据对应的类别标签
    :return:
    """

    # 将数据集转换成matrix
    data_matrix = np.mat(data_mat_in)
    # 转置,  转换成列向量
    label_mat_ = np.mat(class_labels).transpose()
    # 获取矩阵的 维数, m行, n列
    m, n = np.shape(data_matrix)
    # 向目标移动的步长
    alpha = 0.001
    # 迭代的次数
    max_cycles = 500
    # 回归系数初始化为1, n行1列, 列向量
    weights = np.ones((n, 1))
    # 迭代500次
    for k in range(max_cycles):
        # 首先让data_matrix 与weights做矩阵相乘,
        # 数据集中的每个特征元素都和对应的系数相乘,然后每一行的数据想加,获得一个列向量
        # 结果就是一个 m行1列的数据
        data_matrix_weights = data_matrix * weights
        # 通过sigmoid算法计算得到一个初始的值, 这里的h依然是一个列向量
        # 每一行数据对应的sigmoid函数值,取值 0 ~ 1 之间, 0.5作为分界线
        # h就是一个m行1列的数据
        h = sigmoid(data_matrix_weights)
        # 和标签结果比较,获取误差值,
        # 这里error也成了一个列向量了, 长度和数据量等同
        # error就是一个m行1列的数据
        error = (label_mat_ - h)

        # data_matrix.transpose() 变成一个 n行m列的矩阵, 每一行表示一个特征的所有取值
        # error 是一个 m行1列的矩阵
        # 每一行数据的 特征值 和error中当前行对应的error相乘, 然后当前行所有乘积相加, 结果然后和步长alpha相乘,结果是一个 n行1列的矩阵
        # 然后 + weights, 依然是列向量, 元素和元素想加, 结果依然是列向量, weighs就是一个n行1列的矩阵
        # 则完成一次迭代,获取到回归系数的迭代值
        weights = weights + alpha * data_matrix.transpose() * error

    return weights


def stoc_grad_ascent0(data_matrix, class_labels, num_iter=150):
    """
    随机梯度上升算法
    一次仅用一个样本点来更新回归系数
    由于可以在新样本到来时对分类器进行增量更新,因而随机梯度上升算法是一个在线学习算法.
    :param data_matrix: 数据集合, list
    :param class_labels:  数据分类标签, list
    :param num_iter: 迭代次数, 可选
    :return:
    """
    # 获得数据的维度: m行n列
    m, n = np.shape(data_matrix)
    # 步长,默认0.01
    alpha = 0.01
    # 回归系数,初始化为n行1列的全1 矩阵
    weights = np.ones(n)
    # 改进1, 增加总的迭代次数
    # 在总的迭代次数的基础上,迭代150次
    for j in range(num_iter):
        # 在总的迭代开始时,给定一个0~m的一个ndarray数组
        data_index = list(range(m))
        # 遍历m行数据
        for i in range(m):
            # 改进2
            # 每次迭代时,都需要调整alpha, 缓解获取到的优化系数的波动
            # 这里有常数项0.01, 保证在多次迭代之后新数据任然具有一定的影响,
            # 如果要处理的问题是动态变化的, 可以适当修改此常数项
            # 在这里, alpha每次减少 1/(j+i),j是迭代次数,i是样本索引,当max(i)>>j时,alpha不是严格下降的
            alpha = 4/(1.0 + j + i) + 0.01
            # 改进3 通过随机选取样本来更新回归系数
            # 这种方法用于减少周期性波动
            # 从data_index 中随机取出一个值, 一个随机索引值,对应一条数据
            rand_index = int(np.random.uniform(0, len(data_index)))

            # 第rand_index行数据,每一项与weights相乘,然后结果相加, 计算sigmoid函数值,得到结果h
            h = sigmoid(sum(data_matrix[rand_index] * weights))
            # 计算第rand_index行数据的参数误差
            error = class_labels[rand_index] - h
            # 第rand_index行数据, 获取误差更新之后的优化系数
            weights = weights + alpha * error * data_matrix[rand_index]
            # 从索引集合中删除当前已经遍历过的索引值
            del(data_index[rand_index])
            # 然后根据这个优化系数, 遍历下一行数据

    return weights


def plot_best_fit(wei):
    """
    画出数据集和Logistic回归最佳拟合直线的函数
    :param wei: 测试数据的最佳拟合系数, matrix数据类型 或者 ndarray数据类型
    :return:
    """
    # 通过matrix.getA()函数获得numpy.ndarray数量类型
    if type(wei) == np.matrixlib.defmatrix.matrix:
        print(plot_best_fit.__name__, "接受参数类型为matrix")
        weights = wei.getA()
        print(plot_best_fit.__name__, "接受参数类型为ndarray")
    elif type(wei) == np.ndarray:
        weights = wei
    else:
        print(plot_best_fit.__name__, "接受参数类型不是指定类型")
        raise BaseException('接受的参数类型错误')

    # 获得测试数据集和测试数据分类标签
    data_mat_, label_mat_ = load_data_set()
    # 通过np.array()函数将list转换成numpy.ndarray数据类型
    data_arr = np.array(data_mat_)
    # 获取到测试数据有多少特征, 也就是有多少列
    n = np.shape(data_arr)[0]
    # 标签为1 的数据点集
    # 这个数据集总共只有2个特征, x表示第一个特征, y表示第二个特征, 第0个特征全是1
    x_cord1 = []
    y_cord1 = []
    # 标签为2 的数据点集
    x_cord2 = []
    y_cord2 = []

    # 编辑数据集, 对其分类
    for i in range(n):
        if int(label_mat_[i]) == 1:
            x_cord1.append(data_arr[i, 1])
            y_cord1.append(data_arr[i, 2])
        else:
            x_cord2.append(data_arr[i, 1])
            y_cord2.append(data_arr[i, 2])

    # 开始画图
    fig = plt.figure()
    axis = fig.add_subplot(111)
    #画出点
    # 红色标识为 分类标签为1
    axis.scatter(x_cord1, y_cord1, s=30, c='red', marker='s')
    # 绿色标识为分类标签为0
    axis.scatter(x_cord2, y_cord2, s=30, c='green')

    # x是从-3.0 ~ 3.0,步长为0.1的所有取值
    x = np.arange(-3.0, 3.0, 0.1)
    # (w0 * 1 + w1 * x + w2 * y) = 0 当这个成立时, 那么signoid函数计算得到的值为0.5, 表示分割线
    # 通过如上公式, 这里已经知道了 最优化系数(weights),那么可以建立y和x特征之间的关系
    # 就是 : y = (-w0 - w1 * x)/ w2,
    # 通过公式获取到对应x 的y的所有取值
    y = (-weights[0] - weights[1] * x) / weights[2]

    # 画出折线
    axis.plot(x, y)

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


data_mat, label_mat = load_data_set()
print("计算之后,调整了的回归系数")
# ascent_weights = grad_ascent(data_mat, label_mat)
stoc_grad_ascent_ = stoc_grad_ascent0(np.array(data_mat), label_mat)
print("stoc_grad_ascent_ ", stoc_grad_ascent_)
plot_best_fit(stoc_grad_ascent_)
