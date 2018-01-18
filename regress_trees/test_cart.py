from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from regress_trees.cart import *
import matplotlib.pyplot as plt
import tkinter as tkinter

def test_create_tree1():
    data_arr = load_data_set('ex00.txt')
    trees = create_tree(np.mat(data_arr))
    print(trees, type(trees['split_value']))

    figure = plt.figure()
    axis = figure.add_subplot(111)

    x = [a for a in (np.mat(data_arr)[:, 0]).T.flatten()[0].tolist()[0]]
    y = [b for b in ((np.mat(data_arr)[:, 1]).T.flatten()[0]).tolist()[0]]

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for index in range(len(x)):
        if x[index] < float(trees['split_value']):
            x1.append(x[index])
            y1.append(y[index])
        else:
            x2.append(x[index])
            y2.append(y[index])

    axis.scatter(x1, y1, s=10, c='blue')
    axis.scatter(x2, y2, s=10, c='green')
    x = trees['split_value']
    axis.plot([x, x], [-0.5, 1.5], c='red')
    figure.show()


def get_split_values(trees, split_values):
    """
    将树中的所有这个元素都放入到数组split_values中

    解析数中的 元素 :split_value,
    :param trees:
    :param split_values:
    :return:
    """
    split_values.append(float(trees['split_value']))

    left_val = trees['left']
    if is_tree(left_val):
        get_split_values(left_val, split_values)

    right_val = trees['right']
    if is_tree(right_val):
        get_split_values(right_val, split_values)


def test_create_tree2(filename='ex0.txt', leaf_type=reg_leaf, err_type=reg_err,  x_index=1, ops=[1, 4]):
    """
    :param err_type:
    :param leaf_type:
    :param ops: 误差的数量级
    :param filename: 加载的文件
    :param x_index:  X轴取的数据 是矩阵的那一列
    :return:
    """
    data_arr = load_data_set(filename)
    trees = create_tree(data_set=np.mat(data_arr), leaf_type=leaf_type, err_type=err_type, ops=ops)
    print(trees)

    figure = plt.figure()
    axis = figure.add_subplot(111)
    x = [a for a in (np.mat(data_arr)[:, x_index]).T.flatten()[0].tolist()[0]]
    y = [b for b in ((np.mat(data_arr)[:, -1]).T.flatten()[0]).tolist()[0]]
    x_copy = x.copy()
    y_copy = y.copy()

    split_values = list()
    get_split_values(trees, split_values)

    split_values.sort()
    xi = []
    yi = []

    for i in range(len(split_values)):
        x_item = []
        y_item = []
        for index in range(len(x_copy)):
            if x_copy[index] < split_values[i]:
                x_item.append(x_copy[index])
                y_item.append(y_copy[index])
                x_copy[index] = np.inf
                y_copy[index] = np.inf

        xi.append(x_item)
        yi.append(y_item)
    xi.append([item for item in x_copy if item != np.inf])
    yi.append([item for item in y_copy if item != np.inf])

    color = ['red', 'blue', 'green', 'yellow']

    # axis.scatter(x, y, s=5, c='blue')
    for i in range(len(xi)):
        axis.scatter(xi[i], yi[i], s=5, c=color[i%len(color)])

    for val in split_values:
        axis.plot([val, val], [min(y), max(y)], c='red')
    figure.show()


def test_prune():
    my_data2 = load_data_set('ex2.txt')
    my_mat2 = np.mat(my_data2)
    my_tree = create_tree(my_mat2, ops=[0, 1])
    print('my_tree', my_tree)

    my_data_set = load_data_set('ex2test.txt')
    my_data_mat = np.mat(my_data_set)
    test_tree = prune(my_tree, my_data_mat)
    print('test_tree', test_tree)


def test_model_tree():
    data_arr = load_data_set('exp2.txt')
    my_mat2 = np.mat(data_arr)
    tree = create_tree(my_mat2, model_leaf, model_err, [1, 10])
    print('tree', tree)

    figure = plt.figure()
    axis = figure.add_subplot(111)
    x = [a for a in (np.mat(data_arr)[:, 0]).T.flatten()[0].tolist()[0]]
    y = [b for b in ((np.mat(data_arr)[:, -1]).T.flatten()[0]).tolist()[0]]

    x1 = [];
    y1 = []
    x2 = [];
    y2 = []
    for index in range(len(x)):
        if x[index] < float(tree['split_value']):
            x1.append(x[index])
            y1.append(y[index])
        else:
            x2.append(x[index])
            y2.append(y[index])
    axis.scatter(x1, y1, s=2, c='blue')
    axis.scatter(x2, y2, s=2, c='green')

    split_value = tree['split_value']
    x0 = [split_value, 1.0]
    x1 = [0.0, split_value]

    # 使用模型树时, 如果叶子节点不能再划分的话,那么
    # 叶子节点中存储的就是 线性模型系数
    # 这里数据集只有1个参数,那么可以通过返回的结果计算模型线段
    # y = a0 + a1 * x
    y0 = [float(tree['left'][0][0]) + float(tree['left'][1][0]) * x0[0],
          float(tree['left'][0][0]) + float(tree['left'][1][0]) * x0[1]]

    y1 = [float(tree['right'][0][0]) + float(tree['right'][1][0]) * x1[0],
          float(tree['right'][0][0]) + float(tree['right'][1][0]) * x1[1]]
    axis.plot(x0, y0, c='red')
    axis.plot(x1, y1, c='orange')

    figure.show()


def split_data2_x_y(tree, mat, feature_index=0):
    x = [a for a in (mat[:, feature_index]).T.flatten()[0].tolist()[0]]
    y = [b for b in ((mat[:, -1]).T.flatten()[0]).tolist()[0]]
    x_copy = x.copy()
    y_copy = y.copy()

    split_values = list()
    get_split_values(tree, split_values)

    split_values.sort()
    xi = []
    yi = []

    for i in range(len(split_values)):
        x_item = []
        y_item = []
        for index in range(len(x_copy)):
            if x_copy[index] < split_values[i]:
                x_item.append(x_copy[index])
                y_item.append(y_copy[index])
                x_copy[index] = np.inf
                y_copy[index] = np.inf

        xi.append(x_item)
        yi.append(y_item)
    xi.append([item for item in x_copy if item != np.inf])
    yi.append([item for item in y_copy if item != np.inf])
    return xi, yi


def test_compare_trees():
    train_mat = np.mat(load_data_set('bikeSpeedVsIq_train.txt'))
    test_mat = np.mat(load_data_set('bikeSpeedVsIq_test.txt'))
    train_tree = create_tree(train_mat, ops=[1, 20])
    print('train_tree', train_tree)
    y_hat = create_fore_cast(train_tree, test_mat[:, 0])
    print('y_hat' , y_hat)
    r = np.corrcoef(y_hat, test_mat[:, 1], rowvar=0)[0, 1]
    print('r', r)

    train_tree2 = create_tree(train_mat, model_leaf, model_err, [1, 20])
    y_hat2 = create_fore_cast(train_tree2, test_mat[:, 0], model_tree_eval)
    r2 = np.corrcoef(y_hat2, test_mat[:, 1], rowvar=0)[0, 1]
    print('r2', r2)

    # 对训练数据集和测试数据集图形化显示
    plt.figure(22)
    color = ['red', 'blue', 'green', 'yellow']

    figure = plt.subplot(221)
    xi, yi = split_data2_x_y(train_tree, train_mat)
    for i in range(len(xi)):
        figure.scatter(xi[i], yi[i], s=5, c=color[i%len(color)])
    plt.title('Train Data', fontsize=14)

    figure = plt.subplot(222)
    xi, yi = split_data2_x_y(train_tree, test_mat)
    for i in range(len(xi)):
        figure.scatter(xi[i], yi[i], s=5, c=color[i%len(color)])
    plt.title('Test Data', fontsize=14)

    figure = plt.subplot(223)
    xi, yi = split_data2_x_y(train_tree2, train_mat)
    for i in range(len(xi)):
        figure.scatter(xi[i], yi[i], s=5, c=color[i%len(color)])
    plt.title('Train Data2', fontsize=14)

    figure = plt.subplot(224)
    xi, yi = split_data2_x_y(train_tree2, test_mat)
    for i in range(len(xi)):
        figure.scatter(xi[i], yi[i], s=5, c=color[i%len(color)])
    plt.title('Test Data2', fontsize=14)

    plt.show()


def test_tkinter1():
    ttk = tkinter.Tk()
    # frame = tkinter.Frame(ttk, borderwidth=20)
    my_label = tkinter.Label(ttk, text='Hello World')
    my_label.grid()
    # frame.pack()
    tkinter.mainloop()

# test_create_tree1()
# test_create_tree2()
# test_create_tree2('ex2.txt', 0)
# test_create_tree2('ex2.txt', 0, [1, 80])
# test_prune()
# test_model_tree()
# test_compare_trees()
