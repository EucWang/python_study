import matplotlib.pyplot as plt

# 节点格式
decision_node = dict(boxstyle='sawtooth', fc='0.8')
# 叶子节点格式
leaf_node = dict(boxstyle='round4', fc='0.8')
# 箭头格式
arrow_args = dict(arrowstyle='<-')


def plot_node(node_txt, center_pt, parent_pt, node_type):
    """
    绘制 图上给定点的注释内容
    :param node_txt:  注解文字
    :param center_pt: 注解文字显示的中心点
    :param parent_pt: 注解的对象的点所在位置, 箭头指向的位置
    :param node_type: 是决策点还是分支点, 取值: decision_node, leaf_node
    :return: 无返回
    """

    # create_plot 因为是全局变量,
    # 所以这里直接使用 create_plot()函数中的这个变量
    create_plot.ax1.annotate(node_txt,
                             xy=parent_pt,
                             xycoords='axes fraction',
                             xytext=center_pt,
                             textcoords='axes fraction',
                             va='center',
                             ha='center',
                             bbox=node_type,
                             arrowprops=arrow_args)


def create_plot():
    """
    创建一个图形
    :return: 无返回
    """
    # 创建一个图形
    fig = plt.figure(1, facecolor='white')
    # 清空绘图区
    fig.clf()

    # create_plot 在这里是一个全局变量, python中变量默认都是全局有效的,
    create_plot.ax1 = plt.subplot(111, frameon=False)

    # 绘制两个代表不同类型的树节点
    plot_node('a decision node', (0.5, 0.1), (0.1, 0.5), decision_node)
    plot_node('a decision node', (0.2, 0.8), (0.2, 0.8), decision_node)
    plot_node('a leaf node', (0.8, 0.1), (0.3, 0.8), leaf_node)

    # 显示
    plt.show()


def get_num_leaf(my_tree):
    """
    获取叶节点的数目
    :param my_tree: 决策树, 字典类型
    :return:
    """
    num_leafs = 0

    # 第一个key
    # first_str = list(my_tree.keys())[0]
    for key in list(my_tree.keys()):
        # 第一个key对应的value, 这个value也是一个字典
        # value = my_tree[first_str]
        value = my_tree[key]

        if type(value).__name__ == 'dict':
            # 遍历下一级字典的key
            for key in list(value.keys()):
                # 如果遍历的value 的类型也是 字典
                if type(value[key]).__name__ == 'dict':
                    # 递归计算其叶子节点数
                    num_leafs += get_num_leaf(value[key])
                else:
                    # 如果遍历的value 的类型不是字典, 则表明这是一个叶子节点, 返回结果自增1
                    num_leafs += 1
        else:
            num_leafs += 1
    return num_leafs


def get_tree_depth(my_tree):
    """
    获取决策树的层数

    :param my_tree:   决策树,字典类型
    :return: 最大数结构层级 ,这个结果比预计的结果>1, 拿到这个结果之后需要 -1
    """
    max_depth = 0
    this_depth = 0
    # key = list(my_tree.keys())[0]
    has_next_depth = False
    for key in list(my_tree.keys()):
        value = my_tree[key]
        if type(value).__name__ == 'dict':
            # for key_next in list(value.keys()):
            #     if type(value[key_next]).__name__ == 'dict':
            #         this_depth = 1 + get_tree_depth(value[key_next])
            if not has_next_depth:
                has_next_depth = True
                this_depth += 1

            if max_depth < get_tree_depth(value):
                max_depth = get_tree_depth(value)

        # else:
        #     this_depth = 1

        # if this_depth > max_depth:
        #     max_depth = this_depth

    # return max_depth
    return this_depth + max_depth


# def test_get_num_leaf():
# a = {'a': 'b', 'b': {'c': 'g', 'd': {'zhangsan': 'zhang', 'lisi': {'lisan': 'san', 'liwu': {'liliu': {'liqi': 'qi', 'liba': 'ba'}}}}}, 'c': {'d': 'fff', 'dd': 'ffff', 'ee': {'f': 'fff', 'g': 'gg'}}}
# a = {'a': 'b', 'b': {'c': 'g', 'd': {'zhangsan': 'zhang', 'lisi': {'lisan': 'san', 'liwu': {'liliu': 'liu'}}}}, 'c': {'d': 'fff', 'dd': 'ffff', 'ee': {'f': 'fff', 'g': 'gg'}}}
# a = {'a': 'b', 'b': {'c': 'g', 'd': {'zhangsan': 'zhang', 'lisi': {'lisan': 'san', 'liwu': 'wu'}}}, 'c': {'d': 'fff', 'dd': 'ffff', 'ee': {'f': 'fff', 'g': 'gg'}}}
# a = {'a': 'b', 'b': {'c': 'g', 'd': 'ddd'}, 'c': {'d': 'fff', 'dd': 'ffff', 'ee': {'f': 'fff', 'g': 'gg'}}}
# num = get_num_leaf(a)
# depth = get_tree_depth(a)
# print("leaf_num ", num)
# print("tree depth", depth)


def retrieve_tree(i):
    """
    生成不同的测试数据

    :param i:  接受0,1,2,3 返回不同的测试数据
    :return:   测试数据
    """
    list_of_trees = [
        {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
        {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}},
        {'b': {'c': '×', 'd': {'lisi': {'lisan': '×', 'liwu': {'liliu': {'liqi': 'x', 'liba': '√', 'zz': 'maybe'}}}}}},
        {'c': {'d': 'fff', 'f': 'ffff', 'e': {'z': {'f': 'fff', 'g': 'ggg'}}}}

    ]
    return list_of_trees[i]


def plot_mid_text(cntr_pt, parent_pt, txt_string):
    """
    在父节点 和 子节点 之间填充文本
    :param cntr_pt:  中心点
    :param parent_pt:  父节点
    :param txt_string: 需要填充的文本
    :return:
    """
    x_mid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0]
    y_mid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1]

    create_plot.ax1.text(x_mid, y_mid, txt_string)


def plot_tree(my_tree, parent_pt, node_txt):
    """
    绘制决策树图形的大部分工作都是在这里完成的

    :param my_tree: 决策树
    :param parent_pt:
    :param node_txt:
    :return: 无返回
    """
    # 获取当前的宽和高
    num_leafs = get_num_leaf(my_tree)
    depth = get_tree_depth(my_tree) - 1
    # 获取节点标签文本
    first_str = list(my_tree.keys())[0]
    # 获取 决策点的中心点, 这个不是叶子节点
    cntr_pt = (plot_tree.xOff + (1.0 + float(num_leafs)) / 2.0 / plot_tree.totalW, plot_tree.yOff)
    # 绘制线段上显示的文本内容
    plot_mid_text(cntr_pt, parent_pt, node_txt)
    # 绘制决策点文本以及带箭头的线段
    plot_node(first_str, cntr_pt, parent_pt, decision_node)

    # 下一级的字典
    second_dict = my_tree[first_str]

    plot_tree.yOff = plot_tree.yOff - 1.0 / plot_tree.totalD

    if type(second_dict).__name__ == 'dict':       # 如果下一级是字典
        for key in list(second_dict.keys()):
            if type(second_dict[key]).__name__ == 'dict':         # 当下一级节点也是一个决策点时
                plot_tree(second_dict[key], cntr_pt, str(key))    # 递归调用绘制下一级的图形
            else:                                                 # 否则, 绘制叶子节点
                plot_tree.xOff = plot_tree.xOff + 1.0 / plot_tree.totalW
                plot_node(second_dict[key], (plot_tree.xOff, plot_tree.yOff), cntr_pt, leaf_node)  # 绘制叶子节点, 没有带箭头的线段
                plot_mid_text((plot_tree.xOff, plot_tree.yOff), cntr_pt, str(key))                 # 绘制

    plot_tree.yOff = plot_tree.yOff + 1.0 / plot_tree.totalD


def create_plot2(in_tree):
    """
    最终版本
    生成决策树的主函数

    :param in_tree: 已经生成的决策树的 字典
    :return: 无返回
    """
    # 创建图像
    fig = plt.figure(1, facecolor='white')
    # 清空图形
    fig.clf()

    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)

    # 根据totalW, totalD 这两个数据可以确定绘制决策树的水平方向和垂直方向的中心位置
    # 全局变量plot_tree.totalW 存储树的宽度 , 这里就是叶子节点数
    plot_tree.totalW = float(get_num_leaf(in_tree))
    # 全局变量plot_tree.totalD 存储树的深度, 这里就是决策树的层级
    plot_tree.totalD = float(get_tree_depth(in_tree) - 1)

    # xOff 和 yOff 追踪已经绘制的节点位置,以及放置下一个节点的恰当位置.
    # 绘制图形的x轴的有效范围是0.0 ~ 1.0
    # y轴的有效范围 也是 0.0 ~ 1.0
    plot_tree.xOff = - 0.5 / plot_tree.totalW
    plot_tree.yOff = 1.0

    plot_tree(in_tree, (0.5, 1.0), '')

    plt.show()


create_plot2(retrieve_tree(1))

# print('depth ', get_tree_depth(retrieve_tree(0)))
# print('leafs num ', get_num_leaf(retrieve_tree(0)))
# test_get_num_leaf()
# create_plot()
