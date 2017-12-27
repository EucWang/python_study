import decision_tree_test.decision_tree as dtree


def create_data_set():
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']

    return data_set, labels


def test_tree():
    data_set, labels = create_data_set()

    shannon_ent = dtree.calc_shannon_ent(data_set)
    print("data_set",data_set)
    print("shannon_ent", shannon_ent)

    # 增加一个新的分类之后, 熵值增加了
    data_set.append([1, 1, 'maybe'])
    shannon_ent = dtree.calc_shannon_ent(data_set)
    print("data_set",data_set)
    print("shannon_ent", shannon_ent)


def test_split_data():
    data_set, labels = create_data_set()
    print(data_set)
    print(dtree.split_data_set(data_set, 0, 1))
    print(dtree.split_data_set(data_set, 0, 0))


def test_best_feature():
    data_set, labels = create_data_set()
    feature_index = dtree.choose_best_feature_to_split(data_set)
    print(feature_index)


def test_create_desicion_tree():
    data_set, labels = create_data_set()
    my_tree = dtree.create_desicion_tree(data_set, labels)
    print(my_tree)


# test_tree()
# test_split_data()
# test_best_feature()
test_create_desicion_tree()