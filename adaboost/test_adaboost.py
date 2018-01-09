
import common.load_data_from_file as load_data
import adaboost.ada_boost as ada
import numpy as np

data_arr, label_arr = load_data.load_data_set('../logistic_regression/horseColicTraining.txt')
classifier_arr = ada.adaboost_train_ds(data_arr, label_arr, 100)

test_arr, test_label_arr = load_data.load_data_set('../logistic_regression/horseColicTest.txt')
prediction = ada.ada_classify(test_arr, classifier_arr)
# print('prediction', prediction)
err_arr = np.mat(np.ones((67, 1)))
sum = err_arr[prediction!=np.mat(test_label_arr).T].sum()
print('sum', sum)