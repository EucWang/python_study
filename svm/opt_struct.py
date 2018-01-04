import numpy as np


class OptStruct:

    def __init__(self, data_mat_in, class_labels, c, toler):
        self.x = data_mat_in
        self.label_mat = class_labels
        self.c = c
        self.tol = toler
        self.m = np.shape(data_mat_in)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.e_cache = np.mat(np.zeros((self.m, 2)))