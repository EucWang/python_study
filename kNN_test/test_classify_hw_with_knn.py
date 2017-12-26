import unittest

import kNN_test.classify_handwriting_with_knn as hw_knn


class TestClassifyHandWriting(unittest.TestCase):

    def test_load_data(self):
        trainning_dir = "trainingDigits"
        test_dir = 'testDigits'
        mat, labels, nums = hw_knn.load_digits_file_data(trainning_dir)
        # print(labels)
        print(nums)
        self.assertEqual(1934, nums)

        mat, labels, nums = hw_knn.load_digits_file_data(test_dir)
        # print(labels)
        self.assertEqual(946, nums)


unittest.main()
