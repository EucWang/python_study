import unittest
import numpy as np
from distance_type_calculate.distance_calculate import DistanceCalculate

class TestDistanceCalculate(unittest.TestCase):

    def setUp(self):
        # self.calculate = DistanceCalculate()
        self.vector1 = np.mat([1, 2, 3])
        self.vector2 = np.mat([4, 5, 6])

    def test_euclidean_distance(self):
        # result = self.calculate.euclidean_distance(self.vector2, self.vector1)
        result = DistanceCalculate.euclidean_distance(self.vector2, self.vector1)
        print("euclidean_distance", result)
        self.assertEqual(np.sqrt(27) , result)

    def test_manhattan_distance(self):
        distance = DistanceCalculate.manhattan_distance(self.vector1, self.vector2)
        print("manhattan_distance", distance)
        self.assertEqual(9, distance)

    def test_chebyshev_distance(self):
        distance = DistanceCalculate.chebyshev_distance(self.vector1, self.vector2)
        print("chebyshev_distance", distance)
        self.assertEqual(3, distance)

    def test_cosine_distance(self):
        distance = DistanceCalculate.cosine_distance(self.vector1, self.vector2)
        print("cosine distance", distance)
        sqrt_ = (1 * 4 + 2 * 5 + 3 * 6) / (np.sqrt(14) * np.sqrt(16 + 25 + 36))
        self.assertEqual(distance, sqrt_)

    def test_hamming_distance(self):
        distance = DistanceCalculate.hamming_distance(self.vector1, self.vector2)
        print("hamming_distance", distance)
        self.assertEqual(distance, 3)

        distance = DistanceCalculate.hamming_distance(np.mat([1, 2, 3]),  np.mat([2, 4, 3]))
        print("hamming_distance", distance)
        self.assertEqual(distance, 2)

        distance = DistanceCalculate.hamming_distance(np.mat([1, 2, 3]),  np.mat([2, 2, 3]))
        print("hamming_distance", distance)
        self.assertEqual(distance, 1)

    def test_jaccard_distance(self):
        # distance = DistanceCalculate.jaccard_distance(self.vector1, self.vector2)
        distance = DistanceCalculate.jaccard_distance(self.vector1, np.mat([1, 2, 4]))
        print("jaccard_distance", distance)

unittest.main()