import unittest
import pandas as pd
import numpy as np
from baseline_predictors import BaselinePredictor

class TestBaselinePredictors(unittest.TestCase):

    def setUp(self):
        self.ratings_values = np.asarray([[2, 0, 4, 3, 0],
                                        [0, 3, 0, 3, 0],
                                        [4, 0, 0, 0, 5],
                                        [0, 3, 1, 0, 0],
                                        [5, 0, 4, 3, 1]])
        self.ratings = pd.DataFrame(self.ratings_values)

    def test_init(self):
        bp = BaselinePredictor(self.ratings)
        self.assertEqual(bp.ratings.shape, (5,5))

    def test_calculate_user_means(self):
        bp = BaselinePredictor(self.ratings)
        true_user_means = np.asarray([3., 3., 4.5, 2., 3.25])
        means = bp.calculate_user_means(self.ratings)
        self.assertTrue(np.all((true_user_means - means.values[:, 0]) == 0))
        self.assertEqual(means.shape, (5, 1))

