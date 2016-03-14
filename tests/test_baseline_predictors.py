import unittest
import pandas as pd
import numpy as np
from baseline_predictors import BaselinePredictor

class TestBaselinePredictors(unittest.TestCase):

    def test_init(self):
        ratings_values = np.asarray([[2, 0, 4, 3, 0],
                                        [0, 3, 0, 3, 0],
                                        [4, 0, 0, 0, 5],
                                        [0, 3, 1, 0, 0],
                                        [5, 0, 4, 3, 1]])
        ratings = pd.DataFrame(ratings_values)
        bp = BaselinePredictor(ratings)
        self.assertEqual(bp.ratings.shape, (5,5))

    def test_calculate_user_means(self):
        ratings_values = np.asarray([[2, 0, 4, 3, 0],
                                        [0, 3, 0, 3, 0],
                                        [4, 0, 0, 0, 5],
                                        [0, 3, 1, 0, 0],
                                        [5, 0, 4, 3, 1]])
        ratings = pd.DataFrame(ratings_values)
        bp = BaselinePredictor(ratings)
        true_user_means = np.asarray([3., 3., 4.5, 2., 3.25])
        means = bp.calculate_user_means(ratings)
        self.assertTrue(np.all((true_user_means - means.values[:, 0]) == 0))
        self.assertEqual(means.shape, (5, 1))

    def test_calculate_user_std_devs(self):
        ratings_values = np.asarray([[2, 0, 4, 3, 0],
                                        [0, 3, 0, 3, 0],
                                        [4, 0, 0, 0, 5],
                                        [0, 3, 1, 0, 0],
                                        [5, 0, 4, 3, 1]])
        ratings = pd.DataFrame(ratings_values)
        bp = BaselinePredictor(ratings)

        true_user_std_devs = [np.sqrt(2./3.), 0.,0.5, 1.0, np.sqrt(2.1875)]
        user_std_devs = bp.calculate_user_std_devs(ratings)
        self.assertTrue(np.all((true_user_std_devs - user_std_devs.values[:, 0]) == 0))
        self.assertEqual(user_std_devs.shape, (5, 1))

    def test_calculate_item_means(self):
        ratings_values = np.asarray([[2, 0, 4, 3, 0],
                                        [0, 3, 0, 3, 0],
                                        [4, 0, 0, 0, 5],
                                        [0, 3, 1, 0, 0],
                                        [5, 0, 4, 3, 1]])
        ratings = pd.DataFrame(ratings_values)
        true_item_means = np.asarray([11./3., 3., 3., 3., 3.])
        means = BaselinePredictor(ratings).calculate_item_means(ratings)
        self.assertTrue(np.all((true_item_means - means.values[:, 0]) == 0))
        self.assertEqual(means.shape, (5, 1))

    def test_predict_user_based(self):
        ratings_values = np.asarray([[1, 3, 0],
                                     [2, 0, 4],
                                     [0, 3, 5]])
        ratings = pd.DataFrame(ratings_values)
        predictions = BaselinePredictor(ratings).predict_user_based()
        true_predictions = np.asarray([[2., 2., 2.],
                                     [3., 3., 3.],
                                     [4., 4., 4.]])
        self.assertTrue(np.all((true_predictions - predictions.values[:, :]) == 0))

    def test_predict_item_based(self):
        ratings_values = np.asarray([[2, 3, 0],
                                     [2, 0, 4],
                                     [0, 3, 4]])
        ratings = pd.DataFrame(ratings_values)
        predictions = BaselinePredictor(ratings).predict_item_based()
        true_predictions = np.asarray([[2., 3., 4.],
                                     [2., 3., 4.],
                                     [2., 3., 4.]])
        self.assertTrue(np.all((true_predictions - predictions.values[:, :]) == 0))

    def test_predict_item_user_based(self):
        ratings_values = np.asarray([[1, 3, 0],
                                     [2, 0, 4],
                                     [0, 3, 5]])
        ratings = pd.DataFrame(ratings_values)
        predictions = BaselinePredictor(ratings).predict_item_user_based()
        true_predictions = np.asarray([[2.5, 4., 5.],
                                     [2.5, 4., 5.],
                                     [2.5, 4., 5.]])
        self.assertTrue(np.all((true_predictions - predictions.values[:, :]) == 0))