import unittest
import pandas as pd
import numpy as np
from collaborative_filtering import CollaborativeFiltering


class TestCollaborativeFiltering(unittest.TestCase):

    def test_init(self):
        ratings_values = np.asarray([[2, 0, 4, 3, 0],
                                    [0, 3, 0, 3, 0],
                                    [4, 0, 0, 0, 5],
                                    [0, 3, 1, 0, 0],
                                    [5, 0, 4, 3, 1]])
        ratings = pd.DataFrame(ratings_values)
        cf = CollaborativeFiltering(ratings)
        self.assertEqual(cf.ratings.shape, (5,5))

        true_user_means = np.asarray([3., 3., 4.5, 2., 3.25])
        self.assertTrue(np.all((true_user_means - cf.user_means.values[:, 0]) == 0))

        true_user_std_devs = [np.sqrt(2./3.), 0.,0.5, 1.0, np.sqrt(2.1875)]
        self.assertTrue(np.all((true_user_std_devs - cf.user_std_devs.values[:, 0]) == 0))

    def test_calculate_user_means(self):
        ratings_values = np.asarray([[2, 0, 4, 3, 0],
                                    [0, 3, 0, 3, 0],
                                    [4, 0, 0, 0, 5],
                                    [0, 3, 1, 0, 0],
                                    [5, 0, 4, 3, 1]])
        ratings = pd.DataFrame(ratings_values)
        cf = CollaborativeFiltering(ratings)
        true_user_means = np.asarray([3., 3., 4.5, 2., 3.25])
        means = cf.calculate_user_means(ratings)
        self.assertTrue(np.all((true_user_means - means.values[:, 0]) == 0))
        self.assertEqual(means.shape, (5, 1))

    def test_calculate_user_std_devs(self):
        ratings_values = np.asarray([[2, 0, 4, 3, 0],
                                    [0, 3, 0, 3, 0],
                                    [4, 0, 0, 0, 5],
                                    [0, 3, 1, 0, 0],
                                    [5, 0, 4, 3, 1]])
        ratings = pd.DataFrame(ratings_values)
        cf = CollaborativeFiltering(ratings)

        true_user_std_devs = [np.sqrt(2./3.), 0.,0.5, 1.0, np.sqrt(2.1875)]
        user_std_devs = cf.calculate_user_std_devs(ratings)
        self.assertTrue(np.all((true_user_std_devs - user_std_devs.values[:, 0]) == 0))
        self.assertEqual(user_std_devs.shape, (5, 1))

    def test_get_user_similarity_cosine(self):
        ratings_values = np.asarray([[1, 3, 0],
                                     [2, 0, 4],
                                     [0, 3, 5]])
        ratings = pd.DataFrame(ratings_values)
        user_similarity = CollaborativeFiltering(ratings).get_user_similarity(ratings)

        u1 = np.sqrt(10.)
        u2 = np.sqrt(20.)
        u3 = np.sqrt(34.)
        true_simil = np.asarray([[1, 2./(u1*u2), 9./(u1*u3)],
                                 [2./(u1*u2), 1, 20./(u2*u3)],
                                 [9./(u1*u3), 20./(u2*u3), 1]])

        self.assertTrue(np.all((true_simil - user_similarity.values[:, :]).round(decimals=12) == 0))

    def test_adjust_user_similarity_knn(self):
        ratings_values = np.asarray([[1, 3, 0],
                                     [2, 0, 4],
                                     [0, 3, 5]])
        ratings = pd.DataFrame(ratings_values)
        cf = CollaborativeFiltering(ratings)
        user_similarity = cf.get_user_similarity(ratings)
        user_similarity = cf.adjust_user_similarity_knn(user_similarity, 1)
        u1 = np.sqrt(10.)
        u2 = np.sqrt(20.)
        u3 = np.sqrt(34.)
        true_simil = np.asarray([[1, 0., 9./(u1*u3)],
                                 [0., 1, 20./(u2*u3)],
                                 [0., 20./(u2*u3), 1]])

        self.assertTrue(np.all((true_simil - user_similarity.values[:, :]).round(decimals=12) == 0))

    def test_adjust_ratings_with_means(self):
        ratings_values = np.asarray([[1, 3, 0],
                                     [2, 0, 4],
                                     [0, 3, 5]])
        ratings = pd.DataFrame(ratings_values)
        cf = CollaborativeFiltering(ratings)
        adjusted_ratings = cf.adjust_ratings()
        true_adjusted_ratings_values = np.asarray([[-1., 1., 0.],
                                                    [-1., 0., 1.],
                                                    [0., -1., 1.]])

        self.assertTrue(np.all((true_adjusted_ratings_values - adjusted_ratings.values[:, :]) == 0))

    def test_adjust_ratings_full(self):
        ratings_values = np.asarray([[1, 5, 0],
                                     [5, 0, 1],
                                     [0, 1, 5]])
        ratings = pd.DataFrame(ratings_values)
        cf = CollaborativeFiltering(ratings)
        adjusted_ratings = cf.adjust_ratings(type='full')
        true_adjusted_ratings_values = np.asarray([[-1., 1., 0.],
                                                    [1., 0., -1.],
                                                    [0., -1., 1.]])
        self.assertTrue(np.all((true_adjusted_ratings_values - adjusted_ratings.values[:, :]) == 0))

    def test_adjust_predictions_mean(self):
        ratings_values = np.asarray([[1, 3, 0],
                                     [2, 0, 4],
                                     [0, 3, 5]])
        cf = CollaborativeFiltering(pd.DataFrame(ratings_values))

        prediction_values = np.asarray([[-1., 1., 0.],
                                        [-1., 0., 1.],
                                        [0., -1., 1.]])
        predictions = cf.adjust_predictions(predicted = pd.DataFrame(prediction_values), type='means')
        true_predictions =  np.asarray([[1., 3., 1.],
                                     [2., 1., 4.],
                                     [1., 3., 5.]])
        print predictions
        self.assertTrue(np.all((true_predictions - predictions.values[:, :]) == 0))

    def test_adjust_predictions_full(self):
        ratings_values = np.asarray([[1, 5, 0],
                                     [5, 0, 1],
                                     [0, 1, 5]])
        cf = CollaborativeFiltering(pd.DataFrame(ratings_values))

        prediction_values = np.asarray([[-1., 1., 0.],
                                        [1., 0., -1.],
                                        [0., -1., 1.]])
        predictions = cf.adjust_predictions(predicted = pd.DataFrame(prediction_values), type='full')
        true_predictions =  np.asarray([[1, 5, 1],
                                     [5, 1, 1],
                                     [1, 1, 5]])

        self.assertTrue(np.all((true_predictions - predictions.values[:, :]) == 0))