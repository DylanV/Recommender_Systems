import unittest
import pandas as pd
import numpy as np
import Extras

class TestExtras(unittest.TestCase):

    def test_get_ratings_sparsity(self):
        ratings = pd.DataFrame(np.asarray([[1,0],[1,0]]))
        self.assertEqual(Extras.get_ratings_sparsity(ratings), 50.0, 'Sparsity calculation incorrect')

    def test_get_rmse(self):
        predicted = pd.DataFrame(np.asarray([[2,2],[2,2]]))
        actual = pd.DataFrame(np.asarray([[1,1],[1,1]]))
        self.assertEqual(Extras.get_rmse(predicted, actual), 1, 'get_rmse returning incorrect value')

    def test_split_train_test(self):
        ratings = pd.DataFrame(np.asarray([[3, 0, 3], [0, 5, 2], [1, 0, 4]]))
        train, test = Extras.split_train_test(ratings, test_ratio=0.5)
        self.assertEqual(len(test.values.nonzero()[0]), 3)
        self.assertEqual(len(train.values.nonzero()[0]), 3)
        self.assertTrue(np.all((train.values * test.values) == 0))
        self.assertEqual(train.shape, ratings.shape)
        self.assertEqual(test.shape, ratings.shape)

    def test_clamp_default(self):
        self.assertEqual(Extras.clamp(4), 4, 'Clamp default changing value in range')

    def test_clamp_less_floor(self):
        self.assertEqual(Extras.clamp(-1, floor=1), 1, 'Clamp below floor incorrect')

    def test_clamp_greater_ceiling(self):
        self.assertEqual(Extras.clamp(6, ceiling=5), 5, 'Clamp above ceiling incorrect')

    def test_clamp_map(self):
        test_array = np.asarray([-1, 0, 1, 3 ,5, 6])
        map_results = pd.DataFrame(test_array).applymap(Extras.clamp).values
        self.assertEqual(map_results[0], 1, 'Clamp in map not handling negative below floor')
        self.assertEqual(map_results[1], 1, 'Clamp in map not handling zero below floor')
        self.assertEqual(map_results[2], 1, 'Clamp in map changing value at floor')
        self.assertEqual(map_results[3], 3, 'Clamp in map changing value in range')
        self.assertEqual(map_results[4], 5, 'Clamp in map changing value at ceiling')
        self.assertEqual(map_results[5], 5, 'Clamp not handling value above ceiling')