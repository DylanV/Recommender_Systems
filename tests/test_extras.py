import unittest
import pandas as pd
import numpy as np
import extras

class TestExtrasClamp(unittest.TestCase):

    def test_clamp_default(self):
        self.assertEqual(extras.clamp(4), 4, 'Clamp default changing value in range')

    def test_clamp_less_floor(self):
        self.assertEqual(extras.clamp(-1, floor=1), 1, 'Clamp below floor incorrect')

    def test_clamp_greater_ceiling(self):
        self.assertEqual(extras.clamp(6, ceiling=5), 5, 'Clamp above ceiling incorrect')

    def test_clamp_map(self):
        test_array = np.asarray([-1, 0, 1, 3 ,5, 6])
        map_results = pd.DataFrame(test_array).applymap(extras.clamp).values
        self.assertEqual(map_results[0], 1, 'Clamp in map not handling negative below floor')
        self.assertEqual(map_results[1], 1, 'Clamp in map not handling zero below floor')
        self.assertEqual(map_results[2], 1, 'Clamp in map changing value at floor')
        self.assertEqual(map_results[3], 3, 'Clamp in map changing value in range')
        self.assertEqual(map_results[4], 5, 'Clamp in map changing value at ceiling')
        self.assertEqual(map_results[5], 5, map_results[5])