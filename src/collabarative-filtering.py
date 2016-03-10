import os

import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error


class CollaborativeFiltering(object):

    def __init__(self, ratings):
        """
        :param ratings:  The user x item ratings DataFrame. Ratings on 1 to 5 scale. 0 represents unknown.
        :type ratings: DataFrame
        """
        self.ratings = ratings
