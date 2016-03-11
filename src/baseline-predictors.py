import numpy as np
import pandas as pd


class BaselinePredicor(object):

    def __init__(self, ratings):
        self.ratings = ratings
        self.ratings_index = ratings.index
        self.ratings_col = ratings.columns

    def calculate_user_means(self):
        """
        Calculate the user means of the ratings matrix
        :return: A DataFrame indexed on user_id with one column mean containinng the user means
        """

        ratings = self.ratings.replace(0, np.nan)
        means = pd.DataFrame(ratings.mean(axis=1), index=self.ratings_index, columns=['mean']).fillna(value=0)
        return means

