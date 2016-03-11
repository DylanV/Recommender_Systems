import numpy as np
import pandas as pd


class BaselinePredicor(object):

    def __init__(self, ratings):
        self.ratings = ratings
        self.ratings_index = ratings.index
        self.ratings_col = ratings.columns

    @staticmethod
    def calculate_user_means(ratings):
        """
        Calculate the user means of the ratings matrix
        :param ratings: The ratings DataFrame
        :type ratings: DataFrame
        :return: A DataFrame indexed on user_id with one column mean with the user means
        """

        ratings = ratings.replace(0, np.nan)
        means = pd.DataFrame(ratings.mean(axis=1), index=ratings.index, columns=['mean']).fillna(value=0)
        return means

    @staticmethod
    def calculate_user_std_devs(ratings):
        """
        Calculate the user standatd deviations in the user x item DataFrame
        :param ratings: the ratings DataFrame
        :type ratings: DataFrame
        :return: A DataFrame indexed on user_id with one column std with the user standard deviations
        """

        ratings = ratings.replace(0, np.nan)
        std_devs = pd.DataFrame(ratings.std(axis=1), index=ratings.index, columns=['std'])
        std_devs = std_devs.fillna(value=0)
        return std_devs
