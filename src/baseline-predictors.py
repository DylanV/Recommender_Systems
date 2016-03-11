import numpy as np
import pandas as pd


class BaselinePredictor(object):

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

    @staticmethod
    def clamp(x, floor=1, ceiling=5):
        """
        Clamps a value between the values floor and ceiling
        :param x: The value to be clamped
        :param floor: The minimum value for x
        :param ceiling: The maximum value for x
        :return: The clamped value of x
        """
        if x > ceiling:
            x = max
        elif x < floor:
            x = floor
        return x

    def predict_user_based(self):
        """
        Calculate a baseline prediction based on user means
        :return: A prediction DataFrame same shape as ratings
        """
        user_means = self.calculate_user_means(self.ratings)
        predicted = pd.DataFrame(np.ones(self.ratings.shape) * user_means.values,
                                 index=self.ratings.index, columns=self.ratings.columns)
        return predicted
