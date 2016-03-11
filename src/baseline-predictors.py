import numpy as np
import pandas as pd

from aux import clamp


class BaselinePredictor(object):

    def __init__(self, ratings):
        """
        :param ratings: The ratings DataFrame
        :type ratings: DataFrame
        """
        self.ratings = ratings

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
        Calculate the user standard deviations in the user x item DataFrame
        :param ratings: the ratings DataFrame
        :type ratings: DataFrame
        :return: A DataFrame indexed on user_id with one column std with the user standard deviations
        """
        ratings = ratings.replace(0, np.nan)
        std_devs = pd.DataFrame(ratings.std(axis=1), index=ratings.index, columns=['std']).fillna(value=0)
        return std_devs

    @staticmethod
    def calculate_item_means(ratings):
        """
        Calculate the item means
        :param ratings: ratings DataFrame
        :type ratings: DataFrame
        :return: The item means
        """
        ratings = ratings.replace(0, np.nan)
        means = pd.DataFrame(ratings.mean(axis=0), index=ratings.columns, columns=['mean']).fillna(value=0)
        return means

    def predict_user_based(self):
        """
        Calculate a baseline prediction based on user means
        :return: A prediction DataFrame same shape as ratings
        """
        user_means = self.calculate_user_means(self.ratings)
        predicted = pd.DataFrame(np.ones(self.ratings.shape) * user_means.values,
                                 index=self.ratings.index, columns=self.ratings.columns)
        predicted = predicted.applymap(clamp)
        return predicted

    def predict_item_based(self):
        """
        Calculate a baseline prediction based on item means
        :return: A DataFrame in the same shape as ratings with predictions as values
        """
        movie_means = self.calculate_item_means(self.ratings)
        predicted = pd.DataFrame((np.ones(self.ratings.shape).transpose() * movie_means.values).transpose(),
                                 index=self.ratings.index, columns=self.ratings.columns)
        predicted = predicted.applymap(clamp)
        return predicted

    def predict_item_user_based(self):
        """
        Calculate a baseline prediction based on item means and user average offsets
        :return: The prediction DataFrame
        """
        movie_means = self.calculate_item_means(self.ratings)
        user_std_devs = self.calculate_user_std_devs(self.ratings)
        predicted_values = (np.ones(self.ratings.shape).transpose() * movie_means.values).transpose()
        predicted_values = predicted_values + user_std_devs.values
        predicted = pd.DataFrame(predicted_values, index=self.ratings.index, columns=self.ratings.columns)
        predicted = predicted.applymap(clamp)
        return predicted
