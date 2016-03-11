import numpy as np
import pandas as pd

from aux import clamp

class MatrixFoctorisation(object):
    def __init__(self, ratings):
        self.user_means = self.calculate_user_means(ratings)
        self.user_std_devs = self.calculate_user_std_devs(ratings)
        self.ratings = self.adjust_ratings(ratings)
        self.U, self.eps, self.T = np.linalg.svd(self.ratings.values, full_matrices=False)

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

    def adjust_ratings(self, ratings):
        """
        Adjust the ratings matrix from a 1-5 ratings scale to a -1 to 1 scale indicating user preference
        """
        adjusted_ratings_values = ratings.values - self.user_means.values
        adjusted_ratings_values = adjusted_ratings_values / self.user_std_devs.values
        adjusted_ratings = pd.DataFrame(adjusted_ratings_values, index=ratings.index, columns=ratings.columns)
        adjusted_ratings.fillna(value=0)
        return adjusted_ratings

    def predict(self, k=-1):
        """
        Create the predictions matrix. Optionally for reduced rank k
        :param k: rank of matrix U and T to consider
        :return: The predictions DataFrame
        """
        U_k = self.U
        T_k = self.T
        if k != -1:
            U_k = U_k[:, :k]
            T_k = T_k[:k, :]

        predictions_values = U_k.dot(T_k)
        predictions_values * self.user_std_devs.values + self.user_means.values
        predictions = pd.DataFrame(predictions_values, index=self.ratings.index, columns=self.ratings.columns)
        predictions = predictions.applymap(clamp)
        return predictions
