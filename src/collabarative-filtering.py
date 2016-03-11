import numpy as np
import pandas as pd


class CollaborativeFiltering(object):

    def __init__(self, ratings):
        """
        :param ratings:  The user x item ratings DataFrame. Ratings on 1 to 5 scale. 0 represents unknown.
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
    def get_user_similarity(ratings, method='cosine'):
        """

        :param ratings: The ratings DataFrame
        :type ratings: DataFrame
        :param method: 'cosine' or 'pearson'
        :return:
        """
        if method == 'cosine':
            user_similarity = ratings.dot(ratings.transpose())
            normalisation_terms = pd.DataFrame(np.diagonal(user_similarity.values), index=ratings.index).apply(np.sqrt)
            normalisation_terms = normalisation_terms.dot(normalisation_terms.transpose())
            user_similarity = user_similarity.div(normalisation_terms)
            return user_similarity
        elif method == 'pearson':
            return ratings.transpose().corr(method='pearson')
        else:
            raise KeyError(method+' is not an implemented method.')

    def adjust_ratings(self, user_means, user_std_devs=None):
        """

        :param user_means:
        :param user_std_devs:
        :return:
        """
        adjusted_ratings_values = self.ratings.values - user_means.values
        if user_std_devs:
            adjusted_ratings_values = adjusted_ratings_values / user_std_devs.values
        adjusted_ratings = pd.DataFrame(adjusted_ratings_values, index=self.rating.index, columns=self.index.columns)
        adjusted_ratings.fillna(value=0)
        return adjusted_ratings

    def adjust_predictions(self, predicted,  user_means, user_std_devs=None):
        """
        :param predicted:
        :param user_means:
        :param user_std_devs:
        :return:
        """
        if user_std_devs:
            adjusted_predicitons_values = predicted.values * user_std_devs.values
        else:
            adjusted_predicitons_values = predicted.values
        adjusted_predicted_values = adjusted_predicitons_values + user_means.values
        adjusted_predictions = pd.DataFrame(adjusted_predicted_values,
                                            index=self.ratings.index, columns=self.ratings.columns)
        adjusted_predictions = adjusted_predictions.fillna(value=0).applymap(self.clamp)
        return adjusted_predictions


