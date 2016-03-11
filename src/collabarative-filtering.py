import numpy as np
import pandas as pd

from aux import clamp


class CollaborativeFiltering(object):

    def __init__(self, ratings):
        """
        :param ratings:  The user x item ratings DataFrame. Ratings on 1 to 5 scale. 0 represents unknown.
        :type ratings: DataFrame
        """
        self.ratings = ratings
        self.user_means = self.calculate_user_means(self.ratings)
        self.user_std_devs = self.calculate_user_std_devs(self.ratings)

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
        Calculate the user similarity matrix
        :param ratings: The ratings DataFrame
        :type ratings: DataFrame
        :param method: <'cosine', 'pearson'> the similarity measure to be used
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

    @staticmethod
    def adjust_user_similarity_knn(user_similarity, k):
        """
        Adjust the user similarity matrix for use with k-nearest neighbours.
        For each user the top k neighbours are kept and for the rest the similarity is set to 0.
        Thus in the prediction their ratings will be ignored in the calculation.
        :param user_similarity: The user similarity DataFrame
        :param k: the number of neighbours for k-nearest neighbour
        :return: The adjusted similarity DataFrame
        """
        adjusted_similarity = pd.DataFrame(np.zeros(user_similarity.shape),
                                           index=user_similarity.index, columns=user_similarity.columns)

        for user in user_similarity.iterrows():
            top_k_indexes = user[1].sort_values(ascending=False).iloc[0:k+1].index.values
            adjusted_similarity.loc[user[0], top_k_indexes] = user_similarity.loc[user[0], top_k_indexes]

        return adjusted_similarity

    @staticmethod
    def adjust_ratings(ratings, user_means, user_std_devs=None):
        """
        Adjust the ratings matrix from a 1-5 ratings scale to a -1 to 1 scale indicating user preference
        :param ratings: The ratings DataFrame
        :param user_means: The user means
        :param user_std_devs: The user standatd deviations
        :return: The adjusted ratings DataFrame
        """
        adjusted_ratings_values = ratings.values - user_means.values
        if user_std_devs:
            adjusted_ratings_values = adjusted_ratings_values / user_std_devs.values
        adjusted_ratings = pd.DataFrame(adjusted_ratings_values, index=ratings.index, columns=ratings.columns)
        adjusted_ratings.fillna(value=0)
        return adjusted_ratings

    @staticmethod
    def adjust_predictions(predicted,  user_means, user_std_devs=None):
        """
        Adjust the predictions back to a 1-5 ratings scale
        :param predicted: The predicted DataFrame
        :param user_means: The user means
        :param user_std_devs: The user standard deviations. Optional
        :return: The adjusted predictions
        """
        if user_std_devs:
            adjusted_predictions_values = predicted.values * user_std_devs.values
        else:
            adjusted_predictions_values = predicted.values
        adjusted_predicted_values = adjusted_predictions_values + user_means.values
        adjusted_predictions = pd.DataFrame(adjusted_predicted_values,
                                            index=predicted.index, columns=predicted.columns)
        adjusted_predictions = adjusted_predictions.fillna(value=0).applymap(clamp)
        return adjusted_predictions

    def predict_user_user(self, adjust='full', similarity='cosine', k=-1):
        """
        Do user user prediction.
        Options to use mean or mean and std-dev (full) adjusted ratings (default full),
        Cosine or pearson correlation for user similarity (default cosine),
        and the number of neighbours to consider (default all).
        :param k: The number of neighbours for k-nearest neighbours
        :param adjust: <'mean','full'> adjust the ratings for user means or means and standard deviations respectively
        :param similarity: <'cosine', 'pearson'> The similarity measure for the user similarity
        :return: The prediction DataFrame
        """
        ratings = self.ratings
        if adjust == 'mean':
            ratings = self.adjust_ratings(ratings, self.user_means)
        elif adjust == 'full':
            ratings = self.adjust_ratings(ratings, self.user_means, self.user_std_devs)

        user_similarity = self.get_user_similarity(ratings, method=similarity)
        if k != -1:
            user_similarity = self.adjust_user_similarity_knn(user_similarity, k)

        predictions = user_similarity.dot(ratings)
        denom = user_similarity.abs().sum().transpose()
        predictions = predictions.div(denom, axis='index')

        if adjust == 'mean':
            predictions = self.adjust_predictions(predictions, self.user_means)
        elif adjust == 'full':
            predictions = self.adjust_predictions(self.user_means, self.user_std_devs)

        predictions = pd.DataFrame(predictions.values, index=self.ratings.index, columns=self.ratings.columns)
        return predictions
