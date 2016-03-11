import os

import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error


def read_ratings(file_path, sep='::'):
    """
    Reads the ratings file into a user x item DataFrame. Ratings are stored in 'database' form.
    Where each line is in the form: <user_id><sep><item_id><sep><rating><sep><timestamp>
    Unkown values are 0 and ratings are on a 1-5 scale
    :param file_path: The ratings file path
    :param sep: The separator between items
    :return: The user x item ratings DataFrame
    """
    ratings_file = os.path.abspath(file_path)
    column_names = ['userId', 'movieId', 'rating', 'timestamp']
    ratings = pd.read_csv(ratings_file, names=column_names, sep=sep, engine='python')
    ratings = ratings.drop('timestamp', axis=1)
    ratings[['userId', 'movieId']] = ratings[['userId', 'movieId']].astype('int32')
    ratings[['rating']] = ratings[['rating']].astype('int8')
    ratings = ratings.pivot('userId', 'movieId', 'rating').fillna(value=0)
    return ratings


def get_ratings_sparsity(ratings):
    """
    Calculates the sparsity of the ratings matrix
    :param ratings: The user x item ratings DataFrame
    :type ratings: DataFrame
    :return: The percentage sparsity of the DataFrame
    """
    sparsity = float(len(ratings.values.nonzero()[0]))
    sparsity /= (ratings.shape[0] * ratings.shape[1])
    sparsity *= 100

    return 'Sparsity: {:4.2f}%'.format(sparsity)


def split_train_test(ratings, test_ratio=0.2):
    """
    Split the ratings matrix into test and train matrices.
    :param ratings: The original user x item ratings DataFrame
    :type ratings: DataFrame
    :param test_ratio: The ratio of ratings to take for the test dataset
    :type test_ratio: float
    :return: The train and test ratings dataFrames
    """
    test = pd.DataFrame(np.zeros(ratings.shape), index=ratings.index, columns=ratings.columns)
    train = pd.DataFrame(np.zeros(ratings.shape), index=ratings.index, columns=ratings.columns)

    for user in xrange(ratings.shape[0]):
        user_ratings_indexes = ratings.iloc[user, :].nonzero()[0]
        train_indexes, test_indexes = train_test_split(user_ratings_indexes, test_size=test_ratio)
        train.iloc[user, train_indexes] = ratings.iloc[user, train_indexes]
        test.iloc[user, test_indexes] = ratings.iloc[user, test_indexes]

    return train, test


def get_rmse(predicted, actual):
    """
    Calculates the root mean squared error between the predicted and actual ratings DataFrames
    :param predicted: The predicted ratings
    :type predicted: DataFrame
    :param actual: The actual ratings
    :type actual: DataFrame
    :return: root mean squared error
    """
    return np.sqrt(mean_squared_error(actual.values[actual.values.nonzero()].flatten(),
                                      predicted.values[actual.values.nonzero()].flatten()))