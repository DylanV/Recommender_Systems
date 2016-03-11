import os

import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error


def read_ratings(file_path, sep='::'):
    """

    :param file_path:
    :param sep:
    :return:
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

    :param ratings:
    :type ratings: DataFrame
    :return:
    """
    sparsity = float(len(ratings.values.nonzero()[0]))
    sparsity /= (ratings.shape[0] * ratings.shape[1])
    sparsity *= 100
    return 'Sparsity: {:4.2f}%'.format(sparsity)


def split_train_test(ratings, test_ratio=0.2):
    """

    :param ratings:
    :type ratings: DataFrame
    :param test_ratio:
    :type test_ratio: float
    :return:
    """
    test = pd.DataFrame(np.zeros(ratings.shape), index=ratings.index, columns=ratings.columns)
    train = pd.DataFrame(np.zeros(ratings.shape), index=ratings.index, columns=ratings.columns)
    for user in xrange(ratings.shape[0]):
        user_ratings_indexes = ratings.iloc[user, :].nonzero()[0]
        num_samples = np.rint(len(user_ratings_indexes)*test_ratio).astype('int32')
        train_indexes, test_indexes = train_test_split(user_ratings_indexes, test_size=0.2)
        train.iloc[user, train_indexes] = ratings.iloc[user, train_indexes]
        test.iloc[user, test_indexes] = ratings.iloc[user, test_indexes]
    train.loc[:, :] = train.loc[:, :].astype('float32')
    test.loc[:, :] = test.loc[:, :].astype('float32')
    return train, test


def get_rmse(predicted, actual):
    """

    :param predicted:
    :type predicted: DataFrame
    :param actual:
    :type actual: DataFrame
    :return:
    """
    return np.sqrt(mean_squared_error(actual.values[actual.values.nonzero()].flatten(),
                                      predicted.values[actual.values.nonzero()].flatten()))