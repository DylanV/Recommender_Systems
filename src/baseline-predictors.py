import numpy as np
import pandas as pd


class BaselinePredicor(object):

    def __init__(self, ratings):
        self.ratings = ratings
