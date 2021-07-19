import unittest
import pandas as pd
from sklearn.datasets import make_regression
from Amplo.Regressors import CatBoostRegressor
from .test_regressor import TestRegression


class TestCatBoostRegressor(unittest.TestCase, TestRegression):

    @classmethod
    def setUpClass(cls):
        cls.model = CatBoostRegressor()
        x, y = make_regression(n_informative=15)
        cls.x, cls.y = pd.DataFrame(x), pd.Series(y)
