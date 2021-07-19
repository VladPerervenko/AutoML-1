import unittest
import pandas as pd
from sklearn.datasets import make_regression
from Amplo.Regressors import XGBRegressor
from .test_regressor import TestRegression


class TestXGBRegressor(unittest.TestCase, TestRegression):

    @classmethod
    def setUpClass(cls):
        cls.model = XGBRegressor()
        x, y = make_regression(n_informative=15)
        cls.x, cls.y = pd.DataFrame(x), pd.Series(y)
