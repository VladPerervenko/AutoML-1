import unittest
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from Amplo.AutoML import DataProcessing
from Amplo.Utils import check_dataframe_quality


class TestDataProcessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.classification, y = load_iris(return_X_y=True, as_frame=True)
        cls.classification['target'] = y
        x, y = load_boston(return_X_y=True)
        cls.regression = pd.DataFrame({'target': y})
        for i in range(x.shape[1]):
            cls.regression['feature_{}'.format(i)] = x[:, i]

    def test_regression(self):
        dp = DataProcessing('target', mode='regression', folder='tmp/')
        cleaned = dp.clean(self.regression)
        assert check_dataframe_quality(cleaned)

    def test_classification(self):
        dp = DataProcessing('target', mode='classification', folder='tmp/')
        cleaned = dp.clean(self.classification)
        assert check_dataframe_quality(cleaned)
