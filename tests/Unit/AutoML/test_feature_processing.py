import unittest
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from Amplo.AutoML import FeatureProcessing
from Amplo.Utils import check_dataframe_quality


class TestDataProcessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        x, y = make_classification()
        cls.classification = pd.DataFrame({'target': y})
        for i in range(x.shape[1]):
            cls.classification['Feature_{}'.format(i)] = x[:, i]
        x, y = make_regression()
        cls.regression = pd.DataFrame({'target': y})
        for i in range(x.shape[1]):
            cls.regression['Feature_{}'.format(i)] = x[:, i]

    def test_regression(self):
        fp = FeatureProcessing(max_lags=2, mode='regression', folder='tmp/')
        extract = fp.extract(self.regression.drop('target', axis=1), self.regression['target'])
        assert check_dataframe_quality(extract)

    def test_classification(self):
        fp = FeatureProcessing(max_lags=2, mode='classification', folder='tmp/')
        extract = fp.extract(self.classification.drop('target', axis=1), self.classification['target'])
        assert check_dataframe_quality(extract)
