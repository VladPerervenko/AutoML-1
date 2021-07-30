import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from Amplo.AutoML import DataProcesser
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
        dp = DataProcesser('target')
        cleaned = dp.fit_transform(self.regression)
        assert check_dataframe_quality(cleaned)

    def test_classification(self):
        dp = DataProcesser('target')
        cleaned = dp.fit_transform(self.classification)
        assert check_dataframe_quality(cleaned)

    def test_outliers(self):
        x = pd.DataFrame({'a': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1e15],
                          'target': np.linspace(0, 1, 24).tolist()})

        # Clip
        dp = DataProcesser(outlier_removal='clip', target='target')
        xt = dp.fit_transform(x)
        assert xt.max().max() < 1e15, "Outlier not removed"
        assert not xt.isna().any().any(), "NaN found"

        # z-score
        dp = DataProcesser(outlier_removal='z-score', target='target')
        xt = dp.fit_transform(x)
        assert xt.max().max() < 1e15, "Outlier not removed"
        assert not xt.isna().any().any(), "NaN found"
        assert np.isclose(dp.transform(pd.DataFrame({'a': [1e14], 'b': [1]})).max().max(), 1e14)
        assert dp.transform(pd.DataFrame({'a': [1e16], 'b': [1]})).max().max() == 1

        # Quantiles
        dp = DataProcesser(outlier_removal='quantiles', target='target')
        print(x.quantile(0.75))
        xt = dp.fit_transform(x)
        print(dp._q3, xt)
        assert xt.max().max() < 1e15, "Outlier not removed"
        assert not xt.isna().any().any(), "NaN found"
        assert dp.transform(pd.DataFrame({'a': [2], 'b': [-2]})).max().max() == 0

    def test_duplicates(self):
        x = pd.DataFrame({'a': [1, 2, 1], 'a': [1, 2, 1], 'b': [3, 1, 3]})
        dp = DataProcesser()
        xt = dp.fit_transform(x)
        assert len(xt) == 2, "Didn't remove duplicate rows"
        assert len(xt.keys()) == 2, "Didn't remove duplicate columns"

    def test_constants(self):
        x = pd.DataFrame({'a': [1, 1, 1, 1, 1], 'b': [1, 2, 3, 5, 6]})
        dp = DataProcesser()
        xt = dp.fit_transform(x)
        assert 'a' not in xt.keys(), "Didn't remove constant column"

    def test_dummies(self):
        x = pd.DataFrame({'a': ['a', 'b', 'c', 'b', 'c', 'a']})
        dp = DataProcesser(cat_cols=['a'])
        xt = dp.fit_transform(x)
        assert 'a' not in xt.keys(), "'a' still in keys"
        assert 'a_b' in xt.keys(), "a_b missing"
        assert 'a_c' in xt.keys(), "a_c missing"
        xt2 = dp.transform(pd.DataFrame({'a': ['a', 'c']}))
        assert np.allclose(xt2.values, pd.DataFrame({'a_b': [0, 0], 'a_c': [0, 1]}).values), "Converted not correct"

    def test_settings(self):
        x = pd.DataFrame({'a': ['a', 'b', 'c', 'b', 'c', 'a'], 'b': [1, 1, 1, 1, 1, 1]})
        dp = DataProcesser(cat_cols=['a'])
        xt = dp.fit_transform(x)
        assert len(xt.keys()) == 2
        settings = dp.get_settings()
        dp2 = DataProcesser()
        dp2.load_settings(settings)
        xt2 = dp2.transform(pd.DataFrame({'a': ['a', 'b'], 'b': [1, 2]}))
        assert np.allclose(pd.DataFrame({'b': [1.0, 2.0], 'a_b': [0, 1], 'a_c': [0, 0]}).values, xt2.values)
