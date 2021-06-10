import unittest
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from Amplo.AutoML import Modelling


class TestModelling(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        x, y = make_classification()
        cls.cx, cls.cy = pd.DataFrame(x), pd.Series(y)
        x, y = make_regression()
        cls.rx, cls.ry = pd.DataFrame(x), pd.Series(y)

    def test_regression(self):
        mod = Modelling(mode='regression', objective='r2', folder='tmp/')
        mod.fit(self.rx, self.ry)
        # Tests
        assert not mod.needsProba, 'R2 does not need probability'
        assert isinstance(mod.results, pd.DataFrame), 'Results should be type pd.DataFrame'
        assert len(mod.results) != 0, 'Results empty'
        assert mod.results['mean_objective'].max() < 1, 'R2 needs to be smaller than 1'
        assert not mod.results['mean_objective'].isna().any(), "Mean Objective shouldn't contain NaN"
        assert not mod.results['std_objective'].isna().any(), "Std Objective shouldn't contain NaN"
        assert not mod.results['mean_time'].isna().any(), "Mean time shouldn't contain NaN"
        assert not mod.results['std_time'].isna().any(), "Std time shouldn't contain NaN"
        assert 'date' in mod.results.keys()
        assert 'model' in mod.results.keys()
        assert 'dataset' in mod.results.keys()
        assert 'params' in mod.results.keys()

    def test_classification(self):
        mod = Modelling(mode='classification', objective='neg_log_loss', folder='tmp/')
        mod.fit(self.cx, self.cy)
        # Tests
        assert mod.needsProba, 'Neg Log Loss does not need probability'
        assert isinstance(mod.results, pd.DataFrame), 'Results should be type pd.DataFrame'
        assert len(mod.results) != 0, 'Results empty'
        assert mod.results['mean_objective'].max() < 1, 'R2 needs to be smaller than 1'
        assert not mod.results['mean_objective'].isna().any(), "Mean Objective shouldn't contain NaN"
        assert not mod.results['std_objective'].isna().any(), "Std Objective shouldn't contain NaN"
        assert not mod.results['mean_time'].isna().any(), "Mean time shouldn't contain NaN"
        assert not mod.results['std_time'].isna().any(), "Std time shouldn't contain NaN"
        assert 'date' in mod.results.keys()
        assert 'model' in mod.results.keys()
        assert 'dataset' in mod.results.keys()
        assert 'params' in mod.results.keys()

    def test_return(self):
        Modelling(mode='regression', samples=100).return_models()
        Modelling(mode='regression', samples=100000).return_models()
        Modelling(mode='classification', samples=100).return_models()
        Modelling(mode='classification', samples=100000).return_models()
