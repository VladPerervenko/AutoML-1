import unittest
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from Amplo.GridSearch import HalvingGridSearch
from Amplo.AutoML import Modeller


class TestHalvingGridSearch(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        x, y = make_classification()
        cls.cx = pd.DataFrame(x)
        cls.cy = pd.Series(y)
        x, y = make_regression()
        cls.rx = pd.DataFrame(x)
        cls.ry = pd.Series(y)

    def test_small_regression(self):
        models = Modeller(mode='regression', objective='r2', samples=100).return_models()
        for model in models:
            grid_search = HalvingGridSearch(model,
                                            cv=KFold(n_splits=3),
                                            verbose=0,
                                            candidates=2,
                                            scoring='r2')
            results = grid_search.fit(self.rx, self.ry)

            # Tests
            assert isinstance(results, pd.DataFrame), 'Not a DF {}'.format(type(model).__name__)
            assert all([x in results.keys() for x in ['worst_case', 'mean_objective', 'std_objective', 'params',
                                                      'mean_time', 'std_time']]), 'Keys missing {}'.format(type(model)
                                                                                                           .__name__)
            assert len(results) > 0, 'Empty results {}'.format(type(model).__name__)

    def test_big_regression(self):
        models = Modeller(mode='regression', objective='r2', samples=50000).return_models()
        for model in models:
            grid_search = HalvingGridSearch(model,
                                            cv=KFold(n_splits=3),
                                            verbose=0,
                                            candidates=2,
                                            scoring='r2')
            results = grid_search.fit(self.rx, self.ry)

            # Tests
            assert isinstance(results, pd.DataFrame), 'Not a DF {}'.format(type(model).__name__)
            assert all([x in results.keys() for x in ['worst_case', 'mean_objective', 'std_objective', 'params',
                                                      'mean_time', 'std_time']]), 'Keys missing {}'.format(type(model)
                                                                                                           .__name__)
            assert len(results) > 0, 'Empty results {}'.format(type(model).__name__)

    def test_small_classification(self):
        models = Modeller(mode='classification', objective='accuracy', samples=100).return_models()
        for model in models:
            assert hasattr(model, 'fit'), 'Has no fit attribute'
            grid_search = HalvingGridSearch(model,
                                            cv=StratifiedKFold(n_splits=3),
                                            verbose=0,
                                            candidates=2,
                                            scoring='accuracy')
            results = grid_search.fit(self.cx, self.cy)

            # Tests
            assert isinstance(results, pd.DataFrame), 'Not a DF {}'.format(type(model).__name__)
            assert all([x in results.keys() for x in ['worst_case', 'mean_objective', 'std_objective', 'params',
                                                      'mean_time', 'std_time']]), 'Keys missing {}'.format(type(model)
                                                                                                           .__name__)
            assert len(results) > 0, 'Empty results {}'.format(type(model).__name__)

    def test_big_classification(self):
        models = Modeller(mode='classification', objective='accuracy', samples=50000).return_models()
        for model in models:
            grid_search = HalvingGridSearch(model,
                                            cv=StratifiedKFold(n_splits=3),
                                            verbose=0,
                                            candidates=2,
                                            scoring='accuracy')
            results = grid_search.fit(self.cx, self.cy)

            # Tests
            assert isinstance(results, pd.DataFrame), 'Not a DF {}'.format(type(model).__name__)
            assert all([x in results.keys() for x in ['worst_case', 'mean_objective', 'std_objective', 'params',
                                                      'mean_time', 'std_time']]), 'Keys missing {}'.format(type(model)
                                                                                                           .__name__)
            assert len(results) > 0, 'Empty results {}'.format(type(model).__name__)
