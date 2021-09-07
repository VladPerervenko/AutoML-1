import unittest
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.base import clone
from sklearn.metrics import SCORERS
from Amplo.Regressors import StackingRegressor


class TestStackingRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = StackingRegressor()
        x, y = make_regression()
        cls.x, cls.y = pd.DataFrame(x), pd.Series(y)

    def test_set_params(self):
        self.model.set_params(**{'CatBoostRegressor': {'depth': 10}})

    def test_get_params(self):
        self.model.get_params()

    def test_pd_fit(self):
        model = clone(self.model)
        model.fit(self.x, self.y)

    def test_np_fit(self):
        model = clone(self.model)
        model.fit(self.x.to_numpy(), self.y.to_numpy())

    def test_trained_attr(self):
        model = clone(self.model)
        assert hasattr(model, 'trained')
        assert model.trained is False
        model.fit(self.x, self.y)
        assert model.trained is True

    def test_predict(self):
        model = clone(self.model)
        model.fit(self.x, self.y)
        prediction = model.predict(self.x)

        assert len(prediction.shape) == 1

    def test_cloneable(self):
        model = clone(self.model)
        params = {'CatBoostRegressor': {'depth': 10}}
        cloned = clone(clone(model).set_params(**params))
        cloned.fit(self.x, self.y)
        prediction = cloned.predict(self.x)

        assert len(prediction.shape) == 1

    def test_scorer(self):
        model = clone(self.model)
        model.fit(self.x, self.y)
        SCORERS['neg_mean_absolute_error'](model, self.x, self.y)
        SCORERS['neg_mean_squared_error'](model, self.x, self.y)
        SCORERS['r2'](model, self.x, self.y)

