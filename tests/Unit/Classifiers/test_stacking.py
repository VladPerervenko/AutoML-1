import unittest
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import SCORERS
from sklearn.datasets import make_classification
from Amplo.Classifiers import StackingClassifier


class TestStackingClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = StackingClassifier()
        x, y = make_classification(n_classes=5, n_informative=15)
        cls.x, cls.y = pd.DataFrame(x), pd.Series(y)

    def test_set_params(self):
        self.model.set_params(**{'CatBoostClassifier': {'depth': 10}})

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
        assert np.allclose(prediction.astype('int'), prediction)

    def test_probability(self):
        model = clone(self.model)
        model.fit(self.x, self.y)
        prediction = model.predict_proba(self.x)

        assert not np.isnan(prediction).any(), 'NaN in prediction: {}'.format(prediction)
        assert len(prediction.shape) == 2
        assert prediction.shape[1] == len(np.unique(self.y))
        assert np.allclose(np.sum(prediction, axis=1), 1)

    def test_cloneable(self):
        model = clone(self.model)
        params = {'CatBoostClassifier': {'depth': 10}}
        cloned = clone(clone(model).set_params(**params))
        cloned.fit(self.x, self.y)
        prediction = cloned.predict_proba(self.x)

        assert len(prediction.shape) != 1
        assert np.allclose(np.sum(prediction, axis=1), 1)

    def test_scorer(self):
        model = clone(self.model)
        model.fit(self.x, self.y)
        SCORERS['neg_log_loss'](model, self.x, self.y)
        SCORERS['accuracy'](model, self.x, self.y)
        SCORERS['f1_micro'](model, self.x, self.y)

    def test_binary(self):
        model = clone(self.model)
        x, y = make_classification()
        model.fit(x, y)
        model.predict(x)
        model.predict_proba(x)
        SCORERS['neg_log_loss'](model, x, y)
        SCORERS['accuracy'](model, x, y)
        SCORERS['f1_micro'](model, x, y)
