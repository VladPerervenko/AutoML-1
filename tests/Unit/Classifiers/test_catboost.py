import unittest
import numpy as np
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.metrics import SCORERS
from Amplo.Classifiers import CatBoostClassifier


class TestCatBoostClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.x, cls.y = load_iris(return_X_y=True, as_frame=True)

    def test_set_params(self):
        model = CatBoostClassifier()
        model.set_params(**{'depth': 200})

    def test_get_params(self):
        model = CatBoostClassifier()
        model.get_params()

    def test_pd_fit(self):
        model = CatBoostClassifier()
        model.fit(self.x, self.y)

    def test_np_fit(self):
        model = CatBoostClassifier()
        model.fit(self.x.to_numpy(), self.y.to_numpy())

    def test_trained_attr(self):
        model = CatBoostClassifier()
        assert hasattr(model, 'trained')
        assert model.trained is False
        model.fit(self.x, self.y)
        assert model.trained is True

    def test_predict(self):
        model = CatBoostClassifier()
        model.fit(self.x, self.y)
        prediction = model.predict(self.x)

        assert len(prediction.shape) == 1
        assert np.allclose(prediction.astype('int'), prediction)

    def test_probability(self):
        model = CatBoostClassifier()
        model.fit(self.x, self.y)
        prediction = model.predict_proba(self.x)

        assert not np.isnan(prediction).any(), 'NaN in prediction: {}'.format(prediction)
        assert len(prediction.shape) == 2
        assert prediction.shape[1] == len(np.unique(self.y))
        assert np.allclose(np.sum(prediction, axis=1), 1)

    def test_cloneable(self):
        model = CatBoostClassifier()
        params = {'depth': 5}
        cloned = clone(clone(model).set_params(**params))
        cloned.fit(self.x, self.y)
        prediction = cloned.predict_proba(self.x)

        assert len(prediction.shape) != 1
        assert np.allclose(np.sum(prediction, axis=1), 1)

    def test_scorer(self):
        model = CatBoostClassifier()
        model.fit(self.x, self.y)
        SCORERS['neg_log_loss'](model, self.x, self.y)
        SCORERS['accuracy'](model, self.x, self.y)
        SCORERS['f1_micro'](model, self.x, self.y)
