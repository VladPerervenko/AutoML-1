import numpy as np


class BaseClassifier:

    def __init__(self, defaultParams=None, params=None):
        self.defaultParams = defaultParams
        self.model = None
        self.hasPredictProba = False
        self.trained = False
        self.classes_ = None
        self.callbacks = None
        self.params = params if params is not None else self.defaultParams
        for key in [k for k in self.defaultParams if k not in self.params]:
            self.params[key] = self.defaultParams[key]

    def get_params(self):
        return self.model.get_params()

    def set_params(self, args):
        self.model.set_params(**args)
        return self

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        if self.hasPredictProba:
            return self.model.predict_proba(x)

    def score(self, x, y):
        return self.model.score(x, y)

    def fit(self, x, y):
        self.classes_ = np.unique(y)
        self.model.fit(x, y)
        self.trained = True
        return self
