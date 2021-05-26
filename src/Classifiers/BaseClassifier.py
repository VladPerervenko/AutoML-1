class BaseClassifier:

    def __init__(self, model, params):
        self.defaultParams = params
        self.model = model(**params)
        self.hasPredictProba = hasattr(model, 'predict_proba')

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
        self.model.fit(x, y)
        return self
