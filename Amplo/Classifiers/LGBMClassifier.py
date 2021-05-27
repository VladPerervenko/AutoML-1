import copy
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from Amplo.Classifiers import BaseClassifier


class LGBMClassifier(BaseClassifier):

    def __init__(self, params=None):
        """
        Light GBM wrapper
        """
        default = {'verbosity': 0, 'force_col_wise': True}
        super().__init__(default, params)
        self.hasPredictProba = True
        self.set_params(self.params)

    @staticmethod
    def convert_to_dataset(x, y=None):
        # Convert input
        assert type(x) in [pd.DataFrame, pd.Series, np.ndarray], 'Unsupported data input format'
        if isinstance(x, np.ndarray) and len(x.shape) == 0:
            x = x.reshape((-1, 1))

        if y is None:
            return lgb.Dataset(x)

        else:
            assert type(y) in [pd.Series, np.ndarray], 'Unsupported data label format'
            return lgb.Dataset(x, label=y)

    def fit(self, x, y):
        # Split & Convert data
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)
        d_train = self.convert_to_dataset(train_x, train_y)
        d_test = self.convert_to_dataset(test_x, test_y)

        # Set Attrites
        if len(np.unique(y)) == 2:
            self.params['objective'] = 'binary'
        else:
            self.params['objective'] = 'multiclass'
            self.params['num_class'] = len(np.unique(y))
        self.classes_ = np.unique(y)

        # Model training
        self.model = lgb.train(self.params,
                               d_train,
                               valid_sets=d_test,
                               feval=self.eval,
                               verbose_eval=0,
                               callbacks=[self.callbacks] if self.callbacks is not None else None,
                               early_stopping_rounds=100,
                               )
        self.trained = True

    def predict(self, x):
        assert self.trained is True, 'Model not yet trained'
        prediction = self.model.predict(x)

        # Parse into most-likely class
        if len(prediction.shape) == 2:
            # MULTICLASS
            return np.argmax(prediction, axis=1)
        else:
            # BINARY
            return np.round(prediction)

    def predict_proba(self, x):
        assert self.trained is True, 'Model not yet trained'
        prediction = self.model.predict(x)

        # Parse into probabilities
        if len(prediction.shape) == 2:
            # MULTICLASS
            return prediction
        else:
            # BINARY
            return np.hstack((1 - prediction, prediction)).reshape((-1, 2), order='F')

    def set_params(self, params):
        if 'callbacks' in params.keys():
            self.callbacks = params['callbacks']
            params.pop('callbacks')
        self.params = params

    def get_params(self):
        params = copy.copy(self.params)
        if self.callbacks is not None:
            params['callbacks'] = self.callbacks
        return params

    def eval(self, prediction, d_train):
        if self.params['objective'] == 'binary':
            average = 'binary'
        else:
            average = 'micro'
        target = d_train.get_label()
        weight = d_train.get_weight()
        classes = len(np.unique(target))
        if classes > 2:
            prediction = prediction.reshape((-1, classes))
            prediction = np.argmax(prediction, axis=1)
        assert prediction.shape == target.shape
        return "f1", f1_score(target, prediction,
                              sample_weight=weight,
                              average=average), True

