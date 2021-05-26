import copy
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


class LGBMClassifier:

    def __init__(self, params=None):
        """
        Light GBM wrapper
        """
        # Parameters
        self.model = None
        self.defaultParams = {
            'verbosity': 0,
            'force_col_wise': True
        }
        self.params = params if params is not None else self.defaultParams
        for key in [k for k in self.defaultParams if k not in self.params]:
            self.params[key] = self.defaultParams[key]
        self.callbacks = None
        self.trained = False

        # Parse params
        self.set_params(self.params)

    def fit(self, x, y):
        # Split & Convert data
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)
        d_train = lgb.Dataset(train_x, label=train_y)
        d_test = lgb.Dataset(test_x, label=test_y)

        # Set Objective
        if len(np.unique(y)) == 2:
            self.params['objective'] = 'binary'
        else:
            self.params['objective'] = 'multiclass'
            self.params['num_class'] = len(np.unique(y))

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
        # Convert if dataframe
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        # Convert if single column
        if len(x.shape) == 1:
            x = x.reshape((-1, 1))
        # Convert to Dataset
        d_predict = lgb.Dataset(x)
        return self.model.predict(d_predict)

    def predict_proba(self, x):
        assert self.trained is True, 'Model not yet trained'
        # Convert if dataframe
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        # Convert if single column
        if len(x.shape) == 1:
            x = x.reshape((-1, 1))
        # Convert to DMatrix
        d_predict = lgb.Dataset(x)
        self.model.predict_proba(d_predict)

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

