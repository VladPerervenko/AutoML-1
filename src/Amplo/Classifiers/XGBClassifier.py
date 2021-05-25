import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split


class XGBClassifier:

    def __init__(self, params=None):
        """
        XG Boost wrapper
        @param params: Model parameters
        """
        # Args
        self.params = params

        # Parameters
        self.model = None
        self.trained = False

    def fit(self, x, y, callback=None):
        # Split & Convert data
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)
        d_train = xgb.DMatrix(train_x, label=train_y)
        d_test = xgb.DMatrix(test_x, label=test_y)

        # Model training
        self.model = xgb.train(self.params,
                               d_train,
                               evals=[(d_test, 'validation')],
                               callbacks=[callback])
        self.trained = True

    def predict(self, x):
        assert self.trained is True, 'Model not yet trained'
        # Convert if dataframe
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        # Convert if single column
        if len(x.shape) == 1:
            x = x.reshape((-1, 1))
        # Convert to DMatrix
        d_predict = xgb.DMatrix(x)
        self.model.predict(d_predict)

    def predict_proba(self, x):
        assert self.trained is True, 'Model not yet trained'
        # Convert if dataframe
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        # Convert if single column
        if len(x.shape) == 1:
            x = x.reshape((-1, 1))
        # Convert to DMatrix
        d_predict = xgb.DMatrix(x)
        self.model.predict_proba(d_predict)

    def set_params(self, params):
        self.params = params

    def get_params(self):
        return self.params
