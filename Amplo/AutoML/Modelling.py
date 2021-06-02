import os
import copy
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.experimental import enable_hist_gradient_boosting
from ..Classifiers.CatBoostClassifier import CatBoostClassifier
from ..Classifiers.XGBClassifier import XGBClassifier
from ..Classifiers.LGBMClassifier import LGBMClassifier
from sklearn import linear_model
from sklearn import ensemble
from sklearn import svm
from sklearn import metrics


class Modelling:

    def __init__(self,
                 mode='regression',
                 shuffle=False,
                 n_splits=3,
                 objective='accuracy',
                 samples=None,
                 folder='models/',
                 dataset='set_0',
                 store_models=False,
                 store_results=True):
        """
        Class that analyses the performance of various regressors or classifiers.
        @param [str] mode: 'regression' or 'classification'
        @param [bool] shuffle: Whether to shuffle samples for training / validation
        @param [str] objective: Performance metric
        @param [str] folder: Folder to store models and / or results
        @param [int] n_splits: Number of cross-validation splits
        @param [str] dataset: Name of feature set to use
        @param [bool] store_models: Whether to store the trained models
        @param [bool] store_results: Whether to store the results
        """
        # Args
        assert objective in metrics.SCORERS.keys(), '\nPick scorer from sklearn.metrics.SCORERS: \n{}' \
            .format(list(metrics.SCORERS.keys()))
        self.objective = objective
        self.scoring = metrics.SCORERS[objective]
        self.mode = mode
        self.shuffle = shuffle
        self.cvSplits = n_splits
        self.samples = samples
        self.dataset = str(dataset)
        self.storeResults = store_results
        self.storeModels = store_models
        self.results = pd.DataFrame(columns=['date', 'model', 'dataset', 'params', 'mean_objective', 'std_objective',
                                             'mean_time', 'std_time'])

        # Folder
        self.folder = folder if folder[-1] == '/' else folder + '/'
        if store_results or store_models:
            if not os.path.exists(self.folder):
                os.makedirs(self.folder)

        self.needsProba = False  # Whether scorer requires needs_proba attr
        if 'True' in self.scoring._factory_args():
            self.needsProba = True

    def fit(self, x, y):
        # Copy number of samples
        self.samples = len(y)

        # Regression
        if self.mode == 'regression':
            cv = KFold(n_splits=self.cvSplits, shuffle=self.shuffle)
            return self._fit(x, y, cv)

        # Classification
        if self.mode == 'classification':
            cv = StratifiedKFold(n_splits=self.cvSplits, shuffle=self.shuffle)
            return self._fit(x, y, cv)

    def return_models(self):
        models = []

        # All classifiers
        if self.mode == 'classification':
            # The thorough ones
            if self.samples < 25000:
                if not self.needsProba:
                    models.append(linear_model.RidgeClassifier())
                    models.append(svm.SVC(kernel='rbf'))
                models.append(ensemble.BaggingClassifier())
                models.append(ensemble.GradientBoostingClassifier())
                models.append(XGBClassifier())

            # The efficient ones
            else:
                if not self.needsProba:
                    models.append(linear_model.RidgeClassifier())
                models.append(ensemble.HistGradientBoostingClassifier())
                models.append(LGBMClassifier())

            # And the multifaceted ones
            models.append(CatBoostClassifier())
            models.append(ensemble.RandomForestClassifier())

        elif self.mode == 'regression':
            # The thorough ones
            if self.samples < 25000:
                if not self.needsProba:
                    models.append(linear_model.LinearRegression())
                    models.append(svm.SVR(kernel='rbf'))
                models.append(ensemble.BaggingRegressor())
                models.append(ensemble.GradientBoostingRegressor())
                # todo XG Boost

            # The efficient ones
            else:
                if not self.needsProba:
                    models.append(linear_model.LinearRegression())
                models.append(ensemble.HistGradientBoostingRegressor())
                # todo LGBM

            # And the multifaceted ones
            # todo Catboost
            models.append(ensemble.RandomForestRegressor())

        return models

    def _fit(self, x, y, cross_val):
        # Convert to NumPy
        x = np.array(x)
        y = np.array(y).ravel()

        # Data
        print('[Modelling] Splitting data (shuffle=%s, splits=%i, features=%i)' %
              (str(self.shuffle), self.cvSplits, len(x[0])))

        if self.storeResults and 'Initial_Models.csv' in os.listdir(self.folder):
            self.results = pd.read_csv(self.folder + 'Initial_Models.csv')
            for i in range(len(self.results)):
                self.print_results(self.results.iloc[i])

        else:

            # Models
            self.models = self.return_models()

            # Loop through models
            for master_model in self.models:

                # Time & loops through Cross-Validation
                val_score = []
                train_score = []
                train_time = []
                for t, v in cross_val.split(x, y):
                    t_start = time.time()
                    xt, xv, yt, yv = x[t], x[v], y[t], y[v]
                    model = copy.copy(master_model)
                    model.fit(xt, yt)
                    val_score.append(self.scoring(model, xv, yv))
                    train_score.append(self.scoring(model, xt, yt))
                    train_time.append(time.time() - t_start)

                # Append results
                result = {
                    'date': datetime.today().strftime('%d %b %y'),
                    'model': type(model).__name__,
                    'dataset': self.dataset,
                    'params': model.get_params(),
                    'mean_objective': np.mean(val_score),
                    'std_objective': np.std(val_score),
                    'worst_case': np.mean(val_score) - np.std(val_score),
                    'mean_time': np.mean(train_time),
                    'std_time': np.std(train_time)
                }
                self.results = self.results.append(result, ignore_index=True)
                self.print_results(result)

                # Store model
                if self.storeModels:
                    joblib.dump(model, self.folder + type(model).__name__ + '_{:.4f}.joblib'.format(np.mean(val_score)))

            # Store CSV
            if self.storeResults:
                self.results.to_csv(self.folder + 'Initial_Models.csv')

        # Return results
        return self.results

    def print_results(self, result):
        print('[Modelling] {} {}: {:.4f} \u00B1 {:.4f}, training time: {:.1f} s'.format(
            result['model'].ljust(30), self.objective, result['mean_objective'],
            result['std_objective'], result['mean_time']))
