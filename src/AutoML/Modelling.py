import os
import copy
import time
import joblib
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from AutoML.src.Classifiers.CatBoostClassifier import CatBoostClassifier
from sklearn import linear_model
from sklearn import ensemble
from sklearn import svm


class Modelling:

    def __init__(self,
                 mode='regression',
                 shuffle=False,
                 scoring=None,
                 folder='models/',
                 n_splits=3,
                 dataset=0,
                 store_models=False,
                 store_results=True):
        """
        Class that analyses the performance of various regressors or classifiers.
        @param [str] mode: 'regression' or 'classification'
        @param [bool] shuffle: Whether to shuffle samples for training / validation
        @param [SciKit make_scorer] scoring: Performance metric
        @param [str] folder: Folder to store models and / or results
        @param [int] n_splits: Number of cross-validation splits
        @param [str] dataset: Name of feature set to use
        @param [bool] store_models: Whether to store the trained models
        @param [bool] store_results: Whether to store the results
        """
        # Args
        self.scoring = scoring
        self.mode = mode
        self.shuffle = shuffle
        self.cvSplits = n_splits
        self.dataset = str(dataset)
        self.storeResults = store_results
        self.storeModels = store_models
        self.folder = folder if folder[-1] == '/' else folder + '/'

        self.samples = None      # Number of samples in data
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
                if not self.needsProba:     # Arent probabilistic
                    models.append(linear_model.RidgeClassifier())
                    models.append(svm.SVC(kernel='rbf'))
                models.append(ensemble.BaggingClassifier())
                models.append(ensemble.GradientBoostingClassifier())
                # todo XG Boost

            # The efficient ones
            else:
                if not self.needsProba:
                    models.append(linear_model.RidgeClassifier())
                models.append(ensemble.HistGradientBoostingClassifier())
                # todo LGBM

            # And the multifaceted ones
            models.append(CatBoostClassifier())
            models.append(ensemble.RandomForestClassifier())

        elif self.mode == 'regression':
            # The thorough ones
            if self.samples < 25000:
                if not self.needsProba:     # Arent probabilistic
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
            # models.append(CatBoostClassifier())
            models.append(ensemble.RandomForestRegressor())

        # Filter predict_proba models
        models = [m for m in models if hasattr(m, 'predict_proba')]

        return models

    def _fit(self, x, y, cross_val):
        # Convert to NumPy
        x = np.array(x)
        y = np.array(y).ravel()

        # Data
        print('[Modelling] Splitting data (shuffle=%s, splits=%i, features=%i)' % (str(self.shuffle), self.cvSplits, len(x[0])))

        if self.store_results and 'Initial_Models.csv' in os.listdir(self.folder):
            results = pd.read_csv(self.folder + 'Initial_Models.csv')
        else:
            results = pd.DataFrame(columns=['date', 'model', 'dataset', 'params', 'mean_objective', 'std_objective', 'mean_time', 'std_time'])

        # Models
        self.models = self.return_models()

        # Loop through models
        for master_model in self.models:
            # Check first if we don't already have the results
            ind = np.where(np.logical_and(np.logical_and(
                results['model'] == type(master_model).__name__,
                results['dataset'] == self.dataset),
                results['date'] == datetime.today().strftime('%d %b %y, %Hh')))[0]
            if len(ind) != 0:
                print('[Modelling] %s %s: %.4f, training time: %.1f s' %
                      (type(model).__name__.ljust(60), self.scoring._score_func.__name__,
                       np.mean(results.iloc[0]['mean_objective']), time.time() - t_start))
                continue

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

            # Results
            results = results.append({'date': datetime.today().strftime('%d %b %y'), 'model': type(model).__name__,
                                      'dataset': self.dataset, 'params': model.get_params(),
                                      'mean_objective': np.mean(val_score),
                                      'std_objective': np.std(val_score), 'mean_time': np.mean(train_time),
                                      'std_time': np.std(train_time)}, ignore_index=True)

            print('[Modelling] %s %s: %.4f, training time: %.1f s' %
                  (type(model).__name__.ljust(60), self.scoring._score_func.__name__,
                   np.mean(results.iloc[0]['mean_objective']), time.time() - t_start))

            if self.store_models:
                joblib.dump(model, self.folder + type(model).__name__ + '_%.4f.joblib' % self.acc[-1])

        # Store CSV
        if self.store_results:
            results.to_csv(self.folder + 'Initial_Models.csv')

        return results
