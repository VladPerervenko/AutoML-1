import copy
import time
import optuna
import numpy as np
import pandas as pd
import multiprocessing as mp
import catboost
import xgboost
import lightgbm
from sklearn import svm
from sklearn import ensemble
from sklearn import linear_model
from .BaseGridSearch import BaseGridSearch


class OptunaGridSearch(BaseGridSearch):

    def __init__(self, model, params=None, cv=None, scoring=None, verbose=None, timeout=3600, n_trials=None):
        super().__init__(model, params, cv, scoring, verbose)
        self.timeout = timeout
        self.nTrials = n_trials
        self.x, self.y = None, None

    def get_params(self, trial):
        if isinstance(self.model, linear_model.Lasso) or \
           isinstance(self.model, linear_model.Ridge) or \
           isinstance(self.model, linear_model.RidgeClassifier):
            return {
                'alpha': trial.suggest_uniform('alpha', 0, 10),
            }
        elif isinstance(self.model, svm.SVC) or \
                isinstance(self.model, svm.SVR):
            return {
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto', 0.001, 0.01, 0.1, 0.5, 1]),
                'C': trial.suggest_uniform('C', 0, 10),
            }
        elif isinstance(self.model, catboost.CatBoostRegressor):
            return {
                    'loss_function': trial.suggest_categorical('loss', ['MAE', 'RMSE']),
                    'learning_rate': trial.suggest_uniform('learning_rate', 0, 1),
                    'l2_leaf_reg': trial.suggest_uniform('l2_leaf_reg', 0, 10),
                    'depth': trial.suggest_int('depth', 3, 15),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 1000),
                    'max_leaves': trial.suggest_int('max_leaves', 10, 250),
                }
        elif isinstance(self.model, ensemble.GradientBoostingRegressor):
            return {
                'loss': trial.suggest_categorical('loss', ['ls', 'lad', 'huber']),
                'learning_rate': trial.suggest_uniform('learning_Rate', 0, 1),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
            }
        elif isinstance(self.model, ensemble.HistGradientBoostingRegressor):
            return {
                'max_iter': trial.suggest_int('max_iter', 100, 250),
                'max_bins': trial.suggest_int('max_bins', 100, 255),
                'loss': trial.suggest_categorical('loss', ['least_squares', 'least_absolute_deviation']),
                'l2_regularization': trial.suggest_uniform('l2_regularization', 0, 10),
                'learning_rate': trial.suggest_uniform('learning_rate', 0, 1),
                'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 30, 150),
                'early_stopping': [True],
            }
        elif isinstance(self.model, ensemble.RandomForestRegressor):
            return {
                'criterion': trial.suggest_categorical('criterion', ['mse', 'mae']),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt']),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 1000),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            }
        elif isinstance(self.model, xgboost.XGBRegressor):
            return {
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
                'learning_rate': trial.suggest_uniform('learning_rate', 0, 10),
                'verbosity': [0],
                'n_jobs': [mp.cpu_count() - 1],
            }
        elif isinstance(self.model, lightgbm.LGBMRegressor):
            return {
                'num_leaves': trial.suggest_int('num_leaves', 10, 150),
                'min_child_samples': trial.suggest_int('min_child_samples', 1, 1000),
                'min_child_weight': trial.suggest_uniform('min_child_weight', 0, 1),
                'subsample': trial.suggest_uniform('subsample', 0, 1),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0, 1),
                'reg_alpha': trial.suggest_uniform('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_uniform('reg_lambda', 0, 1),
                'n_jobs': [mp.cpu_count() - 1],
            }
        elif isinstance(self.model, catboost.CatBoostClassifier):
            return {
                'loss_function': trial.suggest_categorical('loss_function', ['Logloss' if self.y.nunique() == 2 else
                                                                             'MultiClass']),
                'learning_rate': trial.suggest_uniform('learning_rate', 0, 1),
                'l2_leaf_reg': trial.suggest_uniform('l2_leaf_reg', 0, 10),
                'depth': trial.suggest_int('depth', 1, 10),
                'min_data_in_leaf': trial.suggest_randint('min_data_in_leaf', 50, 500),
                'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
            }
        elif isinstance(self.model, ensemble.GradientBoostingClassifier):
            return {
                'loss': trial.suggest_categorical('loss', ['deviance', 'exponential']),
                'learning_rate': trial.suggest_uniform('learning_rate', 0, 1),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
            }
        elif isinstance(self.model, ensemble.HistGradientBoostingClassifier):
            return {
                'max_iter': trial.suggest_int('max_iter', 100, 250),
                'max_bins': trial.suggest_int('max_bins', 100, 255),
                'l2_regularization': trial.suggest_uniform('l2_regularization', 0, 10),
                'learning_rate': trial.suggest_uniform('learning_Rate', 0, 1),
                'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 30, 150),
                'early_stopping': [True]
            }
        elif isinstance(self.model, ensemble.RandomForestClassifier):
            return {
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt']),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 1000),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            }
        elif isinstance(self.model, xgboost.XGBClassifier):
            return {
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
                'learning_rate': trial.suggest_uniform('learning_rate', 0, 10),
                'verbosity': [0],
                'n_jobs': [mp.cpu_count() - 1],
                'scale_pos_weight': trial.suggest_uniform('scale_pos_weight', 0, 100)
            }
        elif isinstance(self.model, lightgbm.LGBMClassifier):
            return {
                'num_leaves': trial.suggest_int('num_leaves', 10, 150),
                'min_child_samples': trial.suggest_int('min_child_samples', 1, 1000),
                'min_child_weight': trial.suggest_uniform('min_child_weight', 0, 1),
                'subsample': trial.suggest_uniform('subsample', 0, 1),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0, 1),
                'reg_alpha': trial.suggest_uniform('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_uniform('reg_lambda', 0, 1),
                'n_jobs': [mp.cpu_count() - 1],
            }

        # Raise error if nothing is returned
        raise NotImplementedError('Hyper parameter tuning not implemented for ', type(self.model).__name__)

    def get_callbacks(self):
        pass

    def fit(self, x, y):
        # Store
        self.x, self.y = x, y
        # Set up study
        study = optuna.create_study(sampler=optuna.samplers.TPESampler())
        study.optimize(self.objective, timeout=self.timeout, n_trials=self.nTrials, show_progress_bar=True)

        # Parse results
        optuna_results = study.trials_dataframe()
        results = pd.DataFrame({
            'worst_case': optuna_results['value'] - optuna_results['user_attrs_std_value'],
            'mean_objective': optuna_results['value'],
            'std_objective': optuna_results['user_attrs_std_value'],
            'params': [x.params for x in study.get_trials()],
            'mean_time': optuna_results['user_attrs_mean_time'],
            'std_time': optuna_results['user_attrs_std_time']
        })
        return results

    def objective(self, trial):
        # Metrics
        scores = []
        times = []

        # Cross Validation
        for t, v in self.cv.split(self.x, self.y):
            # Split data
            xt, xv, yt, yv = self.x.iloc[t], self.x.iloc[v], self.y.iloc[t], self.y.iloc[v]

            # Train model
            t_start = time.time()
            model = copy.copy(self.model)
            model.set_params(**self.get_params(trial))
            model.fit(xt, yt)

            # Results
            scores.append(self.scoring(model, xv, yv))
            times.append(time.time() - t_start)

        trial.set_user_attr('mean_time', np.mean(times))
        trial.set_user_attr('std_time', np.std(times))
        trial.set_user_attr('std_value', np.std(scores))
        return np.mean(scores)
