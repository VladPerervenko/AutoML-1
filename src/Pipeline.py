import os
import re
import json
import copy
import math
import shutil
import pickle
import joblib
import warnings
import textwrap
import numpy as np
import pandas as pd
import multiprocessing as mp
from scipy.stats import uniform
from scipy.stats import randint
import matplotlib.pyplot as plt

import sklearn
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from . import Utils
from .AutoML.Sequence import Sequence
from .AutoML.Modelling import Modelling
from .AutoML.DataExploring import DataExploring
from .AutoML.DataProcessing import DataProcessing
from .AutoML.FeatureProcessing import FeatureProcessing
from .GridSearch.BaseGridSearch import BaseGridSearch
from .GridSearch.HalvingGridSearch import HalvingGridSearch
from .GridSearch.OptunaGridSearch import OptunaGridSearch


class Pipeline:

    def __init__(self,
                 target,
                 project='',
                 version=None,
                 mode='regression',
                 objective=None,
                 fast_run=False,

                 # Data Processing
                 num_cols=None,
                 date_cols=None,
                 cat_cols=None,
                 missing_values='interpolate',
                 outlier_removal='clip',
                 z_score_threshold=4,
                 include_output=False,

                 # Feature Processing
                 max_lags=10,
                 max_diff=2,
                 information_threshold=0.99,
                 extract_features=True,

                 # Sequencing
                 sequence=False,
                 back=0,
                 forward=0,
                 shift=0,
                 diff='none',

                 # Initial Modelling
                 normalize=True,
                 shuffle=False,
                 cv_splits=3,
                 store_models=True,

                 # Grid Search
                 grid_search_type='optuna',
                 grid_search_iterations=3,

                 # Production
                 custom_code='',

                 # Flags
                 plot_eda=None,
                 process_data=None,
                 validate_result=None,
                 verbose=1):
        # Starting
        print('\n\n*** Starting Amplo AutoML - %s ***\n\n' % project)

        # Parsing input
        if len(project) == 0:
            self.mainDir = 'AutoML/'
        else:
            self.mainDir = project if project[-1] == '/' else project + '/'
        self.target = re.sub('[^a-z0-9]', '_', target.lower())
        self.verbose = verbose
        self.customCode = custom_code
        self.fastRun = fast_run

        # Checks
        assert mode == 'regression' or mode == 'classification', 'Supported modes: regression, classification.'
        assert isinstance(target, str), 'Target needs to be of type string, key of target'
        assert isinstance(project, str), 'Project is a name, needs to be of type string'
        assert isinstance(num_cols, list), 'Num cols must be a list of strings'
        assert isinstance(date_cols, list), 'Date cols must be a list of strings'
        assert isinstance(cat_cols, list), 'Cat Cols must be a list of strings'
        assert isinstance(shift, int), 'Shift needs to be an integer'
        assert isinstance(max_diff, int), 'max_diff needs to be an integer'
        assert max_lags < 50, 'Max_lags too big. Max 50.'
        assert 0 < information_threshold < 1, 'Information threshold needs to be within [0, 1'
        assert max_diff < 5, 'Max difftoo big. Max 5.'

        # Objective
        if objective is not None:
            self.objective = objective
        else:
            if mode == 'regression:':
                self.objective = 'neg_mean_squared_error'
            elif mode == 'classification':
                self.objective = 'accuracy'
        assert isinstance(objective, str), 'Objective needs to be a string.'
        assert objective in metrics.SCORERS.keys(), 'Metric not supported, look at sklearn.metrics.SCORERS.keys()'
        self.scorer = metrics.SCORERS[self.objective]

        # Params needed
        self.mode = mode
        self.version = version
        self.numCols = num_cols
        self.dateCols = date_cols
        self.catCols = cat_cols
        self.missingValues = missing_values
        self.outlierRemoval = outlier_removal
        self.zScoreThreshold = z_score_threshold
        self.includeOutput = include_output
        self.sequence = sequence
        self.sequenceBack = back
        self.sequenceForward = forward
        self.sequenceShift = shift
        self.sequenceDiff = diff
        self.normalize = normalize
        self.shuffle = shuffle
        self.cvSplits = cv_splits
        self.gridSearchType = grid_search_type
        self.gridSearchIterations = grid_search_iterations
        self.plotEDA = plot_eda
        self.processData = process_data
        self.validateResults = validate_result

        # Instance initiating
        self.X = None
        self.Y = None
        self.colKeep = None
        self.results = None

        # Flags
        self._set_flags()

        # Create Directories
        self._create_dirs()

        # Load Version
        self._load_version()

        # Sub- Classes
        self.DataProcessing = DataProcessing(target=self.target, num_cols=num_cols, date_cols=date_cols,
                                             cat_cols=cat_cols, missing_values=missing_values, mode=mode,
                                             outlier_removal=outlier_removal, z_score_threshold=z_score_threshold,
                                             folder=self.mainDir + 'Data/', version=self.version)
        self.FeatureProcessing = FeatureProcessing(max_lags=max_lags, max_diff=max_diff,
                                                   extract_features=extract_features, mode=mode,
                                                   information_threshold=information_threshold,
                                                   folder=self.mainDir + 'Features/', version=self.version)
        self.Sequence = Sequence(back=back, forward=forward, shift=shift, diff=diff)
        self.Modelling = Modelling(mode=mode, shuffle=shuffle, store_models=store_models,
                                   scoring=metrics.SCORERS[self.objective],
                                   store_results=False, folder=self.mainDir + 'Models/')

        # Store production
        self.bestModel = None
        self.bestFeatures = None
        self.bestScaler = None
        self.bestOScaler = None

    def _set_flags(self):
        if self.plotEDA is None:
            self.plotEDA = Utils.boolean_input('Make all EDA graphs?')
        if self.processData is None:
            self.processData = Utils.boolean_input('Process/prepare data?')
        if self.validateResults is None:
            self.validateResults = Utils.boolean_input('Validate results?')

    def _load_version(self):
        if self.version is None:
            versions = os.listdir(self.mainDir + 'Production')
            # Updates changelog
            if self.processData:
                if len(versions) == 0:
                    if self.verbose > 0:
                        print('[AutoML] No Production files found. Setting version 0.')
                    self.version = 0
                    file = open(self.mainDir + 'changelog.txt', 'w')
                    file.write('Dataset changelog. \nv0: Initial')
                    file.close()
                else:
                    self.version = len((versions))

                    # Check if not already started
                    with open(self.mainDir + 'changelog.txt', 'r') as f:
                        changelog = f.read()

                    # Else ask for changelog
                    if 'v%i' % self.version in changelog:
                        changelog = changelog[changelog.find('v%i' % self.version):]
                        changelog = changelog[:max(0, changelog.find('\n'))]
                    else:
                        changelog = '\nv%i: ' % self.version + input("Data changelog v%i:\n" % self.version)
                        file = open(self.mainDir + 'changelog.txt', 'a')
                        file.write(changelog)
                        file.close()
                    if self.verbose > 0:
                        print('[AutoML] Set version %s' % (changelog[1:]))
            else:
                if len(versions) == 0:
                    if self.verbose > 0:
                        print('[AutoML] No Production files found. Setting version 0.')
                    self.version = 0
                else:
                    self.version = int(len(versions)) - 1
                    with open(self.mainDir + 'changelog.txt', 'r') as f:
                        changelog = f.read()
                    changelog = changelog[changelog.find('v%i' % self.version):]
                    if self.verbose > 0:
                        print('[AutoML] Loading last version (%s)' % changelog[:changelog.find('\n')])

    def _create_dirs(self):
        dirs = ['', 'Data', 'Features', 'Models', 'Production', 'Validation', 'Sets']
        for dir in dirs:
            try:
                os.makedirs(self.mainDir + dir)
            except:
                continue

    @staticmethod
    def _sort_results(results):
        results['worst_case'] = results['mean_objective'] - results['std_objective']
        return results.sort_values('worst_case', ascending=False)

    def _get_best_params(self, model, feature_set):
        # Filter results for model and version
        results = self.results[np.logical_and(
            self.results['model'] == type(model).__name__,
            self.results['data_version'] == self.version,
        )]

        # Filter results for feature set & sort them
        results = self._sort_results(results[results['dataset'] == feature_set])

        # Warning for unoptimized results
        if 'Hyper Parameter' not in results['type'].values:
            warnings.warn('Hyperparameters not optimized for this combination')

        # Parse & return best parameters (regardless of if it's optimized)
        return Utils.parse_json(results.iloc[0]['params'])

    def fit(self, data):
        '''
        Fit the full autoML pipeline.
        2. (optional) Exploratory Data Analysis
        Creates a ton of plots which are helpful to improve predictions manually
        3. Data Processing
        Cleans all the data. See @DataProcessing
        4. Feature Processing
        Extracts & Selects. See @FeatureProcessing
        5. Initial Modelling
        Runs 12 off the shelf models with default parameters for all feature sets
        If Sequencing is enabled, this is where it happens, as here, the feature set is generated.
        6. Grid Search
        Optimizes the hyperparameters of the best performing models
        7. Prepare Production Files
        Nicely organises all required scripts / files to make a prediction

        @param data: DataFrame including target
        '''
        # Execute pipeline
        self._data_processing(data)
        self._eda()
        self._feature_processing()
        self._initial_modelling()
        self.grid_search()
        # Production Env
        if not os.path.exists(self.mainDir + 'Production/v%i/' % self.version) or \
                len(os.listdir(self.mainDir + 'Production/v%i/' % self.version)) == 0:
            self._prepare_production_files()
        print('[AutoML] Done :)')

    def _eda(self):
        if self.plotEDA:
            print('[AutoML] Starting Exploratory Data Analysis')
            self.eda = DataExploring(self.X, Y=self.Y, folder=self.mainDir, version=self.version)

    def _data_processing(self, data):
        # Load if possible
        if os.path.exists(self.mainDir + 'Data/Cleaned_v%i.csv' % self.version):
            print('[AutoML] Loading Cleaned Data')
            data = pd.read_csv(self.mainDir + 'Data/Cleaned_v%i.csv' % self.version, index_col='index')

        # Clean
        else:
            print('[AutoML] Cleaning Data')
            data = self.DataProcessing.clean(data)

        # Split and store in memory
        self.Y = data[[self.target]]
        self.X = data
        if self.includeOutput is False:
            self.X = self.X.drop(self.target, axis=1)

        # Assert classes in case of classification
        if self.mode == 'classification':
            if self.Y[self.target].nunique() >= 50:
                warnings.warn('More than 50 classes, you might want to reconsider')
            if set(self.Y[self.target]) != set([i for i in range(len(set(self.Y[self.target])))]):
                warnings.warn('Classes should be [0, 1, ...]')

    def _feature_processing(self):
        # Extract
        self.X = self.FeatureProcessing.extract(self.X, self.Y[self.target])

        # Select
        self.colKeep = self.FeatureProcessing.select(self.X, self.Y[self.target])

    def _initial_modelling(self):
        # Load existing results
        if 'Results.csv' in os.listdir(self.mainDir):
            self.results = pd.read_csv(self.mainDir + 'Results.csv')
            self.Modelling.samples = len(self.Y)

        # Check if this version has been modelled
        if self.results is not None and \
                (self.version == self.results['data_version']).any():
            self.results = self._sort_results(self.results)

        # Run Modelling
        else:
            for feature_set, cols in self.colKeep.items():
                # Skip empty sets
                if len(cols) == 0:
                    print('[AutoML] Skipping %s features, empty set' % feature_set)
                else:
                    print('[AutoML] Initial Modelling for %s features (%i)' % (feature_set, len(cols)))

                    # Apply Feature Set
                    self.Modelling.dataset = feature_set
                    # X, Y = self.X.reindex(columns=cols), self.Y.loc[:, self.target]
                    X, Y = self.X[cols], self.Y

                    # Normalize Feature Set (Done here to get one normalization file per feature set)
                    if self.normalize:
                        normalizeFeatures = [k for k in X.keys() if k not in self.dateCols + self.catCols]
                        scaler = StandardScaler()
                        X[normalizeFeatures] = scaler.fit_transform(X[normalizeFeatures])
                        pickle.dump(scaler,
                                    open(self.mainDir + 'Features/Scaler_%s_%i.pickle' % (feature_set, self.version),
                                         'wb'))
                        if self.mode == 'regression':
                            oScaler = StandardScaler()
                            Y[self.target] = oScaler.fit_transform(Y)
                            pickle.dump(oScaler,
                                        open(self.mainDir + 'Features/OScaler_%s_%i.pickle' % (
                                            feature_set, self.version),
                                             'wb'))

                    # Sequence if necessary
                    if self.sequence:
                        X, Y = self.Sequence.convert(X, Y)

                    # Do the modelling
                    results = self.Modelling.fit(X, Y)

                    # Add results to memory
                    results['type'] = 'Initial modelling'
                    results['data_version'] = self.version
                    if self.results is None:
                        self.results = results
                    else:
                        self.results = self.results.append(results)

            # Save results
            self.results.to_csv(self.mainDir + 'Results.csv', index=False)

    def _get_hyper_params(self, model):
        # todo Integrate optuna, baseGridSearch
        # todo parameterize
        # todo change if statements all to model.__module__
        # Parameters for both Regression / Classification
        if isinstance(model, sklearn.linear_model.Lasso) or \
                isinstance(model, sklearn.linear_model.Ridge) or \
                isinstance(model, sklearn.linear_model.RidgeClassifier):
            return {
                'alpha': uniform(0, 10),
            }
        elif isinstance(model, sklearn.svm.SVC) or \
                isinstance(model, sklearn.svm.SVR):
            return {
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 0.5, 1],
                'C': uniform(0, 10),
            }
        elif isinstance(model, sklearn.neighbors.KNeighborsRegressor) or \
                isinstance(model, sklearn.neighbors.KNeighborsClassifier):
            return {
                'n_neighbors': randint(5, 50),
                'weights': ['uniform', 'distance'],
                'leaf_size': randint(10, 150),
                'n_jobs': [mp.cpu_count() - 1],
            }
        elif isinstance(model, sklearn.neural_network._multilayer_perceptron.MLPClassifier) or \
                isinstance(model, sklearn.neural_network._multilayer_perceptron.MLPRegressor):
            return {
                'hidden_layer_sizes': [(100,), (100, 100), (100, 50), (200, 200), (200, 100), (200, 50),
                                       (50, 50, 50, 50)],
                'learning_rate': ['adaptive', 'invscaling'],
                'alpha': [1e-4, 1e-3, 1e-5],
                'shuffle': [False],
            }

        # Regressor specific hyperparameters
        elif self.mode == 'regression':
            if isinstance(model, sklearn.linear_model.SGDRegressor):
                return {
                    'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
                    'penalty': ['l2', 'l1', 'elasticnet'],
                    'alpha': randint(0, 5),
                }
            elif isinstance(model, sklearn.tree.DecisionTreeRegressor):
                return {
                    'criterion': ['mse', 'friedman_mse', 'mae', 'poisson'],
                    'max_depth': randint(5, 50),
                }
            elif isinstance(model, sklearn.ensemble.AdaBoostRegressor):
                return {
                    'n_estimators': randint(25, 250),
                    'loss': ['linear', 'square', 'exponential'],
                    'learning_rate': uniform(0, 1)
                }
            elif isinstance(model, sklearn.ensemble.GradientBoostingRegressor):
                return {
                    'loss': ['ls', 'lad', 'huber'],
                    'learning_rate': uniform(0, 1),
                    'max_depth': randint(3, 15),
                }
            elif isinstance(model, sklearn.ensemble.HistGradientBoostingRegressor):
                return {
                    'max_iter': randint(100, 250),
                    'max_bins': randint(100, 255),
                    'loss': ['least_squares', 'least_absolute_deviation'],
                    'l2_regularization': uniform(0, 10),
                    'learning_rate': uniform(0, 1),
                    'max_leaf_nodes': randint(30, 150),
                    'early_stopping': [True],
                }
            elif isinstance(model, sklearn.ensemble.RandomForestRegressor):
                return {
                    'criterion': ['mse', 'mae'],
                    'max_depth': randint(3, 15),
                    'max_features': ['auto', 'sqrt'],
                    'min_samples_split': randint(2, 50),
                    'min_samples_leaf': randint(1, 1000),
                    'bootstrap': [True, False],
                }
            elif model.__module__ == 'catboost.core':
                return {
                    'loss_function': ['MAE', 'RMSE'],
                    'learning_rate': uniform(0, 1),
                    'l2_leaf_reg': uniform(0, 10),
                    'depth': randint(3, 15),
                    'min_data_in_leaf': randint(1, 1000),
                    'max_leaves': randint(10, 250),
                }
            elif model.__module__ == 'xgboost.sklearn':
                return {
                    'max_depth': randint(3, 15),
                    'booster': ['gbtree', 'gblinear', 'dart'],
                    'learning_rate': uniform(0, 10),
                    'verbosity': [0],
                    'n_jobs': [mp.cpu_count() - 1],
                }
            elif model.__module__ == 'lightgbm.sklearn':
                return {
                    'num_leaves': randint(10, 150),
                    'min_child_samples': randint(1, 1000),
                    'min_child_weight': uniform(0, 1),
                    'subsample': uniform(0, 1),
                    'colsample_bytree': uniform(0, 1),
                    'reg_alpha': uniform(0, 1),
                    'reg_lambda': uniform(0, 1),
                    'n_jobs': [mp.cpu_count() - 1],
                }

        # Classification specific hyperparameters
        elif self.mode == 'classification':
            if isinstance(model, sklearn.linear_model.SGDClassifier):
                return {
                    'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],
                    'penalty': ['l2', 'l1', 'elasticnet'],
                    'alpha': uniform(0, 10),
                    'max_iter': randint(250, 2000),
                }
            elif isinstance(model, sklearn.tree.DecisionTreeClassifier):
                return {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': randint(5, 50),
                }
            elif isinstance(model, sklearn.ensemble.AdaBoostClassifier):
                return {
                    'n_estimators': randint(25, 250),
                    'learning_rate': uniform(0, 1)
                }
            elif isinstance(model, sklearn.ensemble.BaggingClassifier):
                return {
                    # 'n_estimators': [5, 10, 15, 25, 50],
                    'max_features': uniform(0, 1),
                    'bootstrap': [False, True],
                    'bootstrap_features': [True, False],
                    'n_jobs': [mp.cpu_count() - 1],
                }
            elif isinstance(model, sklearn.ensemble.GradientBoostingClassifier):
                return {
                    'loss': ['deviance', 'exponential'],
                    'learning_rate': uniform(0, 1),
                    'max_depth': randint(3, 15),
                }
            elif isinstance(model, sklearn.ensemble.HistGradientBoostingClassifier):
                return {
                    'max_iter': randint(100, 250),
                    'max_bins': randint(100, 255),
                    'l2_regularization': uniform(0, 10),
                    'learning_rate': uniform(0, 1),
                    'max_leaf_nodes': randint(30, 150),
                    'early_stopping': [True]
                }
            elif isinstance(model, sklearn.ensemble.RandomForestClassifier):
                return {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': randint(3, 15),
                    'max_features': ['auto', 'sqrt'],
                    'min_samples_split': randint(2, 50),
                    'min_samples_leaf': randint(1, 1000),
                    'bootstrap': [True, False],
                }
            elif model.__module__ == 'catboost.core':
                return {
                    'loss_function': ['Logloss' if self.Y[self.target].nunique() == 2 else 'MultiClass'],
                    'learning_rate': uniform(0, 1),
                    'l2_leaf_reg': uniform(0, 10),
                    'depth': randint(1, 10),
                    'min_data_in_leaf': randint(50, 500),
                    'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],
                }
            elif model.__module__ == 'xgboost.sklearn':
                return {
                    'max_depth': randint(3, 15),
                    'booster': ['gbtree', 'gblinear', 'dart'],
                    'learning_rate': uniform(0, 10),
                    'verbosity': [0],
                    'n_jobs': [mp.cpu_count() - 1],
                    'scale_pos_weight': uniform(0, 100)
                }
            elif model.__module__ == 'lightgbm.sklearn':
                return {
                    'num_leaves': randint(10, 150),
                    'min_child_samples': randint(1, 1000),
                    'min_child_weight': uniform(0, 1),
                    'subsample': uniform(0, 1),
                    'colsample_bytree': uniform(0, 1),
                    'reg_alpha': uniform(0, 1),
                    'reg_lambda': uniform(0, 1),
                    'n_jobs': [mp.cpu_count() - 1],
                }

        # Raise error if nothing is returned
        raise NotImplementedError('Hyperparameter tuning not implemented for ', type(model).__name__)

    def grid_search(self, model=None, feature_set=None, parameter_set=None):
        """
        Runs a grid search. By default, takes the self.results, and runs for the top 3 optimizations.
        There is the option to provide a model & feature_set, but both have to be provided. In this case,
        the model & data set combination will be optimized.
        Implemented types, Base, Halving, Optuna
        """
        assert model is not None and feature_set is not None or model == feature_set, \
            'Model & feature_set need to be either both None or both provided.'

        # If arguments are provided
        if model is not None:

            # Get model string
            if isinstance(model, str):
                models = self.Modelling.return_models()
                model = models[[i for i in range(len(models)) if type(models[i]).__name__ == model][0]]

            # Organise existing results
            results = self.results[np.logical_and(
                self.results['model'] == type(model).__name__,
                self.results['data_version'] == self.version,
            )]
            results = self._sort_results(results[results['dataset'] == feature_set])

            # Check if exists and load
            if ('Hyper Parameter' == results['type']).any():
                print('[AutoML] Loading optimization results.')
                hyper_opt_results = results[results['type'] == 'Hyper Parameter']
                params = Utils.parse_json(hyper_opt_results.iloc[0]['params'])

            # Or run
            else:
                # Parameter check
                if parameter_set is None:
                    parameter_set = self._get_hyper_params(model)
                    
                # Run grid search
                grid_search_results = self._sort_results(self._grid_search_iteration(model, parameter_set, feature_set))

                # Store results
                grid_search_results['model'] = type(model).__name__
                grid_search_results['data_version'] = self.version
                grid_search_results['dataset'] = feature_set 
                grid_search_results['type'] = 'Hyper Parameter'
                self.results = self.results.append(grid_search_results)
                self.results.to_csv(self.mainDir + 'Results.csv', index=False)

                # Get params for validation
                params = Utils.parse_json(grid_search_results.iloc[0]['params'])

            # Validate
            self._validate_result(model, params, feature_set)
            return

        # If arguments aren't provided, run through promising models
        models = self.Modelling.return_models()
        results = self._sort_results(self.results[np.logical_and(
            self.results['type'] == 'Initial modelling',
            self.results['data_version'] == self.version,
        )])
        for iteration in range(self.gridSearchIterations):
            # Grab settings
            settings = results.iloc[iteration]
            model = models[[i for i in range(len(models)) if type(models[i]).__name__ == settings['model']][0]]
            feature_set = settings['dataset']

            # Check whether exists
            model_results = self.results[np.logical_and(
                self.results['model'] == type(model).__name__,
                self.results['data_version'] == self.version,
            )]
            model_results = self._sort_results(model_results[model_results['dataset'] == feature_set])

            # If exists
            if ('Hyper Parameter' == model_results['type']).any():
                hyper_opt_res = model_results[model_results['type'] == 'Hyper Parameter']
                params = Utils.parse_json(hyper_opt_res.iloc[0]['params'])

            # Else run
            else:
                # Get params
                params = self._get_hyper_params(model)

                grid_search_results = self._sort_results(self._grid_search_iteration(model, params, feature_set))

                # Store
                grid_search_results['model'] = type(model).__name__
                grid_search_results['data_version'] = self.version
                grid_search_results['dataset'] = feature_set
                grid_search_results['type'] = 'Hyper Parameter'
                self.results = self.results.append(grid_search_results)
                self.results.to_csv(self.mainDir + 'Results.csv', index=False)
                params = Utils.parseJson(grid_search_results.iloc[0]['params'])

            # Validate
            if self.validateResults:
                self._validate_result(model, params, feature_set)

    def _grid_search_iteration(self, model, params, feature_set):
        """
        INTERNAL | Grid search for defined model, parameter set and feature set.
        """
        print('\n[AutoML] Starting Hyper Parameter Optimization for %s on %s features (%i samples, %i features)' %
              (type(model).__name__, feature_set, len(self.X), len(self.colKeep[feature_set])))

        # Select data
        x, y = self.X[self.colKeep[feature_set]], self.Y

        # Normalize Feature Set (the input remains original)
        if self.normalize:
            features_to_normalize = [k for k in self.colKeep[feature_set] if k not in self.dateCols + self.catCols]
            scaler = pickle.load(
                open(self.mainDir + 'Features/Scaler_%s_%i.pickle' % (feature_set, self.version), 'rb'))
            x[features_to_normalize] = scaler.transform(x[features_to_normalize])
            if self.mode == 'regression':
                output_scaler = pickle.load(
                    open(self.mainDir + 'Features/OScaler_%s_%i.pickle' % (feature_set, self.version), 'rb'))
                y = output_scaler.transform(y)

        # Cross-Validator
        if self.mode == 'regression':
            cv = KFold(n_splits=self.cvSplits, shuffle=self.shuffle)
        elif self.mode == 'classification':
            cv = StratifiedKFold(n_splits=self.cvSplits, shuffle=self.shuffle)

        # Select right hyperparameter optimizer
        if self.gridSearchType == 'Base':
            grid_search = BaseGridSearch(model, params=params, cv=cv, scoring=self.scorer, verbose=self.verbose)
        elif self.gridSearchType == 'Halving':
            grid_search = HalvingGridSearch(model, params=params, cv=cv, scoring=self.scorer, verbose=self.verbose)
        elif self.gridSearchType == 'Optuna':
            grid_search = OptunaGridSearch(model, params=params, cv=cv, scoring=self.scorer, verbose=self.verbose)
        else:
            raise NotImplementedError('Only Base, Halving and Optuna are implemented.')
        # Get results
        results = grid_search.fit(x, y)
        results['worst_case'] = results['mean_objective'] - results['std_objective']
        return results.sort_values('worst_case', ascending=False)

    def _create_stacking(self):
        # todo implement
        '''
        Based on the best performing models, in addition to cheap models based on very different assumptions,
        A stacking model is optimized to enhance/combine the performance of the models.
        '''
        # First, the ideal dataset has to be chosen, we're restricted to a single one...
        results = self._sort_results(self.results[np.logical_and(
            self.results['type'] == 'Hyper Parameter',
            self.results['data_version'] == self.version,
        )])
        feature_set = results['dataset'].value_counts()[0]

    def validate(self, model, feature_set, params=None):
        '''
        Just a wrapper for the outside.
        Parameters:
        Model: The model to optimize, either string or class
        Feature Set: String
        (optional) params: Model parameters for which to validate
        '''
        assert feature_set in self.colKeep.keys(), 'Feature Set not available.'

        # Get model
        if isinstance(model, str):
            models = self.Modelling.return_models()
            model = models[[i for i in range(len(models)) if type(models[i]).__name__ == model][0]]

        # Get params
        if params is not None:
            params = self._getBestParams(model, feature_set)

        # Run validation
        self._validateResult(model, params, feature_set)

    def _validate_result(self, master_model, params, feature_set):
        print('[AutoML] Validating results for %s (%i %s features) (%s)' % (type(master_model).__name__,
                                                                            len(self.colKeep[feature_set]), feature_set,
                                                                            params))
        if not os.path.exists(self.mainDir + 'Validation/'): os.mkdir(self.mainDir + 'Validation/')

        # Select data
        X, Y = self.X[self.colKeep[feature_set]], self.Y

        # Normalize Feature Set (the X remains original)
        if self.normalize:
            normalizeFeatures = [k for k in self.colKeep[feature_set] if k not in self.dateCols + self.catCols]
            scaler = pickle.load(
                open(self.mainDir + 'Features/Scaler_%s_%i.pickle' % (feature_set, self.version), 'rb'))
            X[normalizeFeatures] = scaler.transform(X[normalizeFeatures])
            if self.mode == 'regression':
                oScaler = pickle.load(
                    open(self.mainDir + 'Features/OScaler_%s_%i.pickle' % (feature_set, self.version), 'rb'))
                Y[Y.keys()] = oScaler.transform(Y)
                print('(%.1f, %.1f)' % (np.mean(Y), np.std(Y)))
        X, Y = X.to_numpy(), Y.to_numpy().reshape((-1, 1))

        # For Regression
        if self.mode == 'regression':

            # Cross-Validation Plots
            fig, ax = plt.subplots(math.ceil(self.cvSplits / 2), 2, sharex=True, sharey=True)
            fig.suptitle('%i-Fold Cross Validated Predictions - %s' % (self.cvSplits, type(master_model).__name__))

            # Initialize iterables
            score = []
            cv = KFold(n_splits=self.cvSplits, shuffle=self.shuffle)
            # Cross Validate
            for i, (t, v) in enumerate(cv.split(X, Y)):
                Xt, Xv, Yt, Yv = X[t], X[v], Y[t].reshape((-1)), Y[v].reshape((-1))
                model = copy.copy(master_model)
                model.set_params(**params)
                model.fit(Xt, Yt)

                # Metrics
                score.append(self.scorer(model, Xv, Yv))

                # Plot
                ax[i // 2][i % 2].set_title('Fold-%i' % i)
                ax[i // 2][i % 2].plot(Yv, color='#2369ec')
                ax[i // 2][i % 2].plot(model.predict(Xv), color='#ffa62b', alpha=0.4)

            # Print & Finish plot
            print('[AutoML] %s:        %.2f \u00B1 %.2f' % (self.scorer._score_func.__name__,
                                                            np.mean(score), np.std(score)))
            ax[i // 2][i % 2].legend(['Output', 'Prediction'])
            plt.show()

        # For BINARY classification
        elif self.mode == 'classification' and self.Y[self.target].nunique() == 2:
            # Initiating
            fig, ax = plt.subplots(math.ceil(self.cvSplits / 2), 2, sharex=True, sharey=True)
            fig.suptitle('%i-Fold Cross Validated Predictions - %s (%s)' %
                         (self.cvSplits, type(master_model).__name__, feature_set))
            acc = []
            prec = []
            rec = []
            spec = []
            f1 = []
            aucs = []
            tprs = []
            cm = np.zeros((2, 2))
            mean_fpr = np.linspace(0, 1, 100)

            # Modelling
            cv = StratifiedKFold(n_splits=self.cvSplits)
            for i, (t, v) in enumerate(cv.split(X, Y)):
                n = len(v)
                Xt, Xv, Yt, Yv = X[t], X[v], Y[t].reshape((-1)), Y[v].reshape((-1))
                model = copy.copy(master_model)
                model.set_params(**params)
                model.fit(Xt, Yt)
                predictions = model.predict(Xv).reshape((-1))

                # Metrics
                tp = np.logical_and(np.sign(predictions) == 1, Yv == 1).sum()
                tn = np.logical_and(np.sign(predictions) == 0, Yv == 0).sum()
                fp = np.logical_and(np.sign(predictions) == 1, Yv == 0).sum()
                fn = np.logical_and(np.sign(predictions) == 0, Yv == 1).sum()
                acc.append((tp + tn) / n * 100)
                if tp + fp > 0:
                    prec.append(tp / (tp + fp) * 100)
                if tp + fn > 0:
                    rec.append(tp / (tp + fn) * 100)
                if tn + fp > 0:
                    spec.append(tn / (tn + fp) * 100)
                if tp + fp > 0 and tp + fn > 0:
                    f1.append(2 * prec[-1] * rec[-1] / (prec[-1] + rec[-1]) if prec[-1] + rec[-1] > 0 else 0)
                cm += np.array([[tp, fp], [fn, tn]]) / self.cvSplits

                # Plot
                ax[i // 2][i % 2].plot(Yv, c='#2369ec', alpha=0.6)
                ax[i // 2][i % 2].plot(predictions, c='#ffa62b')
                ax[i // 2][i % 2].set_title('Fold-%i' % i)

            # Results
            print('[AutoML] Accuracy:        %.2f \u00B1 %.2f %%' % (np.mean(acc), np.std(acc)))
            print('[AutoML] Precision:       %.2f \u00B1 %.2f %%' % (np.mean(prec), np.std(prec)))
            print('[AutoML] Recall:          %.2f \u00B1 %.2f %%' % (np.mean(rec), np.std(rec)))
            print('[AutoML] Specificity:     %.2f \u00B1 %.2f %%' % (np.mean(spec), np.std(spec)))
            print('[AutoML] F1-score:        %.2f \u00B1 %.2f %%' % (np.mean(f1), np.std(f1)))
            print('[AutoML] Confusion Matrix:')
            print('[AutoML] Pred \ true |  Faulty   |   Healthy      ')
            print('[AutoML]  Faulty     |  %s|  %.1f' % (('%.1f' % cm[0, 0]).ljust(9), cm[0, 1]))
            print('[AutoML]  Healthy    |  %s|  %.1f' % (('%.1f' % cm[1, 0]).ljust(9), cm[1, 1]))

            # Check whether plot is possible
            if isinstance(model, sklearn.linear_model.Lasso) or isinstance(model, sklearn.linear_model.Ridge):
                return

            # Plot ROC
            fig, ax = plt.subplots(figsize=[12, 8])
            viz = metrics.plot_roc_curve(model, Xv, Yv, name='ROC fold {}'.format(i + 1), alpha=0.3, ax=ax)
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

            # Adjust plots
            ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='#ffa62b',
                    label='Chance', alpha=.8)
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = metrics.auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            ax.plot(mean_fpr, mean_tpr, color='#2369ec',
                    label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                    lw=2, alpha=.8)
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='#729ce9', alpha=.2,
                            label=r'$\pm$ 1 std. dev.')
            ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
                   title="ROC Curve - %s" % type(master_model).__name__)
            ax.legend(loc="lower right")
            fig.savefig(self.mainDir + 'Validation/ROC_%s.png' % type(model).__name__, format='png', dpi=200)

        # For MULTICLASS classification
        elif self.mode == 'classification':
            # Initiating
            fig, ax = plt.subplots(math.ceil(self.cvSplits / 2), 2, sharex=True, sharey=True)
            fig.suptitle('%i-Fold Cross Validated Predictions - %s (%s)' %
                         (self.cvSplits, type(master_model).__name__, feature_set))
            n_classes = self.Y[self.target].nunique()
            f1Score = np.zeros((self.cvSplits, n_classes))
            logLoss = np.zeros(self.cvSplits)
            avgAcc = np.zeros(self.cvSplits)

            # Modelling
            cv = StratifiedKFold(n_splits=self.cvSplits)
            for i, (t, v) in enumerate(cv.split(X, Y)):
                n = len(v)
                Xt, Xv, Yt, Yv = X[t], X[v], Y[t].reshape((-1)), Y[v].reshape((-1))
                model = copy.copy(master_model)
                model.set_params(**params)
                model.fit(Xt, Yt)
                predictions = model.predict(Xv).reshape((-1))

                # Metrics
                f1Score[i] = metrics.f1_score(Yv, predictions, average=None)
                avgAcc[i] = metrics.accuracy_score(Yv, predictions)
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(Xv)
                    logLoss[i] = metrics.log_loss(Yv, probabilities)

                # Plot
                ax[i // 2][i % 2].plot(Yv, c='#2369ec', alpha=0.6)
                ax[i // 2][i % 2].plot(predictions, c='#ffa62b')
                ax[i // 2][i % 2].set_title('Fold-%i' % i)

            # Results
            print('F1 scores:')
            print(''.join([' Class %i |' % i for i in range(n_classes)]))
            print(''.join([' %.2f '.ljust(9) % f1 + '|' for f1 in np.mean(f1Score, axis=0)]))
            print('Average Accuracy: %.2f \u00B1 %.2f' % (np.mean(avgAcc), np.std(avgAcc)))
            if hasattr(model, 'predict_proba'):
                print('Log Loss:         %.2f \u00B1 %.2f' % (np.mean(logLoss), np.std(logLoss)))

    def _prepare_production_files(self, model=None, feature_set=None, params=None):
        if not os.path.exists(self.mainDir + 'Production/v%i/' % self.version):
            os.mkdir(self.mainDir + 'Production/v%i/' % self.version)
        # Get sorted results for this data version
        results = self._sort_results(self.results[self.results['data_version'] == self.version])

        # In the case args are provided
        if model is not None and feature_set is not None:
            # Take name if model instance is given
            if not isinstance(model, str):
                model = type(model).__name__
            if params is None:
                results = self._sortResults(
                    results[np.logical_and(results['model'] == model, results['dataset'] == feature_set)])
                params = Utils.parseJson(results.iloc[0]['params'])

        # Otherwise find best
        else:
            model = results.iloc[0]['model']
            feature_set = results.iloc[0]['dataset']
            params = results.iloc[0]['params']
            if isinstance(params, str):
                params = Utils.parseJson(params)

        # Notify of results
        print('[AutoML] Preparing Production Env Files for %s, feature set %s' %
              (model, feature_set))
        print('[AutoML] ', params)
        print('[AutoML] %s: %.2f \u00B1 %.2f' %
              (self.scorer._score_func.__name__, results.iloc[0]['mean_objective'], results.iloc[0]['std_objective']))

        # Save Features
        self.bestFeatures = self.colKeep[feature_set]
        json.dump(self.bestFeatures, open(self.mainDir + 'Production/v%i/Features.json' % self.version, 'w'))

        # Copy data
        X, Y = self.X[self.bestFeatures], self.Y

        # Save Scalers & Normalize
        if self.normalize:
            # Save
            shutil.copy(self.mainDir + 'Features/Scaler_%s_%i.pickle' % (feature_set, self.version),
                        self.mainDir + 'Production/v%i/Scaler.pickle' % self.version)
            if self.mode == 'regression':
                shutil.copy(self.mainDir + 'Features/OScaler_%s_%i.pickle' % (feature_set, self.version),
                            self.mainDir + 'Production/v%i/OScaler.pickle' % self.version)

            # Normalize
            normalizeFeatures = [k for k in self.bestFeatures if k not in self.dateCols + self.catCols]
            self.bestScaler = pickle.load(
                open(self.mainDir + 'Features/Scaler_%s_%i.pickle' % (feature_set, self.version), 'rb'))
            X = self.bestScaler.transform(X[normalizeFeatures])
            if self.mode == 'regression':
                self.bestOScaler = pickle.load(
                    open(self.mainDir + 'Features/OScaler_%s_%i.pickle' % (feature_set, self.version), 'rb'))
                Y = self.bestOScaler.transform(Y)
            else:
                Y = self.Y[self.target].to_numpy()

        # Cluster Features require additional 'Centers' file
        if any(['dist__' in key for key in self.bestFeatures]):
            shutil.copy(self.mainDir + 'Features/KMeans_v%i.csv' % self.version,
                        self.mainDir + 'Production/v%i/KMeans.csv' % self.version)

        # Model
        self.bestModel = [mod for mod in self.Modelling.return_models() if type(mod).__name__ == model][0]
        self.bestModel.set_params(**params)
        self.bestModel.fit(X, Y.values.ravel())
        joblib.dump(self.bestModel, self.mainDir + 'Production/v%i/Model.joblib' % self.version)

        # Predict function
        predictCode = self.create_predict_function(self.customCode)
        with open(self.mainDir + 'Production/v%i/Predict.py' % self.version, 'w') as f:
            f.write(predictCode)
        with open(self.mainDir + 'Production/v%i/__init__.py' % self.version, 'w') as f:
            f.write('')

        # Pipeline
        pickle.dump(self, open(self.mainDir + 'Production/v%i/Pipeline.pickle' % self.version, 'wb'))
        return

    def _error_analysis(self):
        # todo implement
        pass

    def _convert_data(self, data):
        # Load files
        folder = 'Production/v%i/' % self.version
        features = json.load(open(self.mainDir + folder + 'Features.json', 'r'))

        if self.normalize:
            scaler = pickle.load(open(self.mainDir + folder + 'Scaler.pickle', 'rb'))
            if self.mode == 'regression':
                oScaler = pickle.load(open(self.mainDir + folder + 'OScaler.pickle', 'rb'))
        if self.mode == 'regression':
            oScaler = pickle.load(open(self.mainDir + folder + 'OScaler.pickle', 'rb'))

        # Clean data
        # todo change to transform
        data = Utils.clean_keys(data)
        data = self.DataProcessing._convert_data_types(data)
        data = self.DataProcessing._remove_duplicates(data)
        data = self.DataProcessing._remove_outliers(data)
        data = self.DataProcessing._remove_missing_values(data)
        if data.astype('float32').replace([np.inf, -np.inf], np.nan).isna().sum().sum() != 0:
            raise ValueError('Data should not contain NaN or Infs after cleaning!')

        # Save output
        if self.target in data.keys():
            Y = data[self.target].to_numpy().reshape((-1, 1))
        else:
            Y = None

        # Convert Features
        if 'KMeans.csv' in os.listdir(self.mainDir + folder):
            k_means = pd.read_csv(self.mainDir + folder + 'KMeans.csv')
            X = self.FeatureProcessing.transform(data=data, features=features, k_means=k_means)
        else:
            X = self.FeatureProcessing.transform(data, features)

        if X.astype('float32').replace([np.inf, -np.inf], np.nan).isna().sum().sum() != 0:
            raise ValueError('Data should not contain NaN or Infs after adding features!')

        # Normalize
        if self.normalize:
            X[X.keys()] = scaler.transform(X)

        # Return
        return X, Y

    def predict(self, data):
        '''
        Full script to make predictions. Uses 'Production' folder with defined or latest version.
        '''
        # Feature Extraction, Selection and Normalization
        model = joblib.load(self.mainDir + 'Production/v%i/Model.joblib' % self.version)
        if self.verbose > 0:
            print('[AutoML] Predicting with %s, v%i' % (type(model).__name__, self.version))
        X, Y = self._convert_data(data)

        # Predict
        if self.mode == 'regression':
            if self.normalize:
                oScaler = pickle.load(open('Production/v%i/OScaler.pickle', 'rb'))
                predictions = oScaler.inverse_transform(model.predict(X))
            else:
                predictions = model.predict(X)
        if self.mode == 'classification':
            try:
                predictions = model.predict_proba(X)[:, 1]
            except AttributeError:
                predictions = model.predict(X)

        return predictions

    def create_predict_function(self, custom_code):
        '''
        This function returns a string, which can be used to make predictions.
        This is in a predefined format, a Predict class, with a predict funtion taking the arguments
        model: trained sklearn-like class with the .fit() function
        features: list of strings containing all features fed to the model
        scaler: trained sklearn-like class with .transform function
        data: the data to predict on
        Now this function has the arg decoding, which allows custom code injection
        '''
        # Check if predict file exists already to increment version
        if os.path.exists(self.mainDir + 'Production/v%i/Predict.py' % self.version):
            with open(self.mainDir + 'Production/v%i/Predict.py' % self.version, 'r') as f:
                predictFile = f.read()
            ind = predictFile.find('self.version = ') + 16
            oldVersion = predictFile[ind: predictFile.find("'", ind)]
            minorVersion = int(oldVersion[oldVersion.find('.') + 1:])
            version = oldVersion[:oldVersion.find('.') + 1] + str(minorVersion + 1)
        else:
            version = 'v%i.0' % self.version
        print('Creating Prediction %s' % version)
        dataProcess = self.DataProcessing.exportFunction()
        featureProcess = self.FeatureProcessing.exportFunction()
        return """import pandas as pd
import numpy as np
import struct, re, copy, os


class Predict(object):

    def __init__(self):
        self.version = '{}'

    def predict(self, model, features, data, **args):
        ''' 
        Prediction function for Amplo's AutoML. 
        This is in a predefined format: 
        - a 'Predict' class, with a 'predict' funtion taking the arguments:        
            model: trained sklearn-like class with the .fit() function
            features: list of strings containing all features fed to the model
            data: the data to predict on
        Note: May depend on additional named arguments within args. 
        '''
        ###############
        # Custom Code #
        ###############""".format(version) + textwrap.indent(custom_code, '    ') \
               + dataProcess + featureProcess + '''
        ###########
        # Predict #
        ###########
        mode, normalize = '{}', {}

        # Normalize
        if normalize:
            assert 'scaler' in args.keys(), 'When Normalizing=True, scaler needs to be provided in args'
            X = args['scaler'].transform(X)

        # Predict
        if mode == 'regression':
            if normalize:
                assert 'o_scaler' in args.keys(), 'When Normalizing=True, o_scaler needs to be provided in args'
                predictions = args['oScaler'].inverse_transform(model.predict(X))
            else:
                predictions = model.predict(X)
        if mode == 'classification':
            try:
                predictions = model.predict_proba(X)[:, 1]
            except:
                predictions = model.predict(X)

        return predictions
'''.format(self.mode, self.normalize)
