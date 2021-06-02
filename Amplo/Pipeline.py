import re
import os
import copy
import json
import math
import joblib
import pickle
import shutil
import textwrap
import warnings
import numpy as np
import pandas as pd
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


# noinspection PyUnresolvedReferences
class Pipeline:
    # todo implement data type detector
    # todo remove _get_hyper_parameters from here, fully implement in BaseGridSearch and HalvingGridSearch

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
                 normalize=False,
                 shuffle=False,
                 cv_splits=3,
                 store_models=True,

                 # Grid Search
                 grid_search_type='optuna',
                 optuna_time_budget=3600,
                 grid_search_iterations=3,

                 # Production
                 custom_code='',

                 # Flags
                 plot_eda=None,
                 process_data=None,
                 validate_result=None,
                 verbose=1):
        # Starting
        print('\n\n*** Starting Amplo AutoML - {} ***\n\n'.format(project))

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
        assert isinstance(num_cols, list) or num_cols is None, 'Num cols must be a list of strings'
        assert isinstance(date_cols, list) or date_cols is None, 'Date cols must be a list of strings'
        assert isinstance(cat_cols, list) or cat_cols is None, 'Cat Cols must be a list of strings'
        assert isinstance(shift, int), 'Shift needs to be an integer'
        assert isinstance(max_diff, int), 'max_diff needs to be an integer'
        assert max_lags < 50, 'Max_lags too big. Max 50.'
        assert 0 < information_threshold < 1, 'Information threshold needs to be within [0, 1'
        assert max_diff < 5, 'Max diff too big. Max 5.'
        assert isinstance(grid_search_type, str), 'Grid Search Type must be string'
        assert grid_search_type.lower() in ['base', 'halving', 'optuna'], 'Grid Search Type must be Base, Halving or ' \
                                                                          'Optuna'

        # Objective
        if objective is not None:
            self.objective = objective
        else:
            if mode == 'regression':
                self.objective = 'neg_mean_squared_error'
            elif mode == 'classification':
                self.objective = 'accuracy'
        assert isinstance(objective, str), 'Objective needs to be a string, not type {}'.format(type(objective))
        assert objective in metrics.SCORERS.keys(), 'Metric not supported, look at sklearn.metrics.SCORERS.keys()'
        self.scorer = metrics.SCORERS[self.objective]

        # Pipeline Params
        self.mode = mode
        self.version = version
        self.includeOutput = include_output
        self.plotEDA = plot_eda
        self.processData = process_data
        self.validateResults = validate_result

        # Data Processing params
        self.numCols = [] if num_cols is None else num_cols
        self.dateCols = [] if date_cols is None else date_cols
        self.catCols = [] if cat_cols is None else cat_cols
        self.missingValues = missing_values
        self.outlierRemoval = outlier_removal
        self.zScoreThreshold = z_score_threshold

        # Feature Processing params
        self.extractFeatures = extract_features
        self.maxLags = max_lags
        self.maxDiff = max_diff
        self.informationThreshold = information_threshold

        # Sequence Params
        self.sequence = sequence
        self.sequenceBack = back
        self.sequenceForward = forward
        self.sequenceShift = shift
        self.sequenceDiff = diff

        # Modelling Params
        self.normalize = normalize
        self.shuffle = shuffle
        self.cvSplits = cv_splits
        self.storeModels = store_models

        # Grid Search params
        self.gridSearchType = grid_search_type.lower()
        self.optunaBudget = optuna_time_budget
        self.gridSearchIterations = grid_search_iterations

        # Instance initiating
        self.x = None
        self.y = None
        self.colKeep = None
        self.results = None

        # Flags
        self._set_flags()

        # Create Directories
        self._create_dirs()

        # Load Version
        self._load_version()

        # Required sub-classes
        self.dataProcessor = DataProcessing(target=self.target, num_cols=self.numCols, date_cols=self.dateCols,
                                            cat_cols=self.catCols, missing_values=self.missingValues, mode=self.mode,
                                            outlier_removal=self.outlierRemoval, z_score_threshold=self.zScoreThreshold,
                                            folder=self.mainDir + 'Data/', version=self.version)
        self.featureProcessor = FeatureProcessing(mode=self.mode, max_lags=self.maxLags, max_diff=self.maxDiff,
                                                  extract_features=self.extractFeatures,
                                                  information_threshold=self.informationThreshold,
                                                  folder=self.mainDir + 'Features/', version=self.version)

        # Store production
        self.bestModel = None
        self.bestFeatures = None
        self.bestScaler = None
        self.bestOutputScaler = None

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
                    self.version = len(versions)

                    # Check if not already started
                    with open(self.mainDir + 'changelog.txt', 'r') as f:
                        changelog = f.read()

                    # Else ask for changelog
                    if 'v{}'.format(self.version) in changelog:
                        changelog = changelog[changelog.find('v{}'.format(self.version)):]
                        changelog = changelog[:max(0, changelog.find('\n'))]
                    else:
                        changelog = '\nv{}: '.format(self.version) + input("Data changelog v{}:\n".format(self.version))
                        file = open(self.mainDir + 'changelog.txt', 'a')
                        file.write(changelog)
                        file.close()
                    if self.verbose > 0:
                        print('[AutoML] Set version {}'.format(changelog[1:]))
            else:
                if len(versions) == 0:
                    if self.verbose > 0:
                        print('[AutoML] No Production files found. Setting version 0.')
                    self.version = 0
                else:
                    self.version = int(len(versions)) - 1
                    with open(self.mainDir + 'changelog.txt', 'r') as f:
                        changelog = f.read()
                    changelog = changelog[changelog.find('v{}'.format(self.version)):]
                    if self.verbose > 0:
                        print('[AutoML] Loading last version ({})'.format(changelog[:changelog.find('\n')]))

    def _create_dirs(self):
        folders = ['', 'Data', 'Features', 'Production']
        for folder in folders:
            try:
                os.makedirs(self.mainDir + folder)
            except FileExistsError:
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
            warnings.warn('Hyper parameters not optimized for this combination')

        # Parse & return best parameters (regardless of if it's optimized)
        return Utils.parse_json(results.iloc[0]['params'])

    def fit(self, data):
        """
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
        Optimizes the hyper parameters of the best performing models
        7. Prepare Production Files
        Nicely organises all required scripts / files to make a prediction

        @param data: DataFrame including target
        """
        # Execute pipeline
        self._data_processing(data)
        self._eda()
        self._feature_processing()
        self._initial_modelling()
        self.grid_search()
        if not os.path.exists(self.mainDir + 'Production/v{}/'.format(self.version)) or \
                len(os.listdir(self.mainDir + 'Production/v{}/'.format(self.version))) == 0:
            self._prepare_production_files()
        print('[AutoML] Done :)')

    def _eda(self):
        if self.plotEDA:
            print('[AutoML] Starting Exploratory Data Analysis')
            eda = DataExploring(self.x, y=self.y,
                                mode=self.mode,
                                folder=self.mainDir,
                                version=self.version)
            eda.run()

    def _data_processing(self, data):
        # Load if possible
        if os.path.exists(self.mainDir + 'Data/Cleaned_v{}.csv'.format(self.version)):
            print('[AutoML] Loading Cleaned Data')
            data = pd.read_csv(self.mainDir + 'Data/Cleaned_v{}.csv'.format(self.version), index_col='index')

        # Clean
        else:
            print('[AutoML] Cleaning Data')
            data = self.dataProcessor.clean(data)

        # Split and store in memory
        self.y = data[self.target]
        self.x = data
        if self.includeOutput is False:
            self.x = self.x.drop(self.target, axis=1)

        # Assert classes in case of classification
        if self.mode == 'classification':
            if self.y.nunique() >= 50:
                warnings.warn('More than 50 classes, you might want to reconsider')
            if set(self.y) != set([i for i in range(len(set(self.y)))]):
                warnings.warn('Classes should be [0, 1, ...]')

    def _feature_processing(self):
        # Extract
        self.x = self.featureProcessor.extract(self.x, self.y)

        # Select
        self.colKeep = self.featureProcessor.select(self.x, self.y)

    def _initial_modelling(self):
        # Load existing results
        if 'Results.csv' in os.listdir(self.mainDir):
            self.results = pd.read_csv(self.mainDir + 'Results.csv')

        # Check if this version has been modelled
        if self.results is not None and self.version in self.results['data_version'].values:
            self.results = self._sort_results(self.results)

        # Run Modelling
        else:
            for feature_set, cols in self.colKeep.items():
                # Skip empty sets
                if len(cols) == 0:
                    print('[AutoML] Skipping {} features, empty set'.format(feature_set))
                else:
                    print('[AutoML] Initial Modelling for {} features ({})'.format(feature_set, len(cols)))

                    # Select data
                    x, y = copy.copy(self.x[cols]), copy.copy(self.y)

                    # Normalize Feature Set (Done here to get one normalization file per feature set)
                    if self.normalize:
                        normalize_features = [k for k in x.keys() if k not in self.dateCols + self.catCols]
                        scaler = StandardScaler()
                        x[normalize_features] = scaler.fit_transform(x[normalize_features])
                        pickle.dump(scaler,
                                    open(
                                        self.mainDir + 'Features/Scaler_{}_{}.pickle'.format(feature_set, self.version),
                                        'wb'))
                        if self.mode == 'regression':
                            output_scaler = StandardScaler()
                            y = pd.Series(output_scaler.fit_transform(y.values.reshape(-1, 1)).reshape(-1),
                                          name=self.target)
                            pickle.dump(output_scaler,
                                        open(self.mainDir + 'Features/OutputScaler_{}_{}.pickle'.format(
                                            feature_set, self.version),
                                             'wb'))

                    # Sequence if necessary
                    if self.sequence:
                        sequencer = Sequence(back=self.sequenceBack, forward=self.sequenceForward,
                                             shift=self.sequenceShift, diff=self.sequenceDiff)
                        x, y = sequencer.convert(x, y)

                    # Do the modelling
                    modeller = Modelling(mode=self.mode, shuffle=self.shuffle, store_models=self.storeModels,
                                         scoring=self.objective, dataset=feature_set,
                                         store_results=False, folder=self.mainDir + 'Models/')
                    results = modeller.fit(x, y)

                    # Add results to memory
                    results['type'] = 'Initial modelling'
                    results['data_version'] = self.version
                    if self.results is None:
                        self.results = results
                    else:
                        self.results = self.results.append(results)

            # Save results
            self.results.to_csv(self.mainDir + 'Results.csv', index=False)

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
                models = Modelling(mode=self.mode, samples=len(self.x), scoring=self.objective).return_models()
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
        models = Modelling(mode=self.mode, samples=len(self.x), scoring=self.objective).return_models()
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
                # For one model
                grid_search_results = self._sort_results(self._grid_search_iteration(model, parameter_set, feature_set))

                # Store
                grid_search_results['model'] = type(model).__name__
                grid_search_results['data_version'] = self.version
                grid_search_results['dataset'] = feature_set
                grid_search_results['type'] = 'Hyper Parameter'
                self.results = self.results.append(grid_search_results)
                self.results.to_csv(self.mainDir + 'Results.csv', index=False)
                params = Utils.parse_json(grid_search_results.iloc[0]['params'])

            # Validate
            if self.validateResults:
                self._validate_result(model, params, feature_set)

    def _grid_search_iteration(self, model, parameter_set, feature_set):
        """
        INTERNAL | Grid search for defined model, parameter set and feature set.
        """
        print('\n[AutoML] Starting Hyper Parameter Optimization for {} on {} features ({} samples, {} features)'.format(
            type(model).__name__, feature_set, len(self.x), len(self.colKeep[feature_set])))

        # Select data
        x, y = copy.copy(self.x[self.colKeep[feature_set]]), copy.copy(self.y)

        # Normalize Feature Set (the input remains original)
        if self.normalize:
            features_to_normalize = [k for k in self.colKeep[feature_set] if k not in self.dateCols + self.catCols]
            scaler = pickle.load(
                open(self.mainDir + 'Features/Scaler_{}_{}.pickle'.format(feature_set, self.version), 'rb'))
            x[features_to_normalize] = scaler.transform(x[features_to_normalize])
            if self.mode == 'regression':
                output_scaler = pickle.load(
                    open(self.mainDir + 'Features/OutputScaler_{}_{}.pickle'.format(feature_set, self.version), 'rb'))
                y = pd.Series(output_scaler.transform(y.values.reshape(-1, 1)).reshape(-1), name=self.target)

        # Cross-Validator
        cv = StratifiedKFold(n_splits=self.cvSplits, shuffle=self.shuffle)
        if self.mode == 'regression':
            cv = KFold(n_splits=self.cvSplits, shuffle=self.shuffle)

        # Select right hyper parameter optimizer
        if self.gridSearchType == 'base':
            grid_search = BaseGridSearch(model, params=parameter_set, cv=cv, scoring=self.objective,
                                         verbose=self.verbose)
        elif self.gridSearchType == 'halving':
            grid_search = HalvingGridSearch(model, params=parameter_set, cv=cv, scoring=self.objective,
                                            verbose=self.verbose)
        elif self.gridSearchType == 'optuna':
            grid_search = OptunaGridSearch(model, params=parameter_set, timeout=self.optunaBudget,
                                           cv=cv, scoring=self.objective, verbose=self.verbose)
        else:
            raise NotImplementedError('Only Base, Halving and Optuna are implemented.')
        # Get results
        results = grid_search.fit(x, y)
        results['worst_case'] = results['mean_objective'] - results['std_objective']
        return results.sort_values('worst_case', ascending=False)

    def _create_stacking(self):
        # todo implement
        """
        Based on the best performing models, in addition to cheap models based on very different assumptions,
        A stacking model is optimized to enhance/combine the performance of the models.
        """
        # First, the ideal dataset has to be chosen, we're restricted to a single one...
        # results = self._sort_results(self.results[np.logical_and(
        #     self.results['type'] == 'Hyper Parameter',
        #     self.results['data_version'] == self.version,
        # )])
        # feature_set = results['dataset'].value_counts()[0]

    def validate(self, model, feature_set, params=None):
        """
        Just a wrapper for the outside.
        Parameters:
        Model: The model to optimize, either string or class
        Feature Set: String
        (optional) params: Model parameters for which to validate
        """
        assert feature_set in self.colKeep.keys(), 'Feature Set not available.'

        # Get model
        if isinstance(model, str):
            models = Modelling(mode=self.mode, samples=len(self.x), scoring=self.objective).return_models()
            model = models[[i for i in range(len(models)) if type(models[i]).__name__ == model][0]]

        # Get params
        if params is not None:
            params = self._get_best_params(model, feature_set)

        # Run validation
        self._validate_result(model, params, feature_set)

    def _validate_result(self, master_model, params, feature_set):
        print('[AutoML] Validating results for {} ({} {} features) (params: {})'.format(
            type(master_model).__name__, len(self.colKeep[feature_set]), feature_set, params))

        # Initiate
        x, y = copy.deepcopy(self.x[self.colKeep[feature_set]]), copy.deepcopy(self.y)
        model = None
        output_scaler = None
        y_not_normalized = None

        # Normalize Feature Set (the X remains original)
        if self.normalize:
            normalize_features = [k for k in self.colKeep[feature_set] if k not in self.dateCols + self.catCols]
            scaler = pickle.load(
                open(self.mainDir + 'Features/Scaler_{}_{}.pickle'.format(feature_set, self.version), 'rb'))
            x[normalize_features] = scaler.transform(x[normalize_features])
            if self.mode == 'regression':
                output_scaler = pickle.load(
                    open(self.mainDir + 'Features/OutputScaler_{}_{}.pickle'.format(feature_set, self.version), 'rb'))
                y_not_normalized = y.values.reshape(-1, 1)
                y = pd.Series(output_scaler.transform(y.values.reshape(-1, 1)).reshape(-1), name=self.target)
        x, y = x.to_numpy(), y.to_numpy().reshape(-1, 1)

        # For Regression
        if self.mode == 'regression':

            # Cross-Validation Plots
            fig, ax = plt.subplots(math.ceil(self.cvSplits / 2), 2, sharex='all', sharey='all')
            fig.suptitle('{}-Fold Cross Validated Predictions - {}'.format(self.cvSplits, type(master_model).__name__))

            # Initialize iterables
            score = []
            cv = KFold(n_splits=self.cvSplits, shuffle=self.shuffle)
            # Cross Validate
            i = 0
            for i, (t, v) in enumerate(cv.split(x, y)):
                xt, xv, yt, yv = x[t], x[v], y[t].reshape((-1)), y[v].reshape((-1))
                model = copy.copy(master_model)
                model.set_params(**params)
                model.fit(xt, yt)

                # Metrics
                score.append(self.scorer(model, xv, yv))

                # Plot
                ax[i // 2][i % 2].set_title('Fold-{}'.format(i))
                if self.normalize:
                    ax[i // 2][i % 2].plot(y_not_normalized[v], color='#2369ec')
                    ax[i // 2][i % 2].plot(output_scaler.inverse_transform(model.predict(xv)),
                                           color='#ffa62b', alpha=0.4)
                else:
                    ax[i // 2][i % 2].plot(yv, color='#2369ec')
                    ax[i // 2][i % 2].plot(model.predict(xv), color='#ffa62b', alpha=0.4)

            # Print & Finish plot
            print('[AutoML] {}:        {:.2f} \u00B1 {:.2f}'.format(self.scorer._score_func.__name__,
                                                                    np.mean(score), np.std(score)))
            ax[i // 2][i % 2].legend(['Output', 'Prediction'])
            plt.show()

        # For BINARY classification
        elif self.mode == 'classification' and self.y.nunique() == 2:
            # Initiating
            fig, ax = plt.subplots(math.ceil(self.cvSplits / 2), 2, sharex='all', sharey='all')
            fig.suptitle('{}-Fold Cross Validated Predictions - {} ({})'.format(
                self.cvSplits, type(master_model).__name__, feature_set))
            fig2, ax2 = plt.subplots(figsize=[12, 8])
            accuracy = []
            precision = []
            sensitivity = []
            specificity = []
            f1_score = []
            area_under_curves = []
            true_positive_rates = []
            cm = np.zeros((2, 2))
            mean_fpr = np.linspace(0, 1, 100)

            # Modelling
            cv = StratifiedKFold(n_splits=self.cvSplits)
            for i, (t, v) in enumerate(cv.split(x, y)):
                n = len(v)
                xt, xv, yt, yv = x[t], x[v], y[t].reshape((-1)), y[v].reshape((-1))
                model = copy.copy(master_model)
                model.set_params(**params)
                model.fit(xt, yt)
                predictions = model.predict(xv).reshape((-1))

                # Metrics
                tp = np.logical_and(np.sign(predictions) == 1, yv == 1).sum()
                tn = np.logical_and(np.sign(predictions) == 0, yv == 0).sum()
                fp = np.logical_and(np.sign(predictions) == 1, yv == 0).sum()
                fn = np.logical_and(np.sign(predictions) == 0, yv == 1).sum()
                accuracy.append((tp + tn) / n * 100)
                if tp + fp > 0:
                    precision.append(tp / (tp + fp) * 100)
                if tp + fn > 0:
                    sensitivity.append(tp / (tp + fn) * 100)
                if tn + fp > 0:
                    specificity.append(tn / (tn + fp) * 100)
                if tp + fp > 0 and tp + fn > 0:
                    f1_score.append(2 * precision[-1] * sensitivity[-1] / (precision[-1] + sensitivity[-1]) if
                                    precision[-1] + sensitivity[-1] > 0 else 0)
                cm += np.array([[tp, fp], [fn, tn]]) / self.cvSplits

                # ROC calculations
                viz = metrics.plot_roc_curve(model, xv, yv, name='ROC fold {}'.format(i + 1), alpha=0.3, ax=ax2)
                interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0.0
                true_positive_rates.append(interp_tpr)
                area_under_curves.append(viz.roc_auc)

                # Plot
                ax[i // 2][i % 2].plot(yv, c='#2369ec', alpha=0.6)
                ax[i // 2][i % 2].plot(predictions, c='#ffa62b')
                ax[i // 2][i % 2].set_title('Fold-{}'.format(i))

            # Results
            print('[AutoML] Accuracy:        {:.2f} \u00B1 {:.2f} %%'.format(np.mean(accuracy), np.std(accuracy)))
            print('[AutoML] Precision:       {:.2f} \u00B1 {:.2f} %%'.format(np.mean(precision), np.std(precision)))
            print('[AutoML] Recall:          {:.2f} \u00B1 {:.2f} %%'.format(np.mean(sensitivity), np.std(sensitivity)))
            print('[AutoML] Specificity:     {:.2f} \u00B1 {:.2f} %%'.format(np.mean(specificity), np.std(specificity)))
            print('[AutoML] F1-score:        {:.2f} \u00B1 {:.2f} %%'.format(np.mean(f1_score), np.std(f1_score)))
            print('[AutoML] Confusion Matrix:')
            print('[AutoML] Prediction / true |  Faulty   |   Healthy      ')
            print('[AutoML]  Faulty           |  {}|  {:.1f}'.format(('{:.1f}'.format(cm[0, 0])).ljust(9), cm[0, 1]))
            print('[AutoML]  Healthy          |  {}|  {:.1f}'.format(('{:.1f}'.format(cm[1, 0])).ljust(9), cm[1, 1]))

            # Check whether plot is possible
            if isinstance(model, sklearn.linear_model.Lasso) or isinstance(model, sklearn.linear_model.Ridge):
                return

            # Adjust plots
            ax2.plot([0, 1], [0, 1], linestyle='--', lw=2, color='#ffa62b',
                     label='Chance', alpha=.8)
            mean_tpr = np.mean(true_positive_rates, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = metrics.auc(mean_fpr, mean_tpr)
            std_auc = np.std(area_under_curves)
            ax2.plot(mean_fpr, mean_tpr, color='#2369ec',
                     label=r'Mean ROC (AUC = {:.2f} $\pm$ {:.2f})'.format(mean_auc, std_auc),
                     lw=2, alpha=.8)
            std_tpr = np.std(true_positive_rates, axis=0)
            true_pos_rates_upper = np.minimum(mean_tpr + std_tpr, 1)
            true_pos_rates_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax2.fill_between(mean_fpr, true_pos_rates_lower, true_pos_rates_upper, color='#729ce9', alpha=.2,
                             label=r'$\pm$ 1 std. dev.')
            ax2.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
                    title="ROC Curve - {}".format(type(master_model).__name__))
            ax2.legend(loc="lower right")
            fig2.savefig(self.mainDir + 'Validation/ROC_{}.png'.format(type(model).__name__), format='png', dpi=200)

        # For MULTICLASS classification
        elif self.mode == 'classification':
            # Initiating
            fig, ax = plt.subplots(math.ceil(self.cvSplits / 2), 2, sharex='all', sharey='all')
            fig.suptitle('{}-Fold Cross Validated Predictions - {} ({})'.format(
                self.cvSplits, type(master_model).__name__, feature_set))
            n_classes = self.y.nunique()
            f1_score = np.zeros((self.cvSplits, n_classes))
            log_loss = np.zeros(self.cvSplits)
            avg_acc = np.zeros(self.cvSplits)

            # Modelling
            cv = StratifiedKFold(n_splits=self.cvSplits)
            for i, (t, v) in enumerate(cv.split(x, y)):
                xt, xv, yt, yv = x[t], x[v], y[t].reshape((-1)), y[v].reshape((-1))
                model = copy.copy(master_model)
                model.set_params(**params)
                model.fit(xt, yt)
                predictions = model.predict(xv).reshape((-1))

                # Metrics
                f1_score[i] = metrics.f1_score(yv, predictions, average=None)
                avg_acc[i] = metrics.accuracy_score(yv, predictions)
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(xv)
                    log_loss[i] = metrics.log_loss(yv, probabilities)

                # Plot
                ax[i // 2][i % 2].plot(yv, c='#2369ec', alpha=0.6)
                ax[i // 2][i % 2].plot(predictions, c='#ffa62b')
                ax[i // 2][i % 2].set_title('Fold-{}'.format(i))

            # Results
            print('F1 scores:')
            print(''.join([' Class {} |'.format(i) for i in range(n_classes)]))
            print(''.join([' {:.2f} '.ljust(9).format(f1) + '|' for f1 in np.mean(f1_score, axis=0)]))
            print('Average Accuracy: {:.2f} \u00B1 {:.2f}'.format(np.mean(avg_acc), np.std(avg_acc)))
            if hasattr(model, 'predict_proba'):
                print('Log Loss:         {:.2f} \u00B1 {:.2f}'.format(np.mean(log_loss), np.std(log_loss)))

    def _prepare_production_files(self, model=None, feature_set=None, params=None):
        if not os.path.exists(self.mainDir + 'Production/v{}/'.format(self.version)):
            os.mkdir(self.mainDir + 'Production/v{}/'.format(self.version))
        # Get sorted results for this data version
        results = self._sort_results(self.results[self.results['data_version'] == self.version])

        # In the case args are provided
        if model is not None and feature_set is not None:
            # Take name if model instance is given
            if not isinstance(model, str):
                model = type(model).__name__
            if params is None:
                results = self._sort_results(
                    results[np.logical_and(results['model'] == model, results['dataset'] == feature_set)])
                params = Utils.parse_json(results.iloc[0]['params'])

        # Otherwise find best
        else:
            model = results.iloc[0]['model']
            feature_set = results.iloc[0]['dataset']
            params = results.iloc[0]['params']
            if isinstance(params, str):
                params = Utils.parse_json(params)

        # Notify of results
        print('[AutoML] Preparing Production Env Files for {}, feature set {}'.format(model, feature_set))
        print('[AutoML] ', params)
        print('[AutoML] {}: {:.2f} \u00B1 {:.2f}'.format(
            self.scorer._score_func.__name__, results.iloc[0]['mean_objective'], results.iloc[0]['std_objective']))

        # Save Features
        self.bestFeatures = self.colKeep[feature_set]
        json.dump(self.bestFeatures, open(self.mainDir + 'Production/v{}/Features.json'.format(self.version), 'w'))

        # Copy data
        x, y = copy.copy(self.x[self.bestFeatures]), copy.copy(self.y)

        # Save Scaler & Normalize
        if self.normalize:
            # Save
            shutil.copy(self.mainDir + 'Features/Scaler_{}_{}.pickle'.format(feature_set, self.version),
                        self.mainDir + 'Production/v{}/Scaler.pickle'.format(self.version))
            if self.mode == 'regression':
                shutil.copy(self.mainDir + 'Features/OutputScaler_{}_{}.pickle'.format(feature_set, self.version),
                            self.mainDir + 'Production/v{}/OutputScaler.pickle'.format(self.version))

            # Normalize
            normalize_features = [k for k in self.bestFeatures if k not in self.dateCols + self.catCols]
            self.bestScaler = pickle.load(
                open(self.mainDir + 'Features/Scaler_{}_{}.pickle'.format(feature_set, self.version), 'rb'))
            x = self.bestScaler.transform(x[normalize_features])
            if self.mode == 'regression':
                self.bestOutputScaler = pickle.load(
                    open(self.mainDir + 'Features/OutputScaler_{}_{}.pickle'.format(feature_set, self.version), 'rb'))
                y = self.bestOutputScaler.transform(y.values.reshape(-1, 1)).reshape(-1)
            else:
                y = self.y.to_numpy()

        # Cluster Features require additional 'Centers' file
        if any(['dist__' in key for key in self.bestFeatures]):
            shutil.copy(self.mainDir + 'Features/KMeans_v{}.csv'.format(self.version),
                        self.mainDir + 'Production/v{}/KMeans.csv'.format(self.version))

        # Model
        models = Modelling(mode=self.mode, samples=len(self.x), scoring=self.objective).return_models()
        self.bestModel = [mod for mod in models if type(mod).__name__ == model][0]
        self.bestModel.set_params(**params)
        self.bestModel.fit(x, y)
        joblib.dump(self.bestModel, self.mainDir + 'Production/v{}/Model.joblib'.format(self.version))

        # Predict function
        predict_code = self.create_predict_function(self.customCode)
        with open(self.mainDir + 'Production/v{}/Predict.py'.format(self.version), 'w') as f:
            f.write(predict_code)
        with open(self.mainDir + 'Production/v{}/__init__.py'.format(self.version), 'w') as f:
            f.write('')

        # Pipeline
        pickle.dump(self, open(self.mainDir + 'Production/v{}/Pipeline.pickle'.format(self.version), 'wb'))
        return

    def _error_analysis(self):
        # todo implement
        pass

    def _convert_data(self, data):
        # todo implement
        # Load files
        folder = 'Production/v{}/'.format(self.version)
        features = json.load(open(self.mainDir + folder + 'Features.json', 'r'))

        # Clean data
        data = Utils.clean_keys(data)
        data = self.dataProcessor.convert_data_types(data)
        data = self.dataProcessor.remove_duplicates(data)
        data = self.dataProcessor.remove_outliers(data)
        data = self.dataProcessor.remove_missing_values(data)
        if data.astype('float32').replace([np.inf, -np.inf], np.nan).isna().sum().sum() != 0:
            raise ValueError('Data should not contain NaN or Infinities after cleaning!')

        # Save output
        if self.target in data.keys():
            y = data[self.target].to_numpy().reshape((-1, 1))
        else:
            y = None

        # Convert Features
        if 'KMeans.csv' in os.listdir(self.mainDir + folder):
            k_means = pd.read_csv(self.mainDir + folder + 'KMeans.csv')
            x = self.featureProcessor.transform(data=data, features=features, k_means=k_means)
        else:
            x = self.featureProcessor.transform(data, features)

        if x.astype('float32').replace([np.inf, -np.inf], np.nan).isna().sum().sum() != 0:
            raise ValueError('Data should not contain NaN or Infinity after adding features!')

        # Normalize
        if self.normalize:
            scaler = pickle.load(open(self.mainDir + folder + 'Scaler.pickle', 'rb'))
            x[x.keys()] = scaler.transform(x)
            if self.mode == 'regression' and y is not None:
                output_scaler = pickle.load(open(self.mainDir + folder + 'OutputScaler.pickle', 'rb'))
                y = output_scaler.transform(y)

        # Return
        return x, y

    def predict(self, data):
        """
        Full script to make predictions. Uses 'Production' folder with defined or latest version.
        @param data: data to do prediction on
        """
        # todo save all inside one pickle
        # Feature Extraction, Selection and Normalization
        model = joblib.load(self.mainDir + 'Production/v{}/Model.joblib'.format(self.version))
        if self.verbose > 0:
            print('[AutoML] Predicting with {}, v{}'.format(type(model).__name__, self.version))
        x, y = self._convert_data(data)

        # Predict
        if self.mode == 'regression':
            if self.normalize:
                output_scaler = pickle.load(open(self.mainDir + 'Production/v{}/OutputScaler.pickle'.format(
                    self.version), 'rb'))
                predictions = output_scaler.inverse_transform(model.predict(x))
            else:
                predictions = model.predict(x)
        elif self.mode == 'classification':
            try:
                predictions = model.predict_proba(x)[:, 1]
            except AttributeError:
                predictions = model.predict(x)
        else:
            raise ValueError('Unsupported mode')

        return predictions

    def predict_proba(self, data):
        """
        Returns probabilistic prediction, only for classification.
        @param data: data to do prediction on
        """
        # Load model
        model = joblib.load(self.mainDir + 'Production/v{}/Model.joblib'.format(self.version))\

        # Tests
        assert self.mode == 'classification', 'Predict_proba only available for classification'
        assert hasattr(model, 'predict_proba'), '{} has no attribute predict_proba'.format(
            type(model).__name__)

        # Print
        if self.verbose > 0:
            print('[AutoML] Predicting with {}, v{}'.format(type(model).__name__, self.version))

        # Convert data
        x, y = self._convert_data(data)

        # Predict
        return model.predict_proba(x)

    def create_predict_function(self, custom_code):
        """
        This function returns a string, which can be used to make predictions.
        This is in a predefined format, a Predict class, with a predict function taking the arguments
        model: trained sklearn-like class with the .fit() function
        features: list of strings containing all features fed to the model
        scaler: trained sklearn-like class with .transform function
        data: the data to predict on
        Now this function has the arg decoding, which allows custom code injection
        """
        # Check if predict file exists already to increment version
        if os.path.exists(self.mainDir + 'Production/v{}/Predict.py'.format(self.version)):
            with open(self.mainDir + 'Production/v{}/Predict.py'.format(self.version), 'r') as f:
                predict_file = f.read()
            ind = predict_file.find('self.version = ') + 16
            old_version = predict_file[ind: predict_file.find("'", ind)]
            minor_version = int(old_version[old_version.find('.') + 1:])
            version = old_version[:old_version.find('.') + 1] + str(minor_version + 1)
        else:
            version = 'v{}.0'.format(self.version)
        print('Creating Prediction {}'.format(version))
        data_process = self.dataProcessor.export_function()
        feature_process = self.featureProcessor.export_function()
        return """import pandas as pd
import numpy as np
import struct, re, copy, os


class Predict(object):

    def __init__(self):
        self.version = '{}'

    def predict(self, model, features, data, **args):
        '''
        Prediction function for Amplo AutoML.
        This is in a predefined format:
        - a 'Predict' class, with a 'predict' function taking the arguments:
            model: trained sklearn-like class with the .fit() function
            features: list of strings containing all features fed to the model
            data: the data to predict on
        Note: May depend on additional named arguments within args.
        '''
        ###############
        # Custom Code #
        ###############""".format(version) + textwrap.indent(custom_code, '    ') \
               + data_process + feature_process + '''
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
                predictions = args['OutputScaler'].inverse_transform(model.predict(X))
            else:
                predictions = model.predict(X)
        if mode == 'classification':
            try:
                predictions = model.predict_proba(X)[:, 1]
            except:
                predictions = model.predict(X)

        return predictions
'''.format(self.mode, self.normalize)
