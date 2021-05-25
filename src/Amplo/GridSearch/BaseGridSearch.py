import time
import copy
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


class BaseGridSearch:

    def __init__(self, model, params, cv=None, scoring=metrics.SCORERS['accuracy']):
        """
        Basic exhaustive grid search.
        @param model: Model object to optimize
        @param params:  Parameter set to optimize
        @param cv: Cross-Validator for scoring
        @param scoring: make_scorer for scoring
        """
        # Args
        self.model = model
        self.params = params
        self.cv = cv
        self.scoring = scoring

        # Checks
        assert model is not None, 'Model not provided'
        assert isinstance(params, dict), 'Parameter set should be dictionary'
        assert all([isinstance(x, list) for x in params.values()]), 'Parameter set dictionairy needs to be filled ' \
                                                                    'with lists '
        
        # Initiate            
        self.parsed_params = []
        self.result = []
        self._parse_params()

    def _parse_params(self):
        k, v = zip(*self.params.items())
        self.parsed_params = [dict(zip(k, v)) for v in itertools.product(*self.params.values())]
        print('[GridSearch] %i folds with %i parameter combinations, %i runs.' % (
            self.cv.n_splits,
            len(self.parsed_params),
            len(self.parsed_params) * self.cv.n_splits))

    def fit(self, x, y):
        # Convert to Numpy
        if isinstance(x, pd.DataFrame) or isinstance(x, pd.core.series.Series):
            x = np.array(x)
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.core.series.Series):
            y = np.array(y).reshape((-1))

        # Loop through parameters
        for i, param in tqdm(enumerate(self.parsed_params)):
            # print('[GridSearch] ', param)
            scoring = []
            timing = []
            for train_ind, val_ind in self.cv.split(x, y):
                # Start Timer
                t = time.time()

                # Split data
                xtrain, xval = x[train_ind], x[val_ind]
                ytrain, yval = y[train_ind], y[val_ind]

                # Model training
                model = copy.copy(self.model)
                model.set_params(**param)
                model.fit(xtrain, ytrain)

                # Results
                scoring.append(self.scoring(model.predict(xval), yval))
                timing.append(time.time() - t)

            # Compare scores
            if scoring == metrics.mean_absolute_error or scoring == metrics.mean_squared_error:
                if np.mean(scoring) + np.std(scoring) <= self.best[0] + self.best[1]:
                    self.best = [np.mean(scoring), np.std(scoring)]
            else:
                if np.mean(scoring) - np.std(scoring) > self.best[0] - self.best[1]:
                    self.best = [np.mean(scoring), np.std(scoring)]

            # print('[GridSearch] [AutoML] Score: %.4f \u00B1 %.4f (in %.1f seconds) (Best score so-far: %.4f \u00B1 %.4f) (%i / %i)' %
            #       (datetime.now().strftime('%H:%M'), np.mean(scoring), np.std(scoring), np.mean(timing), self.best[0], self.best[1], i + 1, len(self.parsed_params)))
            self.result.append({
                'scoring': scoring,
                'mean_objective': np.mean(scoring),
                'std_objective': np.std(scoring),
                'time': timing,
                'mean_time': np.mean(timing),
                'std_time': np.std(timing),
                'params': param
            })
        return pd.DataFrame(self.result)
