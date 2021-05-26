import time
import copy
import datetime
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics


class BaseGridSearch:

    def __init__(self, model, params=None, cv=None, scoring=metrics.SCORERS['accuracy'],
                 verbose=0):
        """
        Basic exhaustive grid search.
        @param model: Model object to optimize
        @param cv: Cross-Validator for scoring
        @param scoring: make_scorer for scoring
        """
        # Args
        self.model = model
        self.params = params
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose

        # Checks
        assert model is not None, 'Model not provided'
        assert isinstance(params, dict), 'Parameter set needs to be a dictionary'
        assert all([isinstance(x, list) for x in params.values()]), 'Parameter set dictionary needs to be filled with' \
                                                                    ' lists.'
        assert scoring in metrics.SCORERS.values(), 'Chose a scorer from sklearn.metrics.SCORERS'
        
        # Initiate
        self.parsedParams = []
        self.result = []
        self._parse_params()
        self.best = [None, None]    # Mean score, Std score

    def _parse_params(self):
        k, v = zip(*self.params.items())
        self.parsedParams = [dict(zip(k, v)) for v in itertools.product(*self.params.values())]
        print('[GridSearch] %i folds with %i parameter combinations, %i runs.' % (
            self.cv.n_splits,
            len(self.parsedParams),
            len(self.parsedParams) * self.cv.n_splits))

    def fit(self, x, y):
        # Convert to Numpy
        if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            x = np.array(x)
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = np.array(y).reshape((-1))

        # Loop through parameters
        for i, param in tqdm(enumerate(self.parsedParams)):
            # print('[GridSearch] ', param)
            scoring = []
            timing = []
            for train_ind, val_ind in self.cv.split(x, y):
                # Start Timer
                t = time.time()

                # Split data
                x_train, x_val = x[train_ind], x[val_ind]
                y_train, y_val = y[train_ind], y[val_ind]

                # Model training
                model = copy.copy(self.model)
                model.set_params(**param)
                model.fit(x_train, y_train)

                # Results
                scoring.append(self.scoring(model, x_val, y_val))
                timing.append(time.time() - t)

            # Compare scores
            if np.mean(scoring) - np.std(scoring) > self.best[0] - self.best[1]:
                self.best = [np.mean(scoring), np.std(scoring)]

            if self.verbose > 0:
                print('[GridSearch][%s] Score: %.4f \u00B1 %.4f (in %.1f seconds) (Best score so-far: %.4f \u00B1 '
                      '%.4f) (%i / %i)' % (datetime.now().strftime('%H:%M'), np.mean(scoring), np.std(scoring),
                                           np.mean(timing), self.best[0], self.best[1], i + 1, len(self.parsedParams)))
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
