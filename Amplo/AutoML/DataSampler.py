import time
import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import RandomUnderSampler


class DataSampler:

    def __init__(self,
                 method: str = 'both',
                 margin: float = 0.1,
                 cv_splits: int = 3,
                 shuffle: bool = True,
                 fast_run: bool = True,
                 objective: str = None,
                 verbosity: int = 1,
                 ):
        """
        Trial and error various strategies:
        - Under-sample all to minority
        - Over-sample all to majority
        - Combination of the two, with say 1k, 5k, 25k, 50k, 75k and 100k samples
        This sensitivity is problem dependent, but has major consequence for hyper
        parameter optimization. Need to get it right.
        Need to make sure that metrics are evaluated on original distribution.
        https://imbalanced-learn.org/stable/references/index.html#api

        Parameters
        ----------
        method str: Whether to under-sample, over-sample, or a combination from the both ('under', 'over', or 'both')
        margin float: If the imbalance is smaller than this margin, skip the balancing, should be within [0, 1]
        fast_run bool: If true, skips exhaustive samplers
        objective str: Scikit-Learn metric
        """
        # Tests
        assert method in ["under", "over", "both"], "Type needs to be 'under', 'over' or 'both'"
        assert 1e-6 < margin < 1, "Margin needs to be within [0, 1]."

        # Objective
        if objective is not None:
            self.objective = objective
        else:
            self.objective = 'neg_log_loss'
        assert objective in metrics.SCORERS.keys(), 'Metric not supported, look at sklearn.metrics.SCORERS.keys()'
        self.scorer = metrics.SCORERS[self.objective]

        # Parse
        self.method = method
        self.margin = margin
        self.fast_run = fast_run

        # Settings
        self.cv = StratifiedKFold(n_splits=cv_splits, shuffle=shuffle)
        self.min_samples = 5e3
        self.max_samples = 1e5
        self.verbosity = verbosity

        # Results
        self.is_fitted = False
        self.opt_sampler = None
        self.results = pd.DataFrame(columns=['sampler', 'mean_objective', 'std_objective', 'mean_time', 'mean_std'])

    def get_samplers(self, y: pd.Series) -> list:
        """
        Return selection of samplers based on the label vector
        Parameters
        ----------
        labels [pd.Series]: Vector with labels
        """
        # Assess labels
        n_classes = len(set(y.tolist()))
        n_samples = len(y)
        imbalance = y.value_counts().min() / y
        print('[AutoML] Data Balancer: Found {:.1f}% imbalance with {:.0f} classes.'.format(imbalance, n_classes))

        # If imbalance is negligible, return nothing
        if imbalance > (1 - self.margin) / n_classes:
            return []

        # Init
        samplers = []

        # Under Samplers
        if self.method != 'under' and n_samples > self.min_samples:
            samplers.append(OneSidedSelection(sampling_strategy='not minority'))
            samplers.append(RandomUnderSampler(sampling_strategy='not minority'))
            if not self.fast_run:
                samplers.append(TomekLinks(sampling_strategy='not minority'))

        # Over Samplers
        if self.method != 'over' and n_samples < self.max_samples:
            samplers.append(RandomOverSampler(sampling_strategy='not majority'))
            if not self.fast_run:
                samplers.append(SMOTE(sampling_strategy='not majority'))

        if self.method == 'both' and self.min_samples < n_samples < self.max_samples and not self.fast_run:
            samplers.append(SMOTETomek(sampling_strategy='all'))

        return samplers

    def _score(self, sampler, x: pd.DataFrame, y: pd.Series) -> [list, list]:
        """
        Scores a sampler for data

        Parameters
        ----------
        sampler : Imbalance-learn sampler
        x [pd.DataFrame]: Input data
        y [pd.Series]: Output data

        Returns
        -------
        Scores [list]: List of objective for every fold
        Times [list]: List of runtimes for every fold
        """
        # Initiate
        scores, times = [], []

        # Cross-Validate
        for i, (ti, vi) in self.cv.split(x, y):
            # Split data
            xt, xv, yt, yv = x[ti], x[vi], y[ti], y[vi]

            # Note start time
            t_start = time.time()

            # Re - sample
            if sampler is not None:
                x_res, y_res = sampler.fit_resample(xt, yt)
            else:
                x_res, y_res = xt, yt

            # Train a model
            model = RandomForestClassifier(n_estimators=50, max_depth=5)
            model.fit(x_res, y_res)

            # Score
            scores.append(self.scorer(model, xv, yv))
            times.append(time.time() - t_start)

        return scores, times

    def fit(self, x: pd.DataFrame, y: pd.Series):
        """

        Parameters
        ----------
        x [pd.DataFrame]: Input data
        y [pd.Series]: Output data
        """
        # Get samplers
        samplers = self.get_samplers(y)
        if self.verbosity > 0:
            print('[AutoML] Balancing data with {:.0f} samplers'.format(len(samplers)))

        # Get baseline
        scores, times = self._score(None, x, y)
        self.results.append({
            'sampler': 'Baseline',
            'mean_objective': np.mean(scores), 'std_objective': np.std(scores),
            'worst_case': np.mean(scores) - np.std(scores),
            'mean_time': np.mean(times), 'std_time': np.std(scores),
        })

        # Iterate through samplers
        for sampler in samplers:
            name = type(sampler).__name__

            # Score
            scores, times = self._score(sampler, x, y)

            # Store results
            self.results.append({
                'sampler': name,
                'mean_objective': np.mean(scores), 'std_objective': np.std(scores),
                'worst_case': np.mean(scores) - np.std(scores),
                'mean_time': np.mean(time), 'std_time': np.std(time),
                }, ignore_index=True)

            # Print results
            if self.verbosity > 0:
                print('[AutoML] Fitted {}, {}: {:.4f} \u00B1 {:.4f}'
                      .format(name.ljust(25), self.objective, np.mean(scores), np.std(scores)))

        # Set fitted
        self.is_fitted = True

        # Set optimal sampler
        opt_ind = np.where(self.results['worst_case'] == self.results['worst_case'].min())[0][0]
        self.opt_sampler = samplers[opt_ind]

    def resample(self, x: pd.DataFrame, y: pd.Series) -> [pd.DataFrame, pd.Series]:
        """
        Resamples the data with the best sampler

        Parameters
        ----------
        x [pd.DataFrame]: Input data
        y [pd.Series]: Output data

        Returns
        -------
        x_res [pd.DataFrame]: Resampled input data
        y_res [pd.DataFrame]: Resampled output data
        """
        # Check whether fitted
        assert self.is_fitted, "Data Sampler not yet fitted. Run .fit(x, y) first. "

        # Print start
        if self.verbosity > 0:
            print('[AutoML] Resampling data with {}'.format(type(self.opt_sampler).__name__))

        # Resample
        x_res, y_res = self.opt_sampler.fit_resample(x, y)

        return x_res, y_res

    def fit_resample(self, x: pd.DataFrame, y: pd.Series):
        """
        Fits and returns resampled data

        Parameters
        ----------
        x [pd.DataFrame]: Input data
        y [pd.Series]: Output data

        Returns
        -------
        x_res [pd.DataFrame]: Resampled input data
        y_res [pd.Series]: Resampled output data
        """
        # Fit
        self.fit(x, y)

        # Resample
        x_res, y_res = self.resample(x, y)

        # Return
        return x_res, y_res
