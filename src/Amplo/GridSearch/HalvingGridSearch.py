import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
import multiprocessing as mp
from .BaseGridSearch import BaseGridSearch


class HalvingGridSearch(BaseGridSearch):

    def __init__(self, model, params=None, cv=None, scoring=None, verbose=None):
        super().__init__(model, params, cv, scoring, verbose)
        # Params
        self.resource = 'n_samples'
        self.max_resource = 'auto'
        self.min_resource = 200

        # Update them
        self.set_resources()

    def set_resources(self):
        if self.model.__module__ == 'catboost.core':
            self.resource = 'n_estimators'
            self.max_resource = 3000
            self.min_resource = 250
        if self.model.__module__ == 'sklearn.ensemble._bagging' or \
                self.model.__module__ == 'xgboost.sklearn' or \
                self.model.__module__ == 'lightgbm.sklearn' or \
                self.model.__module__ == 'sklearn.ensemble._forest':
            self.resource = 'n_estimators'
            self.max_resource = 1500
            self.min_resource = 50

    def fit(self, x, y):
        # Update minimum resource for samples (based on dataset)
        if self.resource == 'n_samples':
            self.min_resource = int(0.2 * len(x)) if len(x) > 5000 else len(x)

        # Set up and run grid search
        halving_random_search = HalvingRandomSearchCV(self.model, self.params,
                                                      n_candidates=200,
                                                      resource=self.resource,
                                                      max_resources=self.max_resource,
                                                      min_resources=self.min_resource,
                                                      cv=self.cv,
                                                      scoring=self.scoring,
                                                      factor=3, n_jobs=mp.cpu_count() - 1,
                                                      verbose=self.verbose)
        halving_random_search.fit(x, y)

        # Parse results
        scikit_results = pd.DataFrame(halving_random_search.cv_results_)
        results = pd.DataFrame()
        results[['params', 'mean_objective', 'std_objective', 'mean_time', 'std_time']] = scikit_results[
            ['params', 'mean_test_score', 'std_test_score', 'mean_fit_time', 'std_fit_time']]
        results['worst_case'] = - results['mean_objective'] - results['std_objective']

        # Update resource in results
        if self.resource != 'n_samples':
            for i in range(len(results)):
                results.loc[results.index[i], 'params'][self.resource] = self.max_resource

        return results
