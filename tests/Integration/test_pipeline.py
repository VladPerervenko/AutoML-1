import os
import json
import shutil
import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.metrics import log_loss
from Amplo import Pipeline


class TestPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        x, y = make_classification()
        cls.c_data = pd.DataFrame(x, columns=['Feature_{}'.format(i) for i in range(x.shape[1])])
        cls.c_data['target'] = y
        x, y = make_regression()
        cls.r_data = pd.DataFrame(x, columns=['Feature_{}'.format(i) for i in range(x.shape[1])])
        cls.r_data['target'] = y

    def test_regression(self):
        if os.path.exists('AutoML'):
            shutil.rmtree('AutoML')

        pipeline = Pipeline('target',
                            name='AutoReg',
                            mode='regression',
                            objective='r2',
                            feature_timeout=5,
                            grid_search_iterations=1,
                            grid_search_time_budget=10,
                            grid_search_candidates=2,
                            plot_eda=False,
                            process_data=False,
                            document_results=False,
                            )
        pipeline.fit(self.r_data)

        # Test Directories
        assert os.path.exists('AutoML')
        assert os.path.exists('AutoML/Data')
        assert os.path.exists('AutoML/Features')
        assert os.path.exists('AutoML/Production')
        assert os.path.exists('AutoML/Documentation')
        assert os.path.exists('AutoML/Results.csv')

        # Test data handling
        c, _ = pipeline.convert_data(self.r_data.drop('target', axis=1))
        x = pipeline.x[pipeline.settings['features']]
        y = self.r_data['target']
        assert np.allclose(c.values, x.values), "Inconsistent X: max diff: {:.2e}"\
            .format(np.max(abs(c.values - x.values)))
        assert np.allclose(y, pipeline.y), "Inconsistent Y"

        # Pipeline Prediction
        prediction = pipeline.predict(self.r_data)
        assert len(prediction.shape) == 1
        assert r2_score(self.r_data['target'], prediction) > 0.75

        # Settings prediction
        settings = json.load(open('AutoML/Production/v1/Settings.json', 'r'))
        p = Pipeline()
        p.load_settings(settings, pipeline.bestModel)
        assert np.allclose(p.predict(self.r_data), prediction)

        # Cleanup
        shutil.rmtree('AutoML')

    def test_classification(self):
        if os.path.exists('AutoML'):
            shutil.rmtree('AutoML')
        pipeline = Pipeline('target',
                            name='AutoClass',
                            mode='classification',
                            objective='neg_log_loss',
                            standardize=True,
                            feature_timeout=5,
                            grid_search_iterations=1,
                            grid_search_time_budget=10,
                            grid_search_candidates=2,
                            plot_eda=False,
                            process_data=True,
                            document_results=False,
                            )
        pipeline.fit(self.c_data)

        # Tests
        assert os.path.exists('AutoML')
        assert os.path.exists('AutoML/EDA')
        assert os.path.exists('AutoML/Data')
        assert os.path.exists('AutoML/Features')
        assert os.path.exists('AutoML/Production')
        assert os.path.exists('AutoML/Results.csv')

        # Pipeline Prediction
        prediction = pipeline.predict_proba(self.c_data)
        assert log_loss(self.c_data['target'], prediction) > -1

        # Settings prediction
        settings = json.load(open('AutoML/Production/v1/Settings.json', 'r'))
        p = Pipeline()
        p.load_settings(settings, pipeline.bestModel)
        assert np.allclose(p.predict_proba(self.c_data), prediction)

        # Cleanup
        shutil.rmtree('AutoML')
