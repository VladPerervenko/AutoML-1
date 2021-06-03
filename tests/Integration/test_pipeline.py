import os
import pickle
import pytest
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
        pipeline = Pipeline('target',
                            project='AutoReg',
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
        assert os.path.exists('AutoReg')
        assert os.path.exists('AutoReg/Data')
        assert os.path.exists('AutoReg/Features')
        assert os.path.exists('AutoReg/Production')
        assert os.path.exists('AutoReg/Documentation')
        assert os.path.exists('AutoReg/Results.csv')

        # Test data handling
        c, _ = pipeline.convert_data(self.r_data.drop('target', axis=1))
        x = pipeline.x[pipeline.bestFeatures]
        y = self.r_data['target']
        assert np.allclose(c.values, x.values)  # x has been manipulated
        assert np.allclose(y, pipeline.y)       # y mustn't have changed

        # Pipeline Prediction
        prediction = pipeline.predict(self.r_data)
        assert len(prediction.shape) == 1
        assert r2_score(self.r_data['target'], prediction) > 0.75

        # Pickle Prediction
        p = pickle.load(open('AutoReg/Production/v0/Pipeline.pickle', 'rb'))
        assert np.allclose(p.predict(self.r_data), prediction)

    # def test_classification(self):
    #     pipeline = Pipeline('target',
    #                         project='AutoClass',
    #                         mode='classification',
    #                         objective='neg_log_loss',
    #                         feature_timeout=5,
    #                         grid_search_iterations=1,
    #                         grid_search_time_budget=10,
    #                         grid_search_candidates=2,
    #                         plot_eda=False,
    #                         process_data=True,
    #                         document_results=False,
    #                         )
    #     pipeline.fit(self.c_data)
    #
    #     # Tests
    #     assert os.path.exists('AutoClass')
    #     assert os.path.exists('AutoClass/EDA')
    #     assert os.path.exists('AutoClass/Data')
    #     assert os.path.exists('AutoClass/Features')
    #     assert os.path.exists('AutoClass/Production')
    #     assert os.path.exists('AutoClass/Results.csv')
    #
    #     # Pipeline Prediction
    #     prediction = pipeline.predict_proba(self.c_data)
    #     assert log_loss(self.c_data['target'], prediction) > -1


@pytest.fixture(scope="session", autouse=True)
def teardown():
    yield
    # Never remove, new run means new data
    if os.path.exists('AutoClass'):
        shutil.rmtree('AutoClass')
    if os.path.exists('AutoReg'):
        shutil.rmtree('AutoReg')
