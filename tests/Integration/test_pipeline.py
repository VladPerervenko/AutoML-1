import os
import pytest
import shutil
import unittest
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
                            optuna_time_budget=10,
                            plot_eda=False,
                            process_data=True,
                            validate_result=True,
                            )
        pipeline.fit(self.r_data)

        # Tests
        assert os.path.exists('AutoReg')
        # assert os.path.exists('AutoReg/EDA')
        assert os.path.exists('AutoReg/Data')
        assert os.path.exists('AutoReg/Features')
        assert os.path.exists('AutoReg/Production')
        assert os.path.exists('AutoReg/Results.csv')

        # Pipeline Prediction
        prediction = pipeline.predict(self.r_data)
        assert len(prediction.shape) == 1
        assert r2_score(self.r_data['target'], prediction) > 0

    def test_classification(self):
        pipeline = Pipeline('target',
                            project='AutoClass',
                            mode='classification',
                            objective='neg_log_loss',
                            optuna_time_budget=10,
                            grid_search_iterations=0,
                            plot_eda=True,
                            process_data=True,
                            validate_result=True,
                            )
        pipeline.fit(self.c_data)

        # Tests
        assert os.path.exists('AutoClass')
        assert os.path.exists('AutoClass/EDA')
        assert os.path.exists('AutoClass/Data')
        assert os.path.exists('AutoClass/Features')
        assert os.path.exists('AutoClass/Production')
        assert os.path.exists('AutoClass/Results.csv')

        # Pipeline Prediction
        prediction = pipeline.predict_proba(self.c_data)
        assert log_loss(self.c_data['target'], prediction) > -0.5


@pytest.fixture(scope="session", autouse=True)
def teardown():
    yield
    if os.path.exists('AutoClass'):
        shutil.rmtree('AutoClass')
    if os.path.exists('AutoReg'):
        shutil.rmtree('AutoReg')
