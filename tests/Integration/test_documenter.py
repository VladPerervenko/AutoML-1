import os
import shutil
import pytest
import unittest
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.datasets import make_classification
from Amplo import Pipeline


# class TestDocumenting(unittest.TestCase):

    # def test_binary(self):
    #     x, y = make_classification(n_classes=2, n_informative=15)
    #     x = pd.DataFrame(x, columns=['Feature_{}'.format(i) for i in range(x.shape[1])])
    #     x['target'] = y
    #     pipeline = Pipeline('target',
    #                         project='Tritium',
    #                         device='PK350',
    #                         issue='DC Domes',
    #                         mode='classification',
    #                         objective='neg_log_loss',
    #                         feature_timeout=5,
    #                         grid_search_iterations=1,
    #                         grid_search_time_budget=10,
    #                         grid_search_candidates=2,
    #                         plot_eda=False,
    #                         process_data=False,
    #                         document_results=True)
    #     pipeline.fit(x)
    #     assert len(os.listdir('AutoML/Documentation/v0')) != 0

    # def test_multi(self):
    #     x, y = make_classification(n_classes=5, n_informative=15)
    #     x = pd.DataFrame(x, columns=['Feature_{}'.format(i) for i in range(x.shape[1])])
    #     x['target'] = y
    #     pipeline = Pipeline('target',
    #                         project='Tritium',
    #                         device='PK350',
    #                         issue='DC Domes',
    #                         mode='classification',
    #                         objective='neg_log_loss',
    #                         feature_timeout=5,
    #                         grid_search_iterations=1,
    #                         grid_search_time_budget=10,
    #                         grid_search_candidates=2,
    #                         plot_eda=False,
    #                         process_data=False,
    #                         document_results=True)
    #     pipeline.fit(x)
    #     assert len(os.listdir('AutoML/Documentation/')) != 0
#
    # def test_regression(self):
    #     x, y = make_regression()
    #     x = pd.DataFrame(x, columns=['Feature_{}'.format(i) for i in range(x.shape[1])])
    #     x['target'] = y
    #     pipeline = Pipeline('target',
    #                         project='Tritium',
    #                         device='PK350',
    #                         issue='DC Domes',
    #                         mode='regression',
    #                         objective='r2',
    #                         feature_timeout=5,
    #                         grid_search_iterations=1,
    #                         grid_search_time_budget=10,
    #                         grid_search_candidates=2,
    #                         plot_eda=False,
    #                         process_data=False,
    #                         document_results=True)
    #     pipeline.fit(x)
    #     assert len(os.listdir('AutoML/Documentation/')) != 0
#
#
# @pytest.fixture(scope='session', autouse=True)
# def teardown():
#     yield
#     for i in range(3):
#         if os.path.exists('doc{}'.format(i)):
#             shutil.rmtree('doc{}'.format(i))
