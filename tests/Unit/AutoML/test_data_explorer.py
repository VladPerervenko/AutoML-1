import shutil
import unittest
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from Amplo.AutoML import DataExploring


class TestDataExploring(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.class_x, cls.class_y = load_iris(return_X_y=True, as_frame=True)
        cls.reg_x, cls.reg_y = load_boston(return_X_y=True)

    def test_regression(self):
        eda = DataExploring(self.reg_x, y=self.reg_y, mode='regression')
        eda.run()

    def test_classification(self):
        eda = DataExploring(self.class_x, y=self.class_y, mode='classification')
        eda.run()


def teardown_func():
    shutil.rmtree('EDA')
