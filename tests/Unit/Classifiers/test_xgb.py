import unittest
import pandas as pd
from sklearn.datasets import make_classification
from Amplo.Classifiers import XGBClassifier
from .test_classifier import TestClassifier


class TestXGBClassifier(unittest.TestCase, TestClassifier):

    @classmethod
    def setUpClass(cls):
        cls.model = XGBClassifier()
        x, y = make_classification(n_classes=5, n_informative=15)
        cls.x, cls.y = pd.DataFrame(x), pd.Series(y)
