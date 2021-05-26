import catboost
from Amplo.Classifiers import BaseClassifier


class CatBoostClassifier(BaseClassifier):

    def __init__(self):
        super().__init__(
            catboost.CatBoostClassifier(),
            {'verbose': 0, 'n_estimators': 1000, 'allow_writing_files': False},
        )
