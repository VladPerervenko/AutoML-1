from catboost import CatBoostClassifier
from BaseClassifier import BaseClassifier


class CatboostClassifier(BaseClassifier):

    def __init__(self):
        super().__init__(
            CatBoostClassifier,
            {'verbose': 0, 'n_estimators': 1000, 'allow_writing_files': False},
        )
