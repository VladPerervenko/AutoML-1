import catboost
from Amplo.Classifiers import BaseClassifier


class CatBoostClassifier(BaseClassifier):

    def __init__(self, params=None):
        """
        Catboost Classifier wrapper
        """
        default = {'verbose': 0, 'n_estimators': 1000, 'allow_writing_files': False}
        super().__init__(default, params)
        self.model = catboost.CatBoostClassifier()
        self.hasPredictProba = True
        self.set_params(self.params)
