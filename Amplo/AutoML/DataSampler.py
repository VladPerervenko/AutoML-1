import pandas as pd


class DataSampler:

    def __init__(self):
        '''
        Trial and error various strategies:
        - Undersample all to minority
        - Upsample all to majority
        - Combination of the two, with say 1k, 5k, 25k, 50k, 75k and 100k samples
        This sensitivity is problem dependent, but has major consequence for hyper
        parameter optimization. Need to get it right.
        https://imbalanced-learn.org/stable/references/index.html#api
        '''
        pass
