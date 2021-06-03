import re
import os
import copy
import inspect
import numpy as np
import pandas as pd
from ..Utils import clean_keys


class DataProcessing:
    # todo implement data type detector

    def __init__(self,
                 target=None,
                 num_cols=None,
                 date_cols=None,
                 cat_cols=None,
                 missing_values='interpolate',
                 outlier_removal='clip',
                 z_score_threshold=4,
                 folder='',
                 version=1,
                 mode='regression',
                 ):
        """
        Preprocessing Class. Deals with Outliers, Missing Values, duplicate rows, data types (floats, categorical and
        dates), Not a Numbers, Infinities.

        :param target: Column name of target variable
        :param num_cols: Numerical columns, all parsed to integers and floats
        :param date_cols: Date columns, all parsed to pd.datetime format
        :param cat_cols: Categorical Columns. Currently all one-hot encoded.
        :param missing_values: How to deal with missing values ('remove', 'interpolate' or 'mean')
        :param outlier_removal: How to deal with outliers ('boxplot', 'z-score' or 'none')
        :param z_score_threshold: If outlierRemoval='z-score', the threshold is adaptable, default=4.
        :param folder: Directory for storing the output files
        :param version: Versioning the output files
        """
        # Parameters
        self.folder = folder if len(folder) == 0 or folder[-1] == '/' else folder + '/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.version = version
        self.target = re.sub("[^a-z0-9]", '_', target.lower())
        self.mode = mode
        self.numCols = [] if num_cols is None else [re.sub('[^a-z0-9]', '_', nc.lower()) for nc in num_cols]
        self.catCols = [] if cat_cols is None else [re.sub('[^a-z0-9]', '_', cc.lower()) for cc in cat_cols]
        self.dateCols = [] if date_cols is None else [re.sub('[^a-z0-9]', '_', dc.lower()) for dc in date_cols]
        if self.target in self.numCols:
            self.numCols.remove(self.target)

        # Variables
        self.scaler = None
        self.oScaler = None

        # Algorithms
        missing_values_implemented = ['remove_rows', 'remove_cols', 'interpolate', 'mean', 'zero']
        outlier_removal_implemented = ['boxplot', 'z-score', 'clip', 'none']
        if outlier_removal not in outlier_removal_implemented:
            raise ValueError(
                "Outlier Removal Algorithm not implemented. Should be in " + str(outlier_removal_implemented))
        if missing_values not in missing_values_implemented:
            raise ValueError("Missing Values Algorithm not implemented. Should be in " +
                             str(missing_values_implemented))
        self.missingValues = missing_values
        self.outlierRemoval = outlier_removal
        self.zScoreThreshold = z_score_threshold

        # Statistics for Documenting
        self.removedDuplicateRows = 0
        self.removedDuplicateColumns = 0
        self.imputedMissingValues = 0
        self.removedConstantColumns = 0

    def clean(self, data):
        print('[Data] Data Cleaning Started, ({} x {}) samples'.format(len(data), len(data.keys())))
        if len(data[self.target].unique()) == 2:
            self.mode = 'classification'

        # Note down
        rows, columns = len(data), len(data.keys())

        # Clean Keys
        data = clean_keys(data)

        # Remove Duplicates
        data = self.remove_duplicates(data)
        # Note
        self.removedDuplicateColumns = len(data.keys()) - columns
        self.removedDuplicateRows = len(data) - rows

        # Convert data types
        data = self.convert_data_types(data)
        # Remove outliers
        data = self.remove_outliers(data)
        # Note
        self.imputedMissingValues = np.sum(np.isnan(data.values)) + np.sum(data.values == np.Inf) + \
                                    np.sum(data.values == -np.Inf)

        # Remove missing values
        data = self.remove_missing_values(data)
        # Note
        columns = len(data.keys())

        # Remove Constants
        data = self.remove_constants(data)
        # Note
        self.removedConstantColumns = len(data.keys()) - columns

        # Finish
        self._store(data)
        print('[Data] Processing completed, ({} x {}) samples returned'.format(len(data), len(data.keys())))
        return data

    def convert_data_types(self, data):
        # Convert Data Types
        for key in self.dateCols:
            data.loc[:, key] = pd.to_datetime(data[key], errors='coerce', infer_datetime_format=True, utc=True)
        for key in [key for key in data.keys() if key not in self.dateCols and key not in self.catCols]:
            data.loc[:, key] = pd.to_numeric(data[key], errors='coerce', downcast='float')
        for key in self.catCols:
            if key in data.keys():
                dummies = pd.get_dummies(data[key])
                for dummy_key in dummies.keys():
                    dummies = dummies.rename(
                        columns={dummy_key: key + '_' + re.sub('[^a-z0-9]', '_', str(dummy_key).lower())})
                data = data.drop(key, axis=1).join(dummies)
        return data

    @staticmethod
    def remove_duplicates(data):
        # Remove Duplicates
        data = data.drop_duplicates()
        data = data.loc[:, ~data.columns.duplicated()]
        return data

    @staticmethod
    def remove_constants(data):
        # Remove Constants
        data = data.drop(columns=data.columns[data.nunique() == 1])
        return data

    def remove_outliers(self, data):
        # Remove Outliers
        if self.outlierRemoval == 'boxplot':
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            for key in q1.keys():
                data.loc[data[key] < q1[key] - 1.5 * (q3[key] - q1[key]), key] = np.nan
                data.loc[data[key] > q3[key] + 1.5 * (q3[key] - q1[key]), key] = np.nan
        elif self.outlierRemoval == 'z-score':
            z_score = (data - data.mean(skipna=True, numeric_only=True)) \
                     / data.std(skipna=True, numeric_only=True)
            data[z_score > self.zScoreThreshold] = np.nan
        elif self.outlierRemoval == 'clip':
            data = data.clip(lower=-1e12, upper=1e12)
        return data

    def remove_missing_values(self, data):
        # Remove Missing Values
        data = data.replace([np.inf, -np.inf], np.nan)
        if self.missingValues == 'remove_rows':
            data = data[data.isna().sum(axis=1) == 0]
        elif self.missingValues == 'remove_cols':
            data = data.loc[:, data.isna().sum(axis=0) == 0]
        elif self.missingValues == 'zero':
            data = data.fillna(0)
        elif self.missingValues == 'interpolate':
            ik = np.setdiff1d(data.keys().to_list(), self.dateCols)
            data[ik] = data[ik].interpolate(limit_direction='both')
            if data.isna().sum().sum() != 0:
                data = data.fillna(0)
        elif self.missingValues == 'mean':
            data = data.fillna(data.mean())
        return data

    def _store(self, data):
        # Store cleaned data
        data.to_csv(self.folder + 'Cleaned_v{}.csv'.format(self.version), index_label='index')

    def export_function(self):
        function_strings = [
            inspect.getsource(clean_keys),
            inspect.getsource(self.remove_duplicates).replace('self.', ''),
            inspect.getsource(self.convert_data_types).replace('self.', ''),
            inspect.getsource(self.remove_outliers).replace('self.', ''),
            inspect.getsource(self.remove_missing_values).replace('self.', ''),
        ]
        function_strings = '\n'.join([k[k.find('\n'): k.rfind('\n', 0, k.rfind('\n'))] for k in function_strings])

        return """
            #################
            # Data Cleaning #
            #################
            # Copy vars
            catCols, dateCols, target = {}, {}, '{}'
            outlierRemoval, missingValues, zScoreThreshold = '{}', '{}', '{}'
    """.format(self.catCols, self.dateCols, self.target, self.outlierRemoval, self.missingValues,
               self.zScoreThreshold) + function_strings
