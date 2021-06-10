import re
import os
import inspect
import textwrap
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
        Preprocessing Class. Cleans a dataset into a workable format.
        Deals with Outliers, Missing Values, duplicate rows, data types (floats, categorical and
        dates), Not a Numbers, Infinities.

        Parameters
        ----------
        target str: Column name of target variable
        num_cols list: Numerical columns, all parsed to integers and floats
        date_cols list: Date columns, all parsed to pd.datetime format
        cat_cols list: Categorical Columns. Currently all one-hot encoded.
        missing_values str: How to deal with missing values ('remove', 'interpolate' or 'mean')
        outlier_removal str: How to deal with outliers ('clip', 'boxplot', 'z-score' or 'none')
        z_score_threshold int: If outlierRemoval='z-score', the threshold is adaptable, default=4.
        folder str: Directory for storing the output files
        version int: Versioning the output files
        mode str: classification / regression
        """
        # Tests
        assert isinstance(target, str)
        assert isinstance(num_cols, list) or num_cols is None
        assert isinstance(date_cols, list) or date_cols is None
        assert isinstance(cat_cols, list) or cat_cols is None
        assert isinstance(missing_values, str)
        mis_values_algo = ['remove_rows', 'remove_cols', 'interpolate', 'mean', 'zero']
        assert missing_values in mis_values_algo, \
            'Missing values algorithm not implemented, pick from {}'.format(', '.join(mis_values_algo))
        out_rem_algo = ['boxplot', 'z-score', 'clip', 'none']
        assert outlier_removal in out_rem_algo, \
            'Outlier Removal algorithm not implemented, pick from {}'.format(', '.join(out_rem_algo))
        assert isinstance(z_score_threshold, int)
        assert isinstance(folder, str)
        assert isinstance(version, int)
        assert mode in ['classification', 'regression'], 'Mode needs to be classification or regression'

        # Parameters
        self.folder = folder if len(folder) == 0 or folder[-1] == '/' else folder + '/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.version = version
        self.target = re.sub("[^a-z0-9]", '_', target.lower())
        self.mode = mode
        self.num_cols = [] if num_cols is None else [re.sub('[^a-z0-9]', '_', nc.lower()) for nc in num_cols]
        self.cat_cols = [] if cat_cols is None else [re.sub('[^a-z0-9]', '_', cc.lower()) for cc in cat_cols]
        self.date_cols = [] if date_cols is None else [re.sub('[^a-z0-9]', '_', dc.lower()) for dc in date_cols]
        if self.target in self.num_cols:
            self.num_cols.remove(self.target)

        # Variables
        self.scaler = None
        self.oScaler = None

        # Algorithms
        self.missing_values = missing_values
        self.outlier_removal = outlier_removal
        self.z_score_threshold = z_score_threshold

        # Info for Documenting
        self.removedDuplicateRows = 0
        self.removedDuplicateColumns = 0
        self.removedOutliers = 0
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
        self.removedOutliers = self.check_outliers(data)
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
        for key in self.date_cols:
            data.loc[:, key] = pd.to_datetime(data[key], errors='coerce', infer_datetime_format=True, utc=True)
        for key in [key for key in data.keys() if key not in self.date_cols and key not in self.cat_cols]:
            data.loc[:, key] = pd.to_numeric(data[key], errors='coerce', downcast='float')
        for key in self.cat_cols:
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

    def check_outliers(self, data):
        if self.outlier_removal == 'boxplot':
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            return (data > q3).sum().sum() or (data < q1).sum().sum()
        elif self.outlier_removal == 'z-score':
            z_score = (data - data.mean(skipna=True, numeric_only=True)) \
                     / data.std(skipna=True, numeric_only=True)
            return (z_score > self.z_score_threshold).sum().sum()
        elif self.outlier_removal == 'clip':
            return (data > 1e12).sum().sum() + (data < -1e12).sum().sum()

    def remove_outliers(self, data):
        # Remove Outliers
        if self.outlier_removal == 'boxplot':
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            for key in q1.keys():
                data.loc[data[key] < q1[key] - 1.5 * (q3[key] - q1[key]), key] = np.nan
                data.loc[data[key] > q3[key] + 1.5 * (q3[key] - q1[key]), key] = np.nan
        elif self.outlier_removal == 'z-score':
            z_score = (data - data.mean(skipna=True, numeric_only=True)) \
                     / data.std(skipna=True, numeric_only=True)
            data[z_score > self.z_score_threshold] = np.nan
        elif self.outlier_removal == 'clip':
            data = data.clip(lower=-1e12, upper=1e12)
        return data

    def remove_missing_values(self, data):
        # Remove Missing Values
        data = data.replace([np.inf, -np.inf], np.nan)
        if self.missing_values == 'remove_rows':
            data = data[data.isna().sum(axis=1) == 0]
        elif self.missing_values == 'remove_cols':
            data = data.loc[:, data.isna().sum(axis=0) == 0]
        elif self.missing_values == 'zero':
            data = data.fillna(0)
        elif self.missing_values == 'interpolate':
            ik = np.setdiff1d(data.keys().to_list(), self.date_cols)
            data[ik] = data[ik].interpolate(limit_direction='both')
            if data.isna().sum().sum() != 0:
                data = data.fillna(0)
        elif self.missing_values == 'mean':
            data = data.fillna(data.mean())
        return data

    def _store(self, data):
        # Store cleaned data
        data.to_csv(self.folder + 'Cleaned_v{}.csv'.format(self.version), index_label='index')

    def export_function(self):
        duplicates_code = inspect.getsource(self.remove_duplicates)
        duplicates_code = duplicates_code[duplicates_code.find('\n')+1:]
        function_strings = [
            textwrap.indent(inspect.getsource(clean_keys), '    '),
            duplicates_code.replace('self.', ''),
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
        cat_cols, date_cols, target = {}, {}, '{}'
        outlier_removal, missing_values, z_score_threshold = '{}', '{}', '{}'
    """.format(self.cat_cols, self.date_cols, self.target, self.outlier_removal, self.missing_values,
               self.z_score_threshold) + function_strings
