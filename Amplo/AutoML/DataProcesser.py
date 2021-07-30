import re
import os
import warnings
import numpy as np
import pandas as pd
from ..Utils import clean_keys


class DataProcesser:
    # todo implement data type detector

    def __init__(self,
                 target: str = None,
                 num_cols: list = None,
                 date_cols: list = None,
                 cat_cols: list = None,
                 missing_values: str = 'interpolate',
                 outlier_removal: str = 'clip',
                 z_score_threshold: int = 4,
                 folder: str = '',
                 version: int = 1,
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
        outlier_removal str: How to deal with outliers ('clip', 'quantiles', 'z-score' or 'none')
        z_score_threshold int: If outlierRemoval='z-score', the threshold is adaptable, default=4.
        folder str: Directory for storing the output files
        version int: Versioning the output files
        mode str: classification / regression
        """
        # Tests
        mis_values_algo = ['remove_rows', 'remove_cols', 'interpolate', 'mean', 'zero']
        assert missing_values in mis_values_algo, \
            'Missing values algorithm not implemented, pick from {}'.format(', '.join(mis_values_algo))
        out_rem_algo = ['quantiles', 'z-score', 'clip', 'none']
        assert outlier_removal in out_rem_algo, \
            'Outlier Removal algorithm not implemented, pick from {}'.format(', '.join(out_rem_algo))

        # Make Folder
        self.folder = folder if len(folder) == 0 or folder[-1] == '/' else folder + '/'
        if len(self.folder) != 0 and not os.path.exists(self.folder):
            os.makedirs(self.folder)

        # Arguments
        self.version = version
        self.target = target if target is None else re.sub("[^a-z0-9]", '_', target.lower())
        self.num_cols = [] if num_cols is None else [re.sub('[^a-z0-9]', '_', nc.lower()) for nc in num_cols]
        self.cat_cols = [] if cat_cols is None else [re.sub('[^a-z0-9]', '_', cc.lower()) for cc in cat_cols]
        self.date_cols = [] if date_cols is None else [re.sub('[^a-z0-9]', '_', dc.lower()) for dc in date_cols]
        if self.target in self.num_cols:
            self.num_cols.remove(self.target)

        # Algorithms
        self.missing_values = missing_values
        self.outlier_removal = outlier_removal
        self.z_score_threshold = z_score_threshold

        # Fitted Settings
        self.dummies = {}
        self._q1 = None
        self._q3 = None
        self._means = None
        self._stds = None

        # Info for Documenting
        self.is_fitted = False
        self.removedDuplicateRows = 0
        self.removedDuplicateColumns = 0
        self.removedOutliers = 0
        self.imputedMissingValues = 0
        self.removedConstantColumns = 0

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fits this data cleaning module and returns the transformed data.


        Parameters
        ----------
        data [pd.DataFrame]: Input data

        Returns
        -------
        data [pd.DataFrame]: Cleaned input data
        """
        print('[AutoML] Data Cleaning Started, ({} x {}) samples'.format(len(data), len(data.keys())))

        # Clean Keys
        data = clean_keys(data)

        # Remove Duplicates
        data = self.remove_duplicates(data)

        # Convert data types
        data = self.convert_data_types(data, fit_categorical=True)

        # Remove outliers
        data = self.remove_outliers(data, fit=True)

        # Remove missing values
        data = self.remove_missing_values(data)

        # Remove Constants
        data = self.remove_constants(data)

        # Finish
        self.is_fitted = True
        print('[AutoML] Processing completed, ({} x {}) samples returned'.format(len(data), len(data.keys())))
        return data

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Function that takes existing settings (including dummies), and transforms new data.

        Parameters
        ----------
        data [pd.DataFrame]: Input data

        Returns
        -------
        data [pd.DataFrame]: Cleaned input data
        """
        assert self.is_fitted, "Transform only available for fitted objects, run .fit_transform() first."

        # Clean Keys
        data = clean_keys(data)

        # Remove duplicates
        data = self.remove_duplicates(data)

        # Convert data types
        data = self.convert_data_types(data, fit_categorical=False)

        # Remove outliers
        data = self.remove_outliers(data, fit=False)

        # Remove missing values
        data = self.remove_missing_values(data)

        return data

    def get_settings(self) -> dict:
        """
        Get settings to recreate fitted object.
        """
        assert self.is_fitted, "Object not yet fitted."
        return {
            'num_cols': self.num_cols,
            'date_cols': self.date_cols,
            'cat_cols': self.cat_cols,
            'missing_values': self.missing_values,
            'outlier_removal': self.outlier_removal,
            'z_score_threshold': self.z_score_threshold,
            '_means': None if self._means is None else self._means.to_json(),
            '_stds': None if self._stds is None else self._stds.to_json(),
            '_q1': None if self._q1 is None else self._q1.to_json(),
            '_q3': None if self._q3 is None else self._q1.to_json(),
            'dummies': self.dummies
        }

    def load_settings(self, settings: dict) -> None:
        """
        Loads settings from dictionary and recreates a fitted object
        """
        self.num_cols = settings['num_cols']
        self.cat_cols = settings['cat_cols']
        self.date_cols = settings['date_cols']
        self.missing_values = settings['missing_values']
        self.outlier_removal = settings['outlier_removal']
        self.z_score_threshold = settings['z_score_threshold']
        self._means = None if settings['_means'] is None else pd.read_json(settings['_means'])
        self._stds = None if settings['_stds'] is None else pd.read_json(settings['_stds'])
        self._q1 = None if settings['_q1'] is None else pd.read_json(settings['_q1'])
        self._q3 = None if settings['_q3'] is None else pd.read_json(settings['_q3'])
        self.dummies = settings['dummies']
        self.is_fitted = True

    def convert_data_types(self, data: pd.DataFrame, fit_categorical: bool = True) -> pd.DataFrame:
        """
        Cleans up the data types of all columns.

        Parameters
        ----------
        data [pd.DataFrame]: Input data
        fit_categorical [bool]: Whether to fit the categorical encoder

        Returns
        -------
        data [pd.DataFrame]: Cleaned input data
        """
        # Datetime columns
        for key in self.date_cols:
            data.loc[:, key] = pd.to_datetime(data[key], errors='coerce', infer_datetime_format=True, utc=True)

        # Numerical columns
        for key in [key for key in data.keys() if key not in self.date_cols and key not in self.cat_cols]:
            data.loc[:, key] = pd.to_numeric(data[key], errors='coerce', downcast='float')

        # Categorical columns
        if fit_categorical:
            data = self._fit_cat_cols(data)
        else:
            assert self.is_fitted, ".convert_data_types() was called with fit_categorical=False, while categorical " \
                                   "encoder is not yet fitted."
            data = self._transform_cat_cols(data)

        return data

    def _fit_cat_cols(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encoding categorical variables always needs a scheme. This fits the scheme.
        """
        for key in self.cat_cols:
            dummies = pd.get_dummies(data[key], prefix=key, drop_first=True)
            data = data.drop(key, axis=1).join(dummies)
            self.dummies[key] = dummies.keys().tolist()
        return data

    def _transform_cat_cols(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Converts categorical variables according to fitted scheme.
        """
        for key, value in self.dummies.items():
            dummies = [i[len(key) + 1:] for i in value]
            data[value] = np.equal.outer(data[key].values, dummies) * 1
            data = data.drop(key, axis=1)
        return data

    def remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes duplicate columns and rows.

        Parameters
        ----------
        data [pd.DataFrame]: Input data

        Returns
        -------
        data [pd.DataFrame]: Cleaned input data
        """
        # Note down
        rows, columns = len(data), len(data.keys())

        # Remove Duplicates
        data = data.drop_duplicates()
        data = data.loc[:, ~data.columns.duplicated()]

        # Note
        self.removedDuplicateColumns = len(data.keys()) - columns
        self.removedDuplicateRows = len(data) - rows

        return data

    def remove_constants(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes constant columns
        """
        columns = len(data.keys())

        # Remove Constants
        data = data.drop(columns=data.columns[data.nunique() == 1])

        # Note
        self.removedConstantColumns = columns - len(data.keys())

        return data

    def fit_outliers(self, data: pd.DataFrame) -> int:
        """
        Checks outliers
        """
        # With quantiles
        if self.outlier_removal == 'quantiles':
            self._q1 = data.quantile(0.25)
            self._q3 = data.quantile(0.75)
            return (data > self._q3).sum().sum() + (data < self._q1).sum().sum()

        # By z-score
        elif self.outlier_removal == 'z-score':
            self._means = data.mean(skipna=True, numeric_only=True)
            self._stds = data.std(skipna=True, numeric_only=True)
            self._stds[self._stds == 0] = 1
            z_score = (data - self._means) / self._stds
            return (z_score > self.z_score_threshold).sum().sum()

        # By clipping
        elif self.outlier_removal == 'clip':
            return (data > 1e12).sum().sum() + (data < -1e12).sum().sum()

    def remove_outliers(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Removes outliers
        """
        # Check if needs fitting
        if fit:
            self.removedOutliers = self.fit_outliers(data)
        else:
            assert self.is_fitted, ".remove_outliers() is called with fit=False, yet the object isn't fitted yet."

        # With Quantiles
        if self.outlier_removal == 'quantiles':
            data = data.mask(data < self._q1)
            data = data.mask(data > self._q3)

        # With z-score
        elif self.outlier_removal == 'z-score':
            z_score = abs((data - self._means) / self._stds)
            data = data.mask(z_score > self.z_score_threshold)

        # With clipping
        elif self.outlier_removal == 'clip':
            data = data.clip(lower=-1e12, upper=1e12)

        return data

    def remove_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values (infinities and 'not a number's)
        """
        # Note
        self.imputedMissingValues = np.sum(np.isnan(data.values)) + np.sum(data.values == np.Inf) + np.sum(data.values
                                                                                                           == -np.Inf)
        # Replace infinities
        data = data.replace([np.inf, -np.inf], np.nan)

        # Removes all rows with missing values
        if self.missing_values == 'remove_rows':
            data = data[data.isna().sum(axis=1) == 0]

        # Removes all columns with missing values
        elif self.missing_values == 'remove_cols':
            data = data.loc[:, data.isna().sum(axis=0) == 0]

        # Fills all missing values with zero
        elif self.missing_values == 'zero':
            data = data.fillna(0)

        # Linearly interpolates missing values
        elif self.missing_values == 'interpolate':
            # Not recommended when columns are present with >10% missing values
            if (data.isna().sum() / len(data) > 0.1).any():
                warnings.warn('[AutoML] Strongly recommend to NOT use interpolation for features with more than 10% '
                              'missing values')
                # Get all non-date_cols & interpolate
            ik = np.setdiff1d(data.keys().to_list(), self.date_cols)
            data[ik] = data[ik].interpolate(limit_direction='both')
            # Fill rest (date cols)
            if data.isna().sum().sum() != 0:
                data = data.fillna(0)

        # Fill missing values with column mean
        elif self.missing_values == 'mean':
            data = data.fillna(data.mean())

        return data
