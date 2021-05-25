import numpy as np
import pandas as pd


class Sequence:

    def __init__(self, back=0, forward=0, shift=0, diff='none'):
        """
        Sequencer. Sequences and differentiate data.
        Scenarios:
        - Sequenced I/O                     --> back & forward in int or list of ints
        - Sequenced input, single output    --> back: int, forward: list of ints
        - Single input, single output       --> back & forward =

        :param back: Int or List[int]: input indices to include
        :param forward: Int or List[int]: output indices to include
        :param shift: Int: If there is a shift between input and output
        :param diff: differencing algo, pick between 'none', 'diff', 'log_diff', 'fractional' (no revert)
        """
        if type(back) == int:
            back = np.linspace(0, back, back+1).astype('int')
            self.inputDType = 'int'
        elif type(back) == list:
            back = np.array(back)
            self.inputDType = 'list'
        else:
            raise ValueError('Back needs to be int or list(int)')
        if type(forward) == int:
            self.outputDType = 'int'
            forward = np.linspace(0, forward, forward+1).astype('int')
        elif type(forward) == list:
            self.outputDType = 'list'
            forward = np.array(forward)
        else:
            raise ValueError('Forward needs to be int or list(int)')
        self.backVec = back
        self.foreVec = forward
        self.nBack = len(back)
        self.nFore = len(forward)
        self.foreRoll = np.roll(self.foreVec, 1)
        self.foreRoll[0] = 0
        self.mBack = max(back) - 1
        self.mFore = max(forward) - 1
        self.shift = shift
        self.diff = diff
        self.samples = 1
        if diff != 'none':
            self.samples = 0
            self.nBack -= 1
        if diff not in ['none', 'diff', 'log_diff']:
            raise ValueError('Type should be in [None, diff, log_diff, fractional]')

    def convert(self, x, y):
        if isinstance(x, pd.DataFrame) or isinstance(x, pd.core.series.Series):
            assert isinstance(y, pd.DataFrame) or isinstance(y, pd.core.series.Series), \
                'Input and Output need to be the same data type.'
            return self.convert_pandas(x, y)
        elif isinstance(x, np.ndarray):
            assert isinstance(y, np.ndarray), 'Input and Output need to be the same data type.'
            return self.convert_numpy(x, y)
        else:
            TypeError('Input & Output need to be same datatype, either Numpy or Pandas.')

    def convert_numpy(self, x, y):
        # todo implement flat
        if x.ndim == 1:
            x = x.reshape((-1, 1))
        if y.ndim == 1:
            y = y.reshape((-1, 1))
        samples = len(x) - self.mBack - self.mFore - self.shift - 1
        features = len(x[0])
        input_sequence = np.zeros((samples, self.nBack, features))
        output_sequence = np.zeros((samples, self.nFore))
        if self.diff == 'none':
            for i in range(samples):
                input_sequence[i] = x[i + self.backVec]
                output_sequence[i] = y[i - 1 + self.mBack + self.shift + self.foreVec].reshape((-1))
            return input_sequence, output_sequence
        elif self.diff[-4:] == 'diff':
            if self.diff == 'log_diff':
                x = np.log(x)
                y = np.log(y)
            if (self.backVec == 0).all():
                self.backVec = np.array([0, 1])
            for i in range(samples):
                input_sequence[i] = x[i + self.backVec[1:]] - x[i + self.backVec[:-1]]
                output_sequence[i] = (y[i + self.mBack + self.shift + self.foreVec] -
                                      y[i + self.mBack + self.shift + self.foreRoll]).reshape((-1))
            return input_sequence, output_sequence

    def convert_pandas(self, x, y):
        # Check inputs
        if isinstance(x, pd.core.series.Series):
            x = x.to_frame()
        if isinstance(y, pd.core.series.Series):
            y = y.to_frame()
        assert len(x) == len(y)
        assert isinstance(x, pd.DataFrame)
        assert isinstance(y, pd.DataFrame)

        # Keys
        input_keys = x.keys()
        output_keys = y.keys()

        # No Differencing
        if self.diff == 'none':
            # Input
            for lag in self.backVec:
                keys = [key + '_' + str(lag) for key in input_keys]
                x[keys] = x[input_keys].shift(lag)

            # Output
            for shift in self.foreVec:
                keys = [key + '_' + str(shift) for key in output_keys]
                y[keys] = y[keys].shift(-shift)

        # With differencing
        elif self.diff[-4:] == 'diff':
            # Input
            for lag in self.backVec:
                # Shifted
                keys = [key + '_' + str(lag) for key in input_keys]
                x[keys] = x[input_keys].shift(lag)

                # differentiated
                d_keys = [key + '_d_' + str(lag) for key in input_keys]
                x[d_keys] = x[input_keys].shift(lag) - x[input_keys]

            # Output
            for shift in self.foreVec:
                # Only differentiated
                keys = [key + '_' + str(shift) for key in output_keys]
                y[keys] = y[output_keys].shift(shift) - y[output_keys]

        # Drop _0 (same as original)
        x = x.drop([key for key in x.keys() if '_0' in key], axis=1)
        y = y.drop([key for key in y.keys() if '_0' in key], axis=1)

        # Return (first lags are NaN, last shifts are NaN
        return x.iloc[lag:-shift if shift > 0 else None], y.iloc[lag:-shift if shift > 0 else None]

    def revert(self, differentiated, original):
        # Revert the sequencing loop: d = np.h stack((d[0], d[0] + np.cum sum(dd)))
        if self.nFore == 1:
            differentiated = differentiated.reshape((-1, 1))
        y = np.zeros_like(differentiated)
        if self.diff == 'none':
            return
        if self.diff == 'log_diff':
            for i in range(self.nFore):
                y[:, i] = np.log(original[self.mBack + self.foreRoll[i]:-self.foreVec[i]]) + differentiated[:, i]
            return np.exp(y)
        if self.diff == 'diff':
            for i in range(self.nFore):
                y[:, i] = original[self.mBack + self.foreRoll[i]:-self.foreVec[i]] + differentiated[:, i]
            return y
