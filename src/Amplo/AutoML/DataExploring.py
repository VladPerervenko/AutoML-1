import os
import copy
import ppscore
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import missingno
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier


class DataExploring:

    def __init__(self, data,
                 y=None,
                 plot_time_plots=True,
                 plot_box_plots=False,
                 plot_missing_values=True,
                 plot_seasonality=False,
                 plot_co_linearity=True,
                 plot_differencing=False,
                 plot_signal_correlations=False,
                 plot_feature_importance=True,
                 plot_scatter_plots=False,
                 differ=0,
                 pre_tag='',
                 max_samples=10000,
                 season_periods=None,
                 lags=60,
                 skip_completed=True,
                 folder='',
                 version=None):
        """
        Doing all the fun EDA in an automated manner :)
        """
        assert isinstance(data, pd.DataFrame)

        # Running booleans
        self.plotTimePlots = plot_time_plots
        self.plotBoxPlots = plot_box_plots
        self.plotMissingValues = plot_missing_values
        self.plotSeasonality = plot_seasonality
        self.plotCoLinearity = plot_co_linearity
        self.plotDifferencing = plot_differencing
        self.plotSignalCorrelations = plot_signal_correlations
        self.plotFeatureImportance = plot_feature_importance
        self.plotScatterPlots = plot_scatter_plots

        # Register data
        self.data = data.astype('float32').fillna(0)
        if y is not None:
            assert isinstance(y, pd.DataFrame) or isinstance(y, pd.Series)
            if isinstance(y, pd.DataFrame):
                y = y[y.keys()[0]]
            self.Y = y.astype('float32').fillna(0)
            if self.Y.nunique() == 2:
                print('[AutoML] Mode set to classification.')
                self.mode = 'classification'
                if set(self.Y.values) != {0, 1}:
                    assert 1 in self.Y.values, 'Ambiguous classes (either {0, 1} or {-1, 1})'
                    self.Y.loc[self.Y.values != 1] = 0
            else:
                print('[AutoML] Mode set to regression.')
                self.mode = 'regression'
        else:
            self.mode = None

        # General settings
        self.seasonPeriods = season_periods
        self.maxSamples = max_samples       # Time series
        self.differ = differ                # Correlations
        self.lags = lags                    # Correlations

        # Storage settings
        self.tag = pre_tag
        self.version = version if version is not None else 0
        self.folder = folder if folder == '' or folder[-1] == '/' else folder + '/'
        self.skip = skip_completed

        # Create Base folder
        if not os.path.exists(self.folder + 'EDA/'):
            os.mkdir(self.folder + 'EDA')
        self.folder += 'EDA/'
        self.run()

    def run(self):
        # Run all functions
        if self.mode == 'classification':
            self._run_classification()
        else:
            self._run_regression()

    def _run_classification(self):
        print('[EDA] Generating Missing Values Plot')
        self.missing_values()
        print('[EDA] Generating Time plots')
        self.time_plots()
        print('[EDA] Generating Box plots')
        self.box_plots()
        if self.Y is not None:
            print('[EDA] Generating SHAP plot')
            self.shap()
            print('[EDA] Generating Feature Ranking Plot')
            self.feature_ranking()
            print('[EDA] Predictive Power Score Plot')
            self.predictive_power_score()

    def _run_regression(self):
        print('[EDA] Generating Missing Values Plot')
        self.missing_values()
        print('[EDA] Generating Time plots')
        self.time_plots()
        print('[EDA] Generating Box plots')
        self.box_plots()
        self.seasonality()
        print('[EDA] Generating Co-linearity Plots')
        self.co_linearity()
        print('[EDA] Generating Diff Var Plot')
        self.differencing()
        print('[EDA] Generating ACF Plots')
        self.complete_auto_corr()
        print('[EDA] Generating PACF Plots')
        self.partial_auto_corr()
        if self.Y is not None:
            print('[EDA] Generating CCF Plots')
            self.cross_corr()
            print('[EDA] Generating Scatter plots')
            self.scatters()
            print('[EDA] Generating SHAP plot')
            self.shap()
            print('[EDA] Generating Feature Ranking Plot')
            self.feature_ranking()
            print('[EDA] Predictive Power Score Plot')
            self.predictive_power_score()

    def missing_values(self):
        if self.plotMissingValues:
            # Create folder
            if not os.path.exists(self.folder + 'MissingValues/'):
                os.mkdir(self.folder + 'MissingValues/')

            # Skip if exists
            if self.tag + 'v%i.png' % self.version in os.listdir(self.folder + 'MissingValues/'):
                return

            # Plot
            ax = missingno.matrix(self.data, figsize=[24, 16])
            fig = ax.get_figure()
            fig.savefig(self.folder + 'MissingValues/v%i.png' % self.version)

    def box_plots(self):
        if self.plotBoxPlots:
            # Create folder
            if not os.path.exists(self.folder + 'BoxPlots/v%i/' % self.version):
                os.makedirs(self.folder + 'BoxPlots/v%i/' % self.version)

            # Iterate through vars
            for key in tqdm(self.data.keys()):

                # Skip if existing
                if self.tag + key + '.png' in os.listdir(self.folder + 'BoxPlots/v%i/' % self.version):
                    continue

                # Figure prep
                fig = plt.figure(figsize=[24, 16])
                plt.title(key)

                # Classification
                if self.mode == 'classification':
                    plt.boxplot([self.data.loc[self.Y == 1, key], self.data.loc[self.Y == -1, key]], labels=['Faulty',
                                                                                                             'Healthy'])
                    plt.legend(['Faulty', 'Healthy'])

                # Regression
                if self.mode == 'regression':
                    plt.boxplot(self.data[key])

                # Store & Close
                fig.savefig(self.folder + 'BoxPlots/v%i/' % self.version + self.tag + key + '.png',
                            format='png', dpi=300)
                plt.close()

    def time_plots(self):
        if self.plotTimePlots:
            # Create folder
            if not os.path.exists(self.folder + 'TimePlots/v%i/' % self.version):
                os.makedirs(self.folder + 'TimePlots/v%i/' % self.version)

            # Set matplot limit
            matplotlib.use('Agg')
            matplotlib.rcParams['agg.path.chunksize'] = 200000

            # Undersample
            ind = np.linspace(0, len(self.data) - 1, self.maxSamples).astype('int')
            data, y = self.data.iloc[ind], self.Y.iloc[ind]

            # Iterate through features
            for key in tqdm(data.keys()):
                # Skip if existing
                if self.tag + key + '.png' in os.listdir(self.folder + 'TimePlots/v%i/' % self.version):
                    continue

                # Figure preparation
                fig = plt.figure(figsize=[24, 16])
                plt.title(key)

                # Plot
                if self.mode == 'classification':
                    cm = plt.get_cmap('bwr')
                else:
                    cm = plt.get_cmap('summer')
                nm_output = (y - y.min()) / (y.max() - y.min())
                plt.scatter(data.index, data[key], c=cm(nm_output), alpha=0.3)

                # Store & Close
                fig.savefig(self.folder + 'TimePlots/v%i/' % self.version + self.tag + key + '.png',
                            format='png', dpi=100)
                plt.close(fig)

    def seasonality(self):
        if self.plotSeasonality:
            # Create folder
            if not os.path.exists(self.folder + 'Seasonality/'):
                os.mkdir(self.folder + 'Seasonality/')

            # Iterate through features
            for key in tqdm(self.data.keys()):
                for period in self.seasonPeriods:
                    if self.tag + key + '_v%i.png' % self.version in os.listdir(self.folder + 'Seasonality/'):
                        continue
                    seasonality = STL(self.data[key], period=period).fit()
                    fig = plt.figure(figsize=[24, 16])
                    plt.plot(range(len(self.data)), self.data[key])
                    plt.plot(range(len(self.data)), seasonality)
                    plt.title(key + ', period=' + str(period))
                    fig.savefig(self.folder + 'Seasonality/' + self.tag + str(period)+'/'+key +
                                '_v%i.png' % self.version, format='png', dpi=300)
                    plt.close()

    def co_linearity(self):
        if self.plotCoLinearity:
            # Create folder
            if not os.path.exists(self.folder + 'CoLinearity/v%i/'):
                os.makedirs(self.folder + 'CoLinearity/v%i/')

            # Skip if existing
            if self.tag + 'MinimumRepresentation.png' in os.listdir(self.folder + 'Colinearity/v%i/' % self.version):
                return

            # Plot threshold matrix
            threshold = 0.95
            fig = plt.figure(figsize=[24, 16])
            plt.title('Co-linearity matrix, threshold %.2f' % threshold)
            sns.heatmap(abs(self.data.corr()) < threshold, annot=False, cmap='Greys')
            fig.savefig(self.folder + 'CoLinearity/v%i/' % self.version + self.tag + 'Matrix.png',
                        format='png', dpi=300)

            # Minimum representation
            corr_mat = self.data.corr().abs()
            upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype('bool'))
            col_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
            minimal_rep = self.data.drop(self.data[col_drop], axis=1)
            fig = plt.figure(figsize=[24, 16])
            sns.heatmap(abs(minimal_rep.corr()) < threshold, annot=False, cmap='Greys')
            fig.savefig(self.folder + 'CoLinearity/v%i/' % self.version + self.tag + 'Minimum_Representation.png',
                        format='png', dpi=300)

    def differencing(self):
        if self.plotDifferencing:
            # Create folder
            if not os.path.exists(self.folder + 'Lags/'):
                os.mkdir(self.folder + 'Lags/')

            # Skip if existing
            if self.tag + 'Variance.png' in os.listdir(self.folder + 'Lags/'):
                return

            # Setup
            max_lags = 4
            var_vec = np.zeros((max_lags, len(self.data.keys())))
            diff_data = self.data / np.sqrt(self.data.var())

            # Calculate variance per lag
            for i in range(max_lags):
                var_vec[i, :] = diff_data.var()
                diff_data = diff_data.diff(1)[1:]

            # Plot
            fig = plt.figure(figsize=[24, 16])
            plt.title('Variance for different lags')
            plt.plot(var_vec)
            plt.xlabel('Lag')
            plt.yscale('log')
            plt.ylabel('Average variance')
            fig.savefig(self.folder + 'Lags/' + self.tag + 'Variance.png', format='png', dpi=300)

    def complete_auto_corr(self):
        if self.plotSignalCorrelations:
            # Create folder
            if not os.path.exists(self.folder + 'Correlation/ACF/'):
                os.makedirs(self.folder + 'Correlation/ACF/')

            # Difference data
            diff_data = copy.copy(self.data)
            for i in range(self.differ):
                diff_data = diff_data.diff(1)[1:]

            # Iterate through features
            for key in tqdm(self.data.keys()):
                # Skip if existing
                if self.tag + key + '_differ_' + str(self.differ) + '_v%i.png' % self.version in \
                        os.listdir(self.folder + 'Correlation/ACF/'):
                    continue

                # Plot
                fig = plot_acf(diff_data[key], fft=True)
                plt.title(key)
                fig.savefig(self.folder + 'Correlation/ACF/' + self.tag + key + '_differ_' + str(self.differ) +
                            '_v%i.png' % self.version, format='png', dpi=300)
                plt.close()

    def partial_auto_corr(self):
        if self.plotSignalCorrelations:
            # Create folder
            if not os.path.exists(self.folder + 'Correlation/PACF/'):
                os.makedirs(self.folder + 'Correlation/PACF/')

            # Iterate through features
            for key in tqdm(self.data.keys()):
                # Skip if existing
                if self.tag + key + '_differ_' + str(self.differ) + '_v%i.png' % self.version in \
                        os.listdir(self.folder + 'EDA/Correlation/PACF/'):
                    continue

                # Plot
                try:
                    fig = plot_pacf(self.data[key])
                    fig.savefig(self.folder + 'EDA/Correlation/PACF/' + self.tag + key + '_differ_' +
                                str(self.differ) + '_v%i.png' % self.version, format='png', dpi=300)
                    plt.title(key)
                    plt.close()
                except Exception as e:
                    # todo find exception
                    raise ValueError(e)

    def cross_corr(self):
        if self.plotSignalCorrelations:
            # Create folder
            if not os.path.exists(self.folder + 'Correlation/Cross/'):
                os.makedirs(self.folder + 'Correlation/Cross/')

            # Prepare
            folder = 'Correlation/Cross/'
            y = self.Y.to_numpy().reshape((-1))

            # Iterate through features
            for key in tqdm(self.data.keys()):
                # Skip if existing
                if self.tag + key + '_differ_' + str(self.differ) + '_v%i.png' % self.version in \
                        os.listdir(self.folder + folder):
                    continue

                # Plot
                try:
                    fig = plt.figure(figsize=[24, 16])
                    plt.xcorr(self.data[key], y, maxlags=self.lags)
                    plt.title(key)
                    fig.savefig(self.folder + folder + self.tag + key + '_differ_' + str(self.differ) +
                                '_v%i.png' % self.version, format='png', dpi=300)
                    plt.close()
                except Exception as e:
                    raise ValueError(e)

    def scatters(self):
        if self.plotScatterPlots:
            # Create folder
            if not os.path.exists(self.folder + 'Scatters/v%i/' % self.version):
                os.makedirs(self.folder + 'Scatters/v%i/' % self.version)

            # Iterate through features
            for key in tqdm(self.data.keys()):
                # Skip if existing
                if '{}{}.png'.format(self.tag, key) in os.listdir(self.folder + 'Scatters/v%i/' % self.version):
                    continue

                # Plot
                fig = plt.figure(figsize=[24, 16])
                plt.scatter(self.Y, self.data[key], alpha=0.2)
                plt.ylabel(key)
                plt.xlabel('Output')
                plt.title('Scatter Plot ' + key + ' - Output')
                fig.savefig(self.folder + 'Scatters/v%i/' % self.version + self.tag + key + '.png',
                            format='png', dpi=100)
                plt.close(fig)

    def shap(self, args=None):
        if self.plotFeatureImportance:
            # Create folder
            if not os.path.exists(self.folder + 'Features/v%i/' % self.version):
                os.makedirs(self.folder + 'Features/v%i/' % self.version)

            # Skip if existing
            if self.tag + 'SHAP.png' in os.listdir(self.folder + 'Features/v%i/' % self.version):
                return

            # Create model
            if self.mode == 'classification':
                model = RandomForestClassifier(**args).fit(self.data, self.Y)
            else:
                model = RandomForestRegressor(**args).fit(self.data, self.Y)

            # Calculate SHAP values
            import shap
            shap_values = shap.TreeExplainer(model).shap_values(self.data)

            # Plot
            fig = plt.figure(figsize=[8, 32])
            plt.subplots_adjust(left=0.4)
            shap.summary_plot(shap_values, self.data, plot_type='bar')
            fig.savefig(self.folder + 'Features/v%i/' % self.version + self.tag + 'SHAP.png', format='png', dpi=300)

    def feature_ranking(self, **args):
        if self.plotFeatureImportance:
            # Create folder
            if not os.path.exists(self.folder + 'Features/v%i/' % self.version):
                os.mkdir(self.folder + 'Features/v%i/' % self.version)

            # Skip if existing
            if self.tag + 'RF.png' in os.listdir(self.folder + 'Features/v%i/' % self.version):
                return

            # Create model
            if self.mode == 'classification':
                model = RandomForestClassifier(**args).fit(self.data, self.Y)
            else:
                model = RandomForestRegressor(**args).fit(self.data, self.Y)

            # Plot
            fig, ax = plt.subplots(figsize=[4, 6], constrained_layout=True)
            plt.subplots_adjust(left=0.5, top=1, bottom=0)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ind = np.argsort(model.feature_importances_)
            plt.barh(list(self.data.keys()[ind])[-15:], width=model.feature_importances_[ind][-15:],
                     color='#2369ec')
            fig.savefig(self.folder + 'Features/v%i/' % self.version + self.tag + 'RF.png', format='png', dpi=300)
            plt.close()

            # Store results
            results = pd.DataFrame({'x': self.data.keys(), 'score': model.feature_importances_})
            results.to_csv(self.folder + 'Features/v%i/' % self.version + self.tag + 'RF.csv')

    def predictive_power_score(self):
        if self.plotFeatureImportance:
            # Create folder
            if not os.path.exists(self.folder + 'Features/v%i/' % self.version):
                os.mkdir(self.folder + 'Features/v%i/' % self.version)

            # Skip if existing
            if self.tag + 'PPScore.png' in os.listdir(self.folder + 'Features/v%i/' % self.version):
                return

            # Calculate PPS
            data = self.data.copy()
            if isinstance(self.Y, pd.core.series.Series):
                data.loc[:, 'target'] = self.Y
            elif isinstance(self.Y, pd.DataFrame):
                data.loc[:, 'target'] = self.Y.loc[:, self.Y.keys()[0]]
            pp_score = ppscore.predictors(data, 'target').sort_values('ppscore')

            # Plot
            fig, ax = plt.subplots(figsize=[4, 6], constrained_layout=True)
            plt.subplots_adjust(left=0.5, top=1, bottom=0)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['top'].set_visible(False)
            plt.barh(pp_score['x'][-15:], width=pp_score['ppscore'][-15:], color='#2369ec')
            fig.savefig(self.folder + 'Features/v%i/' % self.version + self.tag + 'Ppscore.png', format='png', dpi=400)
            plt.close()

            # Store results
            pp_score.to_csv(self.folder + 'Features/v%i/pp_score.csv' % self.version)
