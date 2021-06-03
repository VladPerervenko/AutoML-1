import os
import copy
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


class Documenting:

    def __init__(self, pipeline):
        if not os.path.exists(pipeline.mainDir + 'Documentation/v{}'.format(pipeline.version)):
            os.makedirs(pipeline.mainDir + 'Documentation/v{}'.format(pipeline.version))
        self.mName = None
        self.p = pipeline
        self.model = None
        self.params = None
        self.featureSet = None
        self.metrics = None
        self.confusionMatrix = None
        self.x = None
        self.y = None
        self.y_not_normalized = None
        self.cv = None
        self.mode = pipeline.mode
        self.classes_ = None
        if self.mode == 'classification' and self.p.y.nunique() > 2:
            self.mode = 'multiclass'
            self.classes_ = self.p.y.unique()

    def prepare_data(self):
        """
        Although data of the Pipeline (self.p.x and self.p.y), is already cleaned, it is not normalized or sequenced.
        @return:
        @rtype:
        """
        x, y = copy.deepcopy(self.p.x[self.p.colKeep[self.featureSet]]), copy.deepcopy(self.p.y)

        # Normalize
        if self.p.normalize:
            normalize_features = [k for k in x.keys() if k not in self.p.dateCols + self.p.catCols]
            x[normalize_features] = self.p.bestScaler.transform(x[normalize_features])
            if self.mode == 'regression':
                self.y_not_normalized = y.values.reshape(-1, 1)
                y = pd.Series(self.p.bestOutputScaler.transform(y.values.reshape(-1, 1)).reshape(-1),
                              name=self.p.target)
        self.x, self.y = x.to_numpy(), y.to_numpy().reshape(-1, 1)
        # todo implement sequencer

    def create(self, model, feature_set):
        """
        Creates Automatic Documentation.
        @param [object] model: Model object, requires set_params, fit, predict, predict_proba
        @param [dict] params: Set of model parameters
        @param [str] feature_set: Feature set
        """
        self.mName = type(model).__name__
        self.model = model
        self.featureSet = feature_set
        self.prepare_data()
        print('[Documenting] {} {} {}'.format(self.mName, feature_set, self.p.version))

        # Introduction
        markdown = "# Amplo AutoML Documentation - {} v{}\n\n".format(self.mName, self.p.version)
        markdown += self.model_description()

        # Mode dependent information
        if self.mode == 'classification':
            markdown += self.binary_markdown()
        elif self.mode == 'multiclass':
            markdown += self.multiclass_markdown()
        elif self.mode == 'regression':
            markdown += self.regression_markdown()

        # Cross validation
        markdown += self.validation_strategy_markdown()

        # Parameters
        markdown += self.parameter_markdown()

        # Features
        markdown += self.features_markdown()

        # Data Processing
        markdown += self.data_markdown()

        # Other models
        markdown += self.modelling_markdown()

        return markdown

    def model_description(self):
        markdown = "## Model Information\n\n"
        if 'CatBoost' in self.mName:
            markdown += "CatBoost, or **Cat**egorical **Boost**ing,  is an algorithm for gradient boosting on " \
                        "decision trees, with natural implementation for categorical variables. It is similar to " \
                        "XGBoost and LightGBM but differs in implementation of the optimization algorithm. We often " \
                        "see this algorithm performing very well."
        if 'XGB' in self.mName:
            markdown += "XGBoost, or E**X**treme **G**radient **Boost**ing, is an algorithm for gradient boosting on " \
                        "decision trees. It trains many decision trees sequentially, the additional tree always " \
                        "trying to mitigate the error of the whole model. XGBoost was the first gradient boosting " \
                        "algorithm to be implemented and is currently widely adopted in the ML world. "
        if 'LGBM' in self.mName:
            markdown += 'LightGBM, or **L**ight **G**radient **B**oosting **M**achine, is an iteration on the XGBoost' \
                        ' algorithm. Similarly, it uses gradient boosting with decision trees. However, XGBoost tend ' \
                        'to be slow for a larger number of samples (>10.000), but with leaf-wise growth instead of ' \
                        'depth-wise growth, LightGBM increases training speed significatnly. Performance is often ' \
                        'close to XGBoost, sometimes for the better and sometimes for the worse. '
        if 'HistGradientBoosting' in self.mName:
            markdown += 'SciKits implementation of LightGBM, or **L**ight **G**radient **B**oosting **M**achine, is' \
                        ' an iteration on the XGBoost ' \
                        'algorithm. Similarly, it uses gradient boosting with decision trees. However, XGBoost tend ' \
                        'to be slow for a larger number of samples (>10.000), but with leaf-wise growth instead of ' \
                        'depth-wise growth, LightGBM increases training speed significatnly. Performance is often ' \
                        'close to XGBoost, sometimes for the better and sometimes for the worse. '
        elif 'GradientBoosting' in self.mName:
            markdown += "SciKits implementation of XGBoost, or E**X**treme **G**radient Boosting, is an algorithm for" \
                        " gradient boosting on " \
                        "decision trees. It trains many decision trees sequentially, the additional tree always " \
                        "trying to mitigate the error of the whole model. XGBoost was the first gradient boosting " \
                        "algorithm to be implemented and is currently widely adopted in the ML world. "
        if 'RandomForest' in self.mName:
            markdown += "Random Forest, is an ensemble algorithm that combines many (100-1000) decision trees and " \
                        "predicts the average of all trained trees. Though gradient boosting methods often outperform " \
                        "Random Forest, some data characteristics favor the Random Forests performance. "
        return markdown + '\n\n'

    def binary_markdown(self):
        # Calculation
        accuracy = []
        precision = []
        sensitivity = []
        specificity = []
        f1_score = []
        area_under_curves = []
        true_positive_rates = []
        cm = np.zeros((self.p.cvSplits, 2, 2))
        mean_fpr = np.linspace(0, 1, 100)

        # Plot Initiator
        fig, ax = plt.subplots(math.ceil(self.p.cvSplits / 2), 2, sharex='all', sharey='all', figsize=[12, 8])
        fig.suptitle('{}-Fold Cross Validated Predictions - {} ({})'.format(
            self.p.cvSplits, type(self.model).__name__, self.featureSet))
        fig2, ax2 = plt.subplots(figsize=[12, 8])

        # Modelling
        self.cv = StratifiedKFold(n_splits=self.p.cvSplits, shuffle=self.p.shuffle)
        for i, (t, v) in enumerate(self.cv.split(self.x, self.y)):
            n = len(v)
            xt, xv, yt, yv = self.x[t], self.x[v], self.y[t].reshape((-1)), self.y[v].reshape((-1))
            model = copy.deepcopy(self.model)
            model.fit(xt, yt)
            predictions = model.predict(xv).reshape((-1))

            # Metrics
            tp = np.logical_and(np.sign(predictions) == 1, yv == 1).sum()
            tn = np.logical_and(np.sign(predictions) == 0, yv == 0).sum()
            fp = np.logical_and(np.sign(predictions) == 1, yv == 0).sum()
            fn = np.logical_and(np.sign(predictions) == 0, yv == 1).sum()
            accuracy.append((tp + tn) / n * 100)
            if tp + fp > 0:
                precision.append(tp / (tp + fp) * 100)
            if tp + fn > 0:
                sensitivity.append(tp / (tp + fn) * 100)
            if tn + fp > 0:
                specificity.append(tn / (tn + fp) * 100)
            if tp + fp > 0 and tp + fn > 0:
                f1_score.append(2 * precision[-1] * sensitivity[-1] / (precision[-1] + sensitivity[-1]) if
                                precision[-1] + sensitivity[-1] > 0 else 0)
            cm[i] = np.array([[tp, fp], [fn, tn]])

            # ROC calculations
            viz = plot_roc_curve(model, xv, yv, name='ROC fold {}'.format(i + 1), alpha=0.3, ax=ax2)
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            true_positive_rates.append(interp_tpr)
            area_under_curves.append(viz.roc_auc)

            # Plot
            ax[i // 2][i % 2].plot(yv, c='#2369ec', alpha=0.6)
            ax[i // 2][i % 2].plot(predictions, c='#ffa62b')
            ax[i // 2][i % 2].set_title('Fold-{}'.format(i))

        # Statistics on results
        totals = np.sum(cm, axis=(1, 2), keepdims=True)
        means = np.mean(cm / totals * 100, axis=0)
        stds = np.std(cm / totals * 100, axis=0)

        # Print Results
        print('[Documenting] Accuracy:        {:.2f} \u00B1 {:.2f} %'.format(np.mean(accuracy), np.std(accuracy)))
        print('[Documenting] Precision:       {:.2f} \u00B1 {:.2f} %'.format(np.mean(precision), np.std(precision)))
        print('[Documenting] Recall:          {:.2f} \u00B1 {:.2f} %'.format(
            np.mean(sensitivity), np.std(sensitivity)))
        print('[Documenting] Specificity:     {:.2f} \u00B1 {:.2f} %'.format(
            np.mean(specificity), np.std(specificity)))
        print('[Documenting] F1-score:        {:.2f} \u00B1 {:.2f} %'.format(np.mean(f1_score), np.std(f1_score)))
        print('[Documenting] Confusion Matrix:')
        print('[Documenting] Prediction / true |    Faulty    |    Healthy      ')
        print('[Documenting]       Faulty      | {} |  {}'.format(
            ('{:.1f} \u00B1 {:.1f} %'.format(means[0, 0], stds[0, 0])).ljust(12),
            ('{:.1f} \u00B1 {:.1f} %'.format(means[0, 1], stds[0, 1])).ljust(12)))
        print('[Documenting]       Healthy     | {} |  {}'.format(
            ('{:.1f} \u00B1 {:.1f} %'.format(means[1, 0], stds[1, 0])).ljust(12),
            ('{:.1f} \u00B1 {:.1f} %'.format(means[1, 1], stds[1, 1])).ljust(12)))

        # Check whether plot is possible
        if type(self.model).__name__ == 'Lasso' or 'Ridge' in type(self.model).__name__:
            return

        # Adjust plots
        ax2.plot([0, 1], [0, 1], linestyle='--', lw=2, color='#ffa62b',
                 label='Chance', alpha=.8)
        mean_tpr = np.mean(true_positive_rates, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(area_under_curves)
        ax2.plot(mean_fpr, mean_tpr, color='#2369ec',
                 label=r'Mean ROC (AUC = {:.2f} $\pm$ {:.2f})'.format(mean_auc, std_auc),
                 lw=2, alpha=.8)
        std_tpr = np.std(true_positive_rates, axis=0)
        true_pos_rates_upper = np.minimum(mean_tpr + std_tpr, 1)
        true_pos_rates_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax2.fill_between(mean_fpr, true_pos_rates_lower, true_pos_rates_upper, color='#729ce9', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')
        ax2.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
                title="ROC Curve - {}".format(self.mName))
        ax2.legend(loc="lower right")
        roc_path = self.p.mainDir + 'Documentation/v{}/ROC_{}.png'.format(self.p.version, self.mName)
        fig2.savefig(roc_path, format='png', dpi=200)
        cross_val_path = self.p.mainDir + 'Documentation/v{}/Cross_Val_{}.png'.format(self.p.version, self.mName)
        fig.savefig(cross_val_path, format='png', dpi=200)

        # Markdown
        metrics_table = """| Metric | Score |
| --- | ---: |
| Accuracy    | {:.2f} \u00B1 {:.2f} % |
| Precision   | {:.2f} \u00B1 {:.2f} % |
| Sensitivity | {:.2f} \u00B1 {:.2f} % |
| Specificity | {:.2f} \u00B1 {:.2f} % |
| F1 Score    | {:.2f} \u00B1 {:.2f} % |""".format(np.mean(accuracy), np.std(accuracy),
                                                   np.mean(precision), np.std(accuracy),
                                                   np.mean(sensitivity), np.std(sensitivity),
                                                   np.mean(specificity), np.std(specificity),
                                                   np.mean(f1_score), np.std(f1_score))
        confusion_matrix = """
<table>
    <thead>
        <tr>
            <td> </td>
            <td> </td>
            <td colspan=2 style="text-align:center">
                True Label
            </td>
        </tr>
        <tr>
            <td> </td>
            <td> </td>
            <td>Faulty</td>
            <td>No Issue</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2 style="vertical-align:middle">
                Prediction
            </td>
            <td>Faulty</td>
            <td>{:.1f} \u00B1 {:.1f} %</td>
            <td>{:.1f} \u00B1 {:.1f} %</td>
        </tr>
        <tr>
            <td>No Issue</td>
            <td>{:.1f} \u00B1 {:.1f} %</td>
            <td>{:.1f} \u00B1 {:.1f} %</td>
        </tr>
    </tbody>
</table>
        """.format(means[0, 0], stds[0, 0],
                   means[0, 1], stds[0, 1],
                   means[1, 0], stds[1, 0],
                   means[1, 1], stds[1, 1])
        return """## Model Performance

Model performance is analysed by various metrics. Below you find various metrics and a confusion matrix.
This model has been selected based on the {} score.

### Metrics

{}

### Confusion Matrix

{}

### Area Under Curve & Cross Validation Plot

<table>
    <tr>
        <td><img src="{}" width=600 height=400></td>
        <td><img src="{}" width=600 height=400></td>
    </tr>
<table>

""".format(self.p.objective,
           metrics_table, confusion_matrix,
           'ROC_{}.png'.format(self.mName), 'Cross_Val_{}.png'.format(self.mName))

    def multiclass_markdown(self):
        # Initiating
        fig, ax = plt.subplots(math.ceil(self.p.cvSplits / 2), 2, sharex='all', sharey='all')
        fig.suptitle('{}-Fold Cross Validated Predictions - {} ({})'.format(
            self.p.cvSplits, self.mName, self.featureSet))
        n_classes = len(self.classes_)
        f1_score = np.zeros((self.p.cvSplits, n_classes))
        log_loss = np.zeros(self.p.cvSplits)
        avg_acc = np.zeros(self.p.cvSplits)
        cm = np.zeros((self.p.cvSplits, n_classes, n_classes))

        # Modelling
        self.cv = StratifiedKFold(n_splits=self.p.cvSplits, shuffle=self.p.shuffle)
        for i, (t, v) in enumerate(self.cv.split(self.x, self.y)):
            xt, xv, yt, yv = self.x[t], self.x[v], self.y[t].reshape((-1)), self.y[v].reshape((-1))
            model = copy.copy(self.model)
            model.fit(xt, yt)
            predictions = model.predict(xv).reshape((-1))
            cm[i] = metrics.confusion_matrix(predictions, yv)

            # Metrics
            f1_score[i] = metrics.f1_score(yv, predictions, average=None)
            avg_acc[i] = metrics.accuracy_score(yv, predictions)
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(xv)
                log_loss[i] = metrics.log_loss(yv, probabilities)

            # Plot
            ax[i // 2][i % 2].plot(yv, c='#2369ec', alpha=0.6)
            ax[i // 2][i % 2].plot(predictions, c='#ffa62b')
            ax[i // 2][i % 2].set_title('Fold-{}'.format(i))

        # Results
        print('F1 scores:')
        print(''.join([' Class {} |'.format(i) for i in range(n_classes)]))
        print(''.join([' {:.2f} % '.ljust(11).format(f1) + '|' for f1 in np.mean(f1_score, axis=0)]))
        print('Average Accuracy: {:.2f} \u00B1 {:.2f} %'.format(np.mean(avg_acc), np.std(avg_acc)))
        if hasattr(model, 'predict_proba'):
            print('Log Loss:         {:.2f} \u00B1 {:.2f}'.format(np.mean(log_loss), np.std(log_loss)))

        # Result statistics
        totals = np.sum(cm, axis=(1, 2), keepdims=True)
        means = np.mean(cm / totals * 100, axis=0)
        stds = np.std(cm / totals * 100, axis=0)

        # Markdown
        metrics_table = """| Metrcis | Score |
| --- | ---: |
| Avg. Accuracy | {:.2f} \u00B1 {:.2f} |
| F1 Score      | {:.2f} \u00B1 {:.2f} |
""".format(np.mean(avg_acc), np.std(avg_acc),
           np.mean(f1_score), np.std(f1_score))
        if hasattr(model, 'predict_proba'):
            metrics_table += "| Log Loss      | {:.2f} \u00B1 {:.2f} |".format(np.mean(log_loss), np.std(log_loss))
        classes_header = "<tr><td> </td><td> </td>{}</tr>""".format(''.join(['<td> {} </td>'.format(self.classes_[i])
                                                                   for i in range(n_classes)]))
        prediction = '<td rowspan={} style="vertical-align:middle">Prediction</td>'.format(n_classes)
        rows = '\n'.join(["<tr>\n{}\n{}\n{}\n</tr>".format(
            prediction if i == 0 else '',
            '<td>\n{}\n</td>'.format(self.classes_[i]),
            '\n'.join(['<td>\n{:.2f} \u00B1 {:.2f}\n</td>'.format(means[j, i], stds[j, i]) for j in range(n_classes)]))
                for i in range(n_classes)])
        confusion_matrix = """
<table>
    <thead>
        <tr>
            <td> </td>
            <td> </td>
            <td colspan={} style="text-align:center">True Label</td>
        </tr>
        {}
    </thead>
    <tbody>
        {}
    </tbody>
</table>
                
""".format(n_classes, classes_header, rows)
        return """## Model Performance

Model performance is analysed by various metrics. Below you find various metrics and a confusion matrix.
This model has been selected based on the {} score.

### Metrics

{}

### Confusion Matrix

{}

""".format(self.p.objective, metrics_table, confusion_matrix)

    def regression_markdown(self):
        # Cross-Validation Plots
        fig, ax = plt.subplots(math.ceil(self.p.cvSplits / 2), 2, sharex='all', sharey='all', figsize=[24, 8])
        fig.suptitle('{}-Fold Cross Validated Predictions - {}'.format(self.p.cvSplits, self.mName))

        # Initialize iterables
        mae = []
        mse = []
        r2 = []
        max_error = []
        rel_error = []
        self.cv = KFold(n_splits=self.p.cvSplits, shuffle=self.p.shuffle)
        # Cross Validate
        i = 0
        for i, (t, v) in enumerate(self.cv.split(self.x, self.y)):
            xt, xv, yt, yv = self.x[t], self.x[v], self.y[t].reshape((-1)), self.y[v].reshape((-1))
            model = copy.deepcopy(self.model)
            model.fit(xt, yt)
            prediction = model.predict(xv)

            # Metrics
            mae.append(metrics.mean_absolute_error(yv, prediction))
            mse.append(metrics.mean_squared_error(yv, prediction))
            r2.append(metrics.r2_score(yv, prediction))
            max_error.append(metrics.max_error(yv, prediction))
            rel_error.append(metrics.mean_absolute_percentage_error(yv, prediction))

            # Plot
            ax[i // 2][i % 2].set_title('Fold-{}'.format(i))
            if self.p.normalize:
                ax[i // 2][i % 2].plot(self.y_not_normalized[v], color='#2369ec')
                ax[i // 2][i % 2].plot(self.p.bestOutputScaler.inverse_transform(model.predict(xv)),
                                       color='#ffa62b', alpha=0.4)
            else:
                ax[i // 2][i % 2].plot(yv, color='#2369ec')
                ax[i // 2][i % 2].plot(model.predict(xv), color='#ffa62b', alpha=0.4)

        # Save figure
        ax[i // 2][i % 2].legend(['Output', 'Prediction'])
        cross_val_path = self.p.mainDir + 'Documentation/v{}/Cross_Val_{}.png'.format(self.p.version, self.mName)
        fig.savefig(cross_val_path, format='png', dpi=200)

        # Print & Finish plot
        print('[AutoML] Mean Absolute Error:            {:.2f} \u00B1 {:.2f}'.format(np.mean(mae), np.std(mae)))
        print('[AutoML] Mean Squared Error:             {:.2f} \u00B1 {:.2f}'.format(np.mean(mse), np.std(mse)))
        print('[AutoML] R2 Score:                       {:.2f} \u00B1 {:.2f}'.format(np.mean(r2), np.std(r2)))
        print('[AutoML] Max Error:                      {:.2f} \u00B1 {:.2f}'.format(
            np.mean(max_error), np.std(max_error)))
        print('[AutoML] Mean Absolute Relative Error:   {:.2f} \u00B1 {:.2f}'.format(
            np.mean(rel_error), np.std(rel_error)))
        metrics_table = """| Metric | Score |
| --- | ---: |
| Mean Absolute Error | {:.2f} \u00B1 {:.2f} % |
| Mean Squared Error  | {:.2f} \u00B1 {:.2f} % |
| R2 Score | {:.2f} \u00B1 {:.2f} % |
| Maximum Error | {:.2f} \u00B1 {:.2f} % |
| Mean Absolute Relative Error: | {:.2f} \u00B1 {:.2f} % |
""".format(np.mean(mae), np.std(mae),
           np.mean(mse), np.std(mse),
           np.mean(r2), np.std(2),
           np.mean(max_error), np.std(max_error),
           np.mean(rel_error), np.std(rel_error))
        return """## Model Performance


Model performance is analysed by various metrics. Below you find various metrics and a cross validated prediction plot.

### Metrics

{}

### Prediction Plot

<img src="Cross_Val_{}.png" width=1200, height=400>

""".format(metrics_table, self.mName)

    def validation_strategy_markdown(self):
        return """## Validation Strategy

All experiments are cross validated. This means that every time a model's performance is evaluated, it's trained on one part of the data, and tested on another. Therefore, the model is always tested against data it has not yet been trained for. This gives the best approximation for real world (out of sample) performance.
The current validation strategy used {}, with {} splits and {} shuffling the data.

""".format(type(self.cv).__name__, self.p.cvSplits, 'with' if self.p.shuffle else 'without')

    @staticmethod
    def split_param_table(params):
        if len(params) > 10:
            table = ''
            for i in range(-(-len(params) // 10)):  # Ceil
                table += '| ' + ' | '.join([k for k in list(params.keys())[i * 10: (i + 1) * 10]]) + '|'
                table += '\n|' + ' --- |' * min(len(params) - i * 10, 10)
                table += "\n| " + ' | '.join([str(v) if not isinstance(v, float) else str(round(v, 4))
                                              for v in list(params.values())[i * 10: (i + 1) * 10]]) + ' |\n\n'
        else:
            table = '| ' + ' | '.join([k for k in params.keys()]) + ' |\n'
            table += '\n|' + ' --- |' * 10
            table += "\n| " + ' | '.join([str(v) if not isinstance(v, float) else str(round(v, 4))
                                          for v in params.values()]) + ' |\n\n'
        return table

    def parameter_markdown(self):
        # Parameter section for stacking
        if 'Stacking' in self.mName:
            estimators = self.model.get_params()['estimators']
            table = ''
            for name, estimator in estimators:
                params = estimator.get_params()
                table += self.split_param_table(params)
        # Other table
        else:
            params = self.model.get_params()
            table = self.split_param_table(params)
        return """## Parameters

The optimized model has the following parameters:

{}

""".format(table)

    def features_markdown(self):
        # Calculate feature importance
        feature_importance_url = self.p.mainDir + 'Documentation/v{}/Feature_Importance_{}.png'.format(
            self.p.version, self.mName)
        if not os.path.exists(feature_importance_url):
            rf = None
            if self.mode == 'regression':
                rf = RandomForestRegressor().fit(self.p.x, self.p.y)
            else:
                rf = RandomForestClassifier().fit(self.p.x, self.p.y)
            fig, ax = plt.subplots(figsize=[4, 6], constrained_layout=True)
            plt.subplots_adjust(left=0.5, top=1, bottom=0)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ind = np.argsort(rf.feature_importances_)
            plt.barh(list(self.p.x.keys()[ind])[-15:], width=rf.feature_importances_[ind][-15:],
                     color='#2369ec')
            fig.savefig(feature_importance_url, format='png', dpi=200)

        # Number of Features
        n_co_linear = len(self.p.featureProcessor.coLinearFeatures)
        n_cross = len(self.p.featureProcessor.crossFeatures)
        n_additive = len(self.p.featureProcessor.addFeatures)
        n_trigonometry = len(self.p.featureProcessor.trigoFeatures)
        n_k_means = len(self.p.featureProcessor.kMeansFeatures)
        n_lagged = len(self.p.featureProcessor.laggedFeatures)
        n_diff = len(self.p.featureProcessor.diffFeatures)
        return """## Features

### Feature Extraction

Firstly, features that are co-linear (a * x = y), up to {} %, were removed. This resulted in {} removed features:
{} 

Subsequently, the features were manipulated and analysed to extract additional information. 
Most combinations are tried (early stopping avoids unpromising features directly). 
The usefulness of a newly extracted features are analysed by a single decision tree. 

| Sort Feature | Quantity | Features |
| --- | ---: | --- |
| Multiplied / Divided Features | {} | {} |
| Added / Subtracted Features   | {} | {} |
| Trigonometric Features        | {} | {} |
| K-Means Features              | {} | {} |
| Lagged Features               | {} | {} |
| Differentiated Features       | {} | {} |

### Feature Selection
Using a Random Forest model, the non-linear Feature Importance is analysed. The Feature Importance is measured
in Mean Decrease in Gini Impurity. 
The Feature Importance is used to create two feature sets, one that contains 95% of all Feature Importance (RFT) and 
one that contains all features that contribute more than 1% to the total Feature Importance (RFI). 

Top 20 features:

<img src="{}" width="400" height="600">

""".format(self.p.informationThreshold * 100,
           n_co_linear, ', '.join([i for i in self.p.featureProcessor.coLinearFeatures]),
           n_cross, ', '.join([i for i in self.p.featureProcessor.crossFeatures]),
           n_additive, ', '.join([i for i in self.p.featureProcessor.addFeatures]),
           n_trigonometry, ', '.join([i for i in self.p.featureProcessor.trigoFeatures]),
           n_k_means, ', '.join([i for i in self.p.featureProcessor.kMeansFeatures]),
           n_lagged, ', '.join([i for i in self.p.featureProcessor.laggedFeatures]),
           n_diff, ', '.join([i for i in self.p.featureProcessor.diffFeatures]),
           'Feature_Importance_{}.png'.format(self.mName))

    def data_markdown(self):
        return """## Data Processing
        
Data cleaning steps: 
1. Removed {} duplicate columns and {} duplicate rows.
2. Handled outliers with {}
3. Imputed {} missing values with {}
4. Removed {} columns with only constant values

""".format(self.p.dataProcessor.removedDuplicateColumns, self.p.dataProcessor.removedDuplicateRows,
           self.p.outlierRemoval,
           self.p.dataProcessor.imputedMissingValues, self.p.missingValues,
           self.p.dataProcessor.removedConstantColumns)

    def modelling_markdown(self):
        top_ten_table = "| Model | {} | Parameters |\n| --- | ---: | --- |\n".format(self.p.objective.ljust(16))
        for i in range(min(10, len(self.p.results))):
            model = self.p.results.iloc[i]['model']
            score = '{:.4f} \u00B1 {:.4f}'.format(self.p.results.iloc[i]['mean_objective'],
                                                  self.p.results.iloc[i]['std_objective']).ljust(16)
            params = self.p.results.iloc[i]['params']
            top_ten_table += "| {} | {} | {} |\n".format(model, score, params)
        return """## Model Score Board

Not only {} has been optimized by the AutoML pipeline. In total, {} models were trained. 
The following table shows the performance of the top 10 performing models:

{}

""".format(self.mName, len(self.p.results), top_ten_table)
