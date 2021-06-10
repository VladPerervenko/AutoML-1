import os
import math
import copy
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from .BinaryDocumenting import BinaryDocumenting


class MultiDocumenting(BinaryDocumenting):

    def __init__(self, pipeline):
        super().__init__(pipeline)

    def analyse(self):
        # Initiating
        f1_score = np.zeros((self.p.cvSplits, self.p.n_classes))
        log_loss = np.zeros(self.p.cvSplits)
        avg_acc = np.zeros(self.p.cvSplits)
        cm = np.zeros((self.p.cvSplits, self.p.n_classes, self.p.n_classes))

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

        # Result statistics
        totals = np.sum(cm, axis=(1, 2), keepdims=True)
        means = np.mean(cm / totals * 100, axis=0)
        stds = np.std(cm / totals * 100, axis=0)

        # Store
        self.metrics = {
            'F1 Score': [np.mean(f1_score), np.std(f1_score)],
            'Accuracy': [np.mean(avg_acc), np.std(avg_acc)],
        }
        self.confusion_matrix = {
            'means': means,
            'stds': stds,
        }

        # Print
        print('F1 scores:')
        print(''.join([' Class {} |'.format(i) for i in range(self.p.n_classes)]))
        print(''.join([' {:.2f} % '.ljust(11).format(f1) + '|' for f1 in np.mean(f1_score, axis=0)]))
        print('Average Accuracy: {:.2f} \u00B1 {:.2f} %'.format(np.mean(avg_acc), np.std(avg_acc)))
        if hasattr(model, 'predict_proba'):
            print('Log Loss:         {:.2f} \u00B1 {:.2f}'.format(np.mean(log_loss), np.std(log_loss)))
            self.metrics['Log Loss'] = [np.mean(log_loss), np.std(log_loss)]

    def model_performance(self):
        # todo adapt multiclass matrix (maybe change to a plot?)
        # todo remove figures
        self.ln(self.lh)
        self.add_h2('Model Performance')
        self.add_text('Model performance is analysed by various metrics. This model has been selected based on the {} '
                      'score.'.format(self.p.objective))

        # Metrics
        self.set_font('Helvetica', 'B', 12)
        self.ln(self.lh)
        self.cell(w=50, h=self.lh, txt='Metric', border='B', align='C')
        self.cell(w=50, h=self.lh, txt='Score', border='LB', align='C')
        self.set_font('Helvetica', '', 12)
        for k, v in self.metrics.items():
            self.ln(self.lh)
            self.cell(w=50, h=self.lh, txt=k, border='R', align='L')
            self.cell(w=50, h=self.lh, txt='{:.2f} \u00B1 {:.2f} %'.format(v[0], v[1]), border='L', align='C')
        self.ln(self.lh * 3)

        # Confusion Matrix
        n_classes = 2
        cell_width = int((self.WIDTH - self.pm * 4 - self.tm * n_classes) / (n_classes + 2))
        self.add_h3('Confusion Matrix')

        # First row
        self.set_font('Helvetica', 'B', 12)
        self.cell(w=cell_width * 2, h=self.lh, txt='', align='L', border='R')
        self.cell(w=cell_width * n_classes, h=self.lh, txt='True Class', align='C')
        self.ln(self.lh)

        # Second Row
        self.cell(w=cell_width * 2, h=self.lh, txt='', align='L', border='BR')
        self.cell(w=cell_width, h=self.lh, txt='Faulty', align='C', border='B')
        self.cell(w=cell_width, h=self.lh, txt='Healthy', align='C', border='B')
        self.ln(self.lh)

        # Third Row (first with values)
        self.cell(w=cell_width, h=self.lh * 2, txt='Prediction', align='L')
        self.cell(w=cell_width, h=self.lh, txt='Faulty', align='L', border='R')
        self.set_font('Helvetica', '', 12)
        self.cell(w=cell_width, h=self.lh, txt='{:.2f} \u00B1 {:.2f} %'.format(
            self.confusion_matrix['means'][0][0], self.confusion_matrix['stds'][0][0]),
                  align='C')
        self.cell(w=cell_width, h=self.lh, txt='{:.2f} \u00B1 {:.2f} %'.format(
            self.confusion_matrix['means'][0][1], self.confusion_matrix['stds'][0][1]),
                  align='C')
        self.ln(self.lh)

        # Fourth Row
        self.set_font('Helvetica', 'B', 12)
        self.cell(w=cell_width, h=self.lh, txt='', align='L')
        self.cell(w=cell_width, h=self.lh, txt='Healthy', align='L', border='R')
        self.set_font('Helvetica', '', 12)
        self.cell(w=cell_width, h=self.lh, txt='{:.2f} \u00B1 {:.2f} %'.format(
            self.confusion_matrix['means'][1][0], self.confusion_matrix['stds'][1][0]),
                  align='C')
        self.cell(w=cell_width, h=self.lh, txt='{:.2f} \u00B1 {:.2f} %'.format(
            self.confusion_matrix['means'][1][1], self.confusion_matrix['stds'][1][1]),
                  align='C')
        self.ln(self.lh * 2)
