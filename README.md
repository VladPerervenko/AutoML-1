# Amplo - AutoML (for Machine Data)
[![image](https://img.shields.io/pypi/v/amplo.svg)](https://pypi.python.org/pypi/amplo)
[![PyPI - License](https://img.shields.io/pypi/l/virtualenv?style=flat-square)](https://opensource.org/licenses/MIT)
![](https://img.shields.io/badge/python-%3E%3D3.6%2C%3C4.0-blue)
![](https://tokei.rs/b1/github/nielsuit227/automl)

Welcome to the Automated Machine Learning package `Amplo`. Amplo's AutoML is designed specifically for machine data and 
works very well with tabular time series data (especially unbalanced classification!).

Though this is a standalone Python package, Amplo's AutoML is also available on Amplo's ML Developer Platform. 
With a graphical user interface and various data connectors, it is the ideal place for service engineers to get started 
on Predictive Maintenance development. 

Amplo's AutoML Pipeline contains the entire Machine Learning development cycle, including exploratory data analysis, 
data cleaning, feature extraction, feature selection, model selection, hyper parameter optimization, stacking, 
version control, production-ready models and documentation. 

# Downloading Amplo
The easiest way is to install our Python package through [PyPi](https://pypi.org/project/amplo/):
```commandline
pip install Amplo
```

# 2. Amplo AutoML Features

## Interval Analyser
`from Amplo.AutoML import IntervalAnalyser`

Interval Analyser for Log file classification. When log files have to be classifier, and there is not enough
data for time series methods (such as LSTMs, ROCKET or Weasel, Boss, etc), one needs to fall back to classical
machine learning models which work better with lower samples. But this raises the problem of which samples to
classify. You shouldn't just simply classify on every sample and accumulate, that may greatly disrupt
classification performance. Therefore, we introduce this interval analyser. By using K-Nearest Neighbors,
one can estimate the strength of correlation for every sample inside a log. Using this allows for better
interval selection for classical machine learning models.

To use this interval analyser, make sure:
- That your logs are of equal length
- That your logs have identical keys
- That your logs are located in a folder of their class, with one parent folder with all classes, e.g.:

```
+-- Parent Folder
|   +-- Class_1
|       +-- Log_1.*
|       +-- Log_2.*
|   +-- Class_2
|       +-- Log_3.*
```
## Exploratory Data Analysis
`from Amplo.AutoML import DataExplorer`
Automated Exploratory Data Analysis. Covers binary classification and regression.
It generates:
- Missing Values Plot
- Line Plots of all features
- Box plots of all features
- Co-linearity Plot
- SHAP Values
- Random Forest Feature Importance
- Predictive Power Score

Additionally fFor Regression:
- Seasonality Plots
- Differentiated Variance Plot
- Auto Correlation Function Plot
- Partial Auto Correlation Function Plot
- Cross Correlation Function Plot
- Scatter Plots

## Data Processing
`from Amplo.AutoML import DataProcesser`
Automated Data Cleaning. Handles the following items:
- Cleans Column Names
- Duplicate Columns and Rows
- Data Types
- Missing Values
- Outliers
- Constant Columns

## Feature Processing
`from Amplo.AutoML import FeatureProcesser`
Automatically extracts and selects features. Removes Co-Linear Features.
Included Feature Extraction algorithms:
- Multiplicative Features
- Dividing Features
- Additive Features
- Subtractive Features
- Trigonometric Features
- K-Means Features
- Lagged Features
- Differencing Features
- Inverse Features
- Datetime Features

Included Feature Selection algorithms:
- Random Forest Feature Importance (Threshold and Increment)
- Predictive Power Score
- Boruta

## Sequencing
`from Amplo.AutoML import Sequencer`
For timeseries regression problems, it is often useful to include multiple previous samples instead of just the latest. 
This class sequences the data, based on which time steps you want included in the in- and output. 
This is also very useful when working with tensors, as a tensor can be returned which directly fits into a Recurrent Neural Network. 

## Modelling
`from Amplo.AutoML import Modeller`
Runs various regression or classification models.
Includes:
- Scikit's Linear Model
- Scikit's Random Forest
- Scikit's Bagging
- Scikit's GradientBoosting
- Scikit's HistGradientBoosting
- DMLC's XGBoost
- Catboost's Catboost
- Microsoft's LightGBM

## Grid Search
`from Amplo.GridSearch import *`
Contains three hyperparameter optimizers, a basic `GridSearch`, an implementation of Scikit's `RandomHalvingSearch` and 
an implementation of Optuna's Tree-structured Parzen Estimator. Generally we advice to use Optuna.  

## Automatic Documntation
`from Amplo.AutoML import Documenter`
Contains a documenter for classification (`binary` and `multiclass` prolems), as well as for regression. 
Creates a pdf report for a Pipeline, including metrics, data processing steps, and everything else to recreate the result.


