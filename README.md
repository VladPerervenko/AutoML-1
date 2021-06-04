# Amplo - AutoML (for Machine Data)
[![image](https://img.shields.io/pypi/v/amplo.svg)](https://pypi.python.org/pypi/amplo)
[![image](https://travis-ci.com/nielsuit227/AutoML.svg?token=CnXXBmk9Wj2AwwF6svhz&branch=main)](https://travis-ci.com/nielsuit227/AutoML.svg?token=CnXXBmk9Wj2AwwF6svhz&branch=main)
[![PyPI - License](https://img.shields.io/pypi/l/virtualenv?style=flat-square)](https://opensource.org/licenses/MIT)
[![](https://img.shields.io/badge/python-%3E%3D3.6%2C%3C4.0-blue)](https://pypi.org/project/amplo/)

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

## Exploratory Data Analysis
`from Amplo.AutoML import DataExploring`
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
`from Amplo.AutoML import DataProcessing`
Automated Data Cleaning. Handles the following items:
- Cleans Column Names
- Duplicate Columns and Rows
- Data Types
- Missing Values
- Outliers
- Constant Columns

## Feature Processing
`from Amplo.AutoML import FeatureProcessing`
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

Included Feature Selection algorithms:
- Random Forest Feature Importance (Threshold and Increment)
- Predictive Power Score
- Boruta

## Modelling
`from Amplo.AutoML import Modelling`
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
