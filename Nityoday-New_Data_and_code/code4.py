import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse, aic, bic, hqic
import warnings
warnings.filterwarnings("ignore") # To suppress potential warnings

# Load your data
file_path = 'new_india-fii-yeild.xlsx'  # Or your file path
df = pd.read_excel(file_path, index_col='date', parse_dates=['date']) # Directly set date as index while reading

# Display data info and head to confirm
print(df.info())
print(df.head())

def adf_test(series, variable_name):
    """
    Performs Augmented Dickey-Fuller test on a time series.
    """
    result = adfuller(series.dropna(), autolag='AIC') # Using AIC to choose lag order automatically
    print(f'ADF Test for: {variable_name}')
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    if result[1] <= 0.05: # Common significance level of 5%
        print(f"{variable_name} is likely Stationary.")
    else:
        print(f"{variable_name} is likely Non-Stationary.")
    print('-' * 50)

# Apply ADF test to FII and Yield
adf_test(df['fii'], 'FII')
adf_test(df['yield'], 'Yield')