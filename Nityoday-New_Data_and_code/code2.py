import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from scipy.stats import jarque_bera

# Load the data
file_path = 'india-fii-yeild.xlsx'  # Replace with your file path
data = pd.read_excel(file_path)

# Inspect the data
print(data.head())

# Ensure data is sorted by date
data['date'] = pd.to_datetime(data['date'])
data.sort_values(by='date', inplace=True)
data.set_index('date', inplace=True)

# Calculate log differences (to stabilize variance)
data['log_fii'] = np.log(data['fii'].replace({0: np.nan}))  # Handle zeros to avoid log errors
data['log_yield'] = np.log(data['yield'])
data['log_diff_fii'] = data['log_fii'].diff()
data['log_diff_yield'] = data['log_yield'].diff()

# Perform second differencing for FII
data['log_diff2_fii'] = data['log_diff_fii'].diff()

# Drop NaN values resulting from differencing
data.dropna(inplace=True)

# Plot second differences
plt.figure(figsize=(14, 6))
plt.plot(data['log_diff2_fii'], label='Second Log Difference of FII', color='red')
plt.title('Second Log Difference of FII')
plt.legend()
plt.show()

# Stationarity test using Augmented Dickey-Fuller (ADF) test
def adf_test(series, series_name):
    result = adfuller(series)
    print(f"ADF Test for {series_name}:")
    print(f"Test Statistic: {result[0]:.4f}")
    print(f"P-Value: {result[1]:.4f}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"\t{key}: {value:.4f}")
    print("Stationary" if result[1] <= 0.05 else "Non-Stationary")

adf_test(data['log_diff2_fii'], 'Second Log Difference of FII')
adf_test(data['log_diff_yield'], 'Log Difference of Yield')

# Check for autocorrelation and partial autocorrelation
lag_acf_fii = acf(data['log_diff2_fii'], nlags=20)
lag_pacf_fii = pacf(data['log_diff2_fii'], nlags=20)

lag_acf_yield = acf(data['log_diff_yield'], nlags=20)
lag_pacf_yield = pacf(data['log_diff_yield'], nlags=20)

# Plot ACF and PACF for both series
plt.figure(figsize=(14, 10))

# FII Plots
plt.subplot(2, 2, 1)
plt.stem(range(len(lag_acf_fii)), lag_acf_fii)
plt.title('Autocorrelation of Second Difference of FII')

plt.subplot(2, 2, 2)
plt.stem(range(len(lag_pacf_fii)), lag_pacf_fii)
plt.title('Partial Autocorrelation of Second Difference of FII')

# Yield Plots
plt.subplot(2, 2, 3)
plt.stem(range(len(lag_acf_yield)), lag_acf_yield)
plt.title('Autocorrelation of Log Difference of Yield')

plt.subplot(2, 2, 4)
plt.stem(range(len(lag_pacf_yield)), lag_pacf_yield)
plt.title('Partial Autocorrelation of Log Difference of Yield')

plt.tight_layout()
plt.show()

# Test for normality using Jarque-Bera Test
def jarque_bera_test(series, series_name):
    stat, p = jarque_bera(series)
    print(f"Jarque-Bera Test for {series_name}:")
    print(f"Statistic: {stat:.4f}, P-Value: {p:.4f}")
    print("Normally Distributed" if p > 0.05 else "Not Normally Distributed")

jarque_bera_test(data['log_diff2_fii'], 'Second Log Difference of FII')
jarque_bera_test(data['log_diff_yield'], 'Log Difference of Yield')

# Model Selection: Information Criteria
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.vecm import select_order

# For ARIMA on FII
fii_model = ARIMA(data['log_diff2_fii'], order=(1, 0, 1)).fit()
print(f"FII ARIMA Model Summary:\n{fii_model.summary()}")

# For ARIMA on Yield
yield_model = ARIMA(data['log_diff_yield'], order=(1, 0, 1)).fit()
print(f"Yield ARIMA Model Summary:\n{yield_model.summary()}")

# VAR Model Selection (for multivariate analysis)
model_selection = select_order(data[['log_diff2_fii', 'log_diff_yield']], maxlags=15, deterministic="ci")
print(f"Selected VAR Model Order: {model_selection.summary()}")



# this has model selection parameters quite nicely. now onto code 3 