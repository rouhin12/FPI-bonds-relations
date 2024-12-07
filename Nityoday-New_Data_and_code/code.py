import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Load the data
data = pd.read_excel('india-fii-yeild.xlsx')
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values(by='date').reset_index(drop=True)

# Log Transformation (Skip Normalization)
data['log_yield'] = np.log(data['yield'])
data['log_fii'] = np.log(data['fii'].replace(0, np.nan))  # Replace 0s to avoid log errors

# Plot log-transformed data
plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['log_yield'], label='Log G-Sec Yield', color='blue')
plt.plot(data['date'], data['log_fii'], label='Log FII Flows', color='orange')
plt.title('Log-Transformed G-Sec Yield and FII Flows Over Time')
plt.xlabel('Date')
plt.ylabel('Log Values')
plt.legend()
plt.show()

# Correlation Analysis
log_correlation = data[['log_yield', 'log_fii']].corr()
print("\nCorrelation between Log-Transformed G-Sec Yield and FII Flows:")
print(log_correlation)

# Stationarity Check
def adf_test(series, title=''):
    print(f'ADF Test for {title}')
    result = adfuller(series.dropna())
    print(f'Test Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')
    if result[1] <= 0.05:
        print("Conclusion: Reject H0, the series is stationary.")
    else:
        print("Conclusion: Fail to reject H0, the series is non-stationary.")

# Check Stationarity for Log Data
print("\nStationarity Check for Log-Transformed Data:")
adf_test(data['log_yield'], title='Log G-Sec Yield')
adf_test(data['log_fii'], title='Log FII Flows')

# Differencing
data['log_yield_diff'] = data['log_yield'].diff()
data['log_fii_diff'] = data['log_fii'].diff()

# Re-check Stationarity for Differenced Log-Transformed Data
print("\nStationarity Check for Differenced Log-Transformed Data:")
adf_test(data['log_yield_diff'], title='Differenced Log G-Sec Yield')
adf_test(data['log_fii_diff'], title='Differenced Log FII Flows')

from statsmodels.tsa.stattools import grangercausalitytests

# Prepare data for Granger causality test
causality_data = pd.DataFrame({
    'log_yield_diff': data['log_yield_diff'].dropna(),
    'log_fii': data['log_fii'][1:]  # Adjust to align with differenced data
}).dropna()

# Perform Granger Causality Tests
max_lag = 12  # Test up to 12 lags
print("\nGranger Causality Test Results:")
granger_results = grangercausalitytests(causality_data, max_lag, verbose=True)

# Holding Var up until further notice to sort issues of stationarity and Granger-Causality.