import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.vector_ar.vecm import select_order, VAR
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import jarque_bera

# Load the data
file_path = 'new_india-fii-yeild.xlsx'  # Replace with your file path
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

# Residual Diagnostics for ARIMA
from statsmodels.tsa.arima.model import ARIMA

# ARIMA for FII
fii_model = ARIMA(data['log_diff2_fii'], order=(1, 0, 1)).fit()
data['fii_residuals'] = fii_model.resid

# ARIMA for Yield
yield_model = ARIMA(data['log_diff_yield'], order=(1, 0, 1)).fit()
data['yield_residuals'] = yield_model.resid

# Residual Diagnostics
def residual_diagnostics(series, series_name):
    print(f"\nResidual Diagnostics for {series_name}:")

    # Ljung-Box Test
    ljung_box = acorr_ljungbox(series, lags=[10], return_df=True)
    print(f"Ljung-Box Test p-value: {ljung_box['lb_pvalue'].iloc[0]:.4f}")

    # Jarque-Bera Test
    stat, p = jarque_bera(series)
    print(f"Jarque-Bera Test: Statistic={stat:.4f}, p-value={p:.4f}")
    print("Normally Distributed" if p > 0.05 else "Not Normally Distributed")

# Apply diagnostics
residual_diagnostics(data['fii_residuals'], 'FII')
residual_diagnostics(data['yield_residuals'], 'Yield')

# Granger Causality Tests
def granger_tests(data, max_lag):
    print("\nGranger Causality Tests:")
    results = grangercausalitytests(data, max_lag, verbose=False)
    for lag, result in results.items():
        f_test_pvalue = result[0]['ssr_ftest'][1]
        print(f"Lag {lag}: p-value={f_test_pvalue:.4f}")

# Check causality: Does Yield influence FII?
granger_tests(data[['log_diff2_fii', 'log_diff_yield']], max_lag=5)

# VAR Model Selection and Fitting
var_selection = select_order(data[['log_diff2_fii', 'log_diff_yield']], maxlags=15, deterministic="ci")
print(f"\nVAR Model Order Selection:\n{var_selection.summary()}")

# Fit the VAR Model
var_model = VAR(data[['log_diff2_fii', 'log_diff_yield']])
var_result = var_model.fit(maxlags=var_selection.selected_orders['aic'])
print(var_result.summary())

# Impulse Response Functions (IRF)
irf = var_result.irf(10)  # 10 periods
irf.plot(orth=False)
plt.show()

# Variance Decomposition
fevd = var_result.fevd(10)
print(fevd.summary())


print("Analysis complete. Review the results for causality and interactions.")