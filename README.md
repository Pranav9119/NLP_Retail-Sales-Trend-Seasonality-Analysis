# Time Series Analysis - Retail Sales Data

A complete, modular Python project for time series analysis of retail sales data. The pipeline covers data preprocessing, EDA, trend/seasonality decomposition, autocorrelation, stationarity testing, and residual analysis—ready for ARIMA/SARIMA forecasting.

## Setup

```bash
pip install -r requirements.txt
```

## Generate Sample Data

If you don't have `retail_sales.csv`, run:

```bash
python generate_sample_data.py
```

This creates a 2-year daily dataset with trend, weekly seasonality, and noise.

## Run Analysis

```bash
python time_series_analysis.py
```

## Output

- **Plots** saved in `output_plots/`:
  - `01_eda_overview.png` - Raw series, rolling mean/std, weekly pattern
  - `02_eda_patterns.png` - Monthly and yearly patterns
  - `03_decomposition_additive.png` - Additive decomposition
  - `04_decomposition_multiplicative.png` - Multiplicative decomposition
  - `05_acf_pacf.png` - ACF and PACF
  - `06_stationarity_differencing.png` - Original vs differenced series
  - `07_residual_analysis.png` - Residual diagnostics

- **Console output**: Step-by-step interpretations and a final report

## Custom Data

Place your CSV with columns `Date` and `Sales` as `retail_sales.csv`, or update `DATA_PATH` in the script.

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn, statsmodels, scipy
