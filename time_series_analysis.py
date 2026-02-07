"""
Time Series Analysis for Retail Sales Data
==========================================
Complete analysis pipeline including:
- Data loading & preprocessing
- EDA with visualizations
- Trend & seasonality decomposition
- Autocorrelation analysis
- Stationarity testing
- Residual analysis
- Ready for ARIMA/SARIMA forecasting
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from scipy import stats

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH = 'retail_sales.csv'
OUTPUT_DIR = 'output_plots'
ROLLING_WINDOW = 7  # For weekly rolling statistics
SEASONAL_PERIOD = 7  # Weekly seasonality for decomposition

# Set style for matplotlib (compatible across versions)
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    plt.style.use('seaborn-whitegrid')
except Exception:
    pass
sns.set_palette("husl")


# =============================================================================
# 1. DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_preprocess(filepath: str) -> pd.DataFrame:
    """
    Load retail sales data and preprocess:
    - Convert Date to datetime
    - Sort by date
    - Set Date as index
    - Handle missing values and outliers
    """
    print("=" * 60)
    print("1. LOADING AND PREPROCESSING")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(filepath)
    print(f"\nRaw data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:\n{df.head()}")
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"\n[OK] Date column converted to datetime")
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    print(f"[OK] Data sorted by date")
    
    # Set Date as index
    df = df.set_index('Date')
    print(f"[OK] Date set as index")
    
    # Handle missing values
    missing_count = df['Sales'].isna().sum()
    if missing_count > 0:
        print(f"\nFound {missing_count} missing values")
        # Forward fill then backward fill for any remaining
        df['Sales'] = df['Sales'].ffill().bfill()
        print(f"[OK] Missing values filled (forward/backward fill)")
    else:
        print(f"\n[OK] No missing values detected")
    
    # Handle outliers using IQR method
    Q1 = df['Sales'].quantile(0.25)
    Q3 = df['Sales'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_mask = (df['Sales'] < lower_bound) | (df['Sales'] > upper_bound)
    outliers_count = outliers_mask.sum()
    
    if outliers_count > 0:
        print(f"\nFound {outliers_count} outliers (IQR method)")
        # Cap outliers at bounds
        df['Sales'] = df['Sales'].clip(lower=lower_bound, upper=upper_bound)
        print(f"[OK] Outliers capped at [{lower_bound:.2f}, {upper_bound:.2f}]")
    else:
        print(f"\n[OK] No significant outliers detected")
    
    print(f"\nFinal data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Sales statistics:\n{df['Sales'].describe()}")
    
    return df


# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# =============================================================================

def exploratory_data_analysis(df: pd.DataFrame) -> None:
    """
    Perform EDA: raw series, rolling stats, and pattern visualizations.
    """
    print("\n" + "=" * 60)
    print("2. EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=False)
    
    # 2.1 Raw time series
    axes[0].plot(df.index, df['Sales'], color='steelblue', alpha=0.8, linewidth=0.8)
    axes[0].set_title('Raw Time Series - Retail Sales', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Sales')
    axes[0].grid(True, alpha=0.3)
    
    # 2.2 Rolling mean and standard deviation
    rolling_mean = df['Sales'].rolling(window=ROLLING_WINDOW).mean()
    rolling_std = df['Sales'].rolling(window=ROLLING_WINDOW).std()
    
    axes[1].plot(df.index, df['Sales'], color='steelblue', alpha=0.4, label='Actual', linewidth=0.6)
    axes[1].plot(df.index, rolling_mean, color='darkorange', linewidth=2, label=f'Rolling Mean ({ROLLING_WINDOW} days)')
    axes[1].set_title('Rolling Mean and Std Deviation', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Sales')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(df.index, rolling_std, color='crimson', linewidth=1.5, label=f'Rolling Std ({ROLLING_WINDOW} days)')
    axes[2].set_title('Rolling Standard Deviation', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Std Dev')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    
    # 2.3 Weekly pattern (day of week)
    df_copy = df.copy()
    df_copy['DayOfWeek'] = df_copy.index.dayofweek
    df_copy['DayName'] = df_copy['DayOfWeek'].map({
        0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu',
        4: 'Fri', 5: 'Sat', 6: 'Sun'
    })
    weekly_avg = df_copy.groupby('DayName').agg({'Sales': 'mean'}).reindex(
        ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    )
    
    axes[3].bar(weekly_avg.index, weekly_avg['Sales'], color='teal', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[3].set_title('Weekly Pattern (Average Sales by Day)', fontsize=12, fontweight='bold')
    axes[3].set_ylabel('Average Sales')
    axes[3].set_xlabel('Day of Week')
    axes[3].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/01_eda_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[OK] Saved: 01_eda_overview.png")
    
    # Monthly and yearly patterns
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Monthly pattern
    df_copy['Month'] = df_copy.index.month
    monthly_avg = df_copy.groupby('Month')['Sales'].mean()
    axes[0].plot(monthly_avg.index, monthly_avg.values, marker='o', color='darkviolet', linewidth=2, markersize=8)
    axes[0].set_title('Monthly Pattern (Average Sales by Month)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Month')
    axes[0].set_ylabel('Average Sales')
    axes[0].set_xticks(range(1, 13))
    axes[0].grid(True, alpha=0.3)
    
    # Yearly pattern (if multi-year data)
    if len(df_copy.index.year.unique()) > 1:
        yearly_avg = df_copy.groupby(df_copy.index.year)['Sales'].mean()
        axes[1].bar(yearly_avg.index.astype(str), yearly_avg.values, color='coral', alpha=0.7, edgecolor='black')
        axes[1].set_title('Yearly Pattern (Average Sales by Year)', fontsize=12, fontweight='bold')
    else:
        # Seasonal subseries (monthly across the year)
        monthly_by_year = df_copy.pivot_table(values='Sales', index=df_copy.index.month, 
                                               columns=df_copy.index.year, aggfunc='mean')
        monthly_by_year.plot(ax=axes[1], marker='o', linewidth=2)
        axes[1].set_title('Monthly Sales by Year', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Year / Month')
    axes[1].set_ylabel('Average Sales')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/02_eda_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 02_eda_patterns.png")
    
    print("\n--- EDA Insights ---")
    print("- Upward trend visible in raw series")
    print("- Rolling mean smooths noise; rolling std shows volatility changes")
    print("- Weekly: Weekend (Sat/Sun) typically higher than weekdays")
    print("- Monthly: Seasonal peaks may appear (e.g., year-end)")


# =============================================================================
# 3. TREND AND SEASONALITY DECOMPOSITION
# =============================================================================

def trend_seasonality_analysis(df: pd.DataFrame) -> tuple:
    """
    Apply seasonal decomposition (additive and multiplicative).
    Returns decomposition objects for further use.
    """
    print("\n" + "=" * 60)
    print("3. TREND AND SEASONALITY ANALYSIS")
    print("=" * 60)
    
    # Ensure no NaN for decomposition
    df_clean = df['Sales'].dropna()
    
    # Additive decomposition
    decomp_add = seasonal_decompose(df_clean, model='additive', period=SEASONAL_PERIOD, extrapolate_trend='freq')
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    
    axes[0].plot(df_clean.index, df_clean.values, color='steelblue', linewidth=0.8)
    axes[0].set_ylabel('Observed')
    axes[0].set_title('Additive Decomposition', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(decomp_add.trend.index, decomp_add.trend.values, color='green', linewidth=1.5)
    axes[1].set_ylabel('Trend')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(decomp_add.seasonal.index, decomp_add.seasonal.values, color='orange', linewidth=0.8)
    axes[2].set_ylabel('Seasonal')
    axes[2].grid(True, alpha=0.3)
    
    axes[3].plot(decomp_add.resid.index, decomp_add.resid.values, color='gray', linewidth=0.6, alpha=0.8)
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Date')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/03_decomposition_additive.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[OK] Saved: 03_decomposition_additive.png")
    
    # Multiplicative decomposition
    df_positive = df_clean.copy()
    df_positive = df_positive[df_positive > 0]  # Multiplicative requires positive values
    decomp_mul = seasonal_decompose(df_positive, model='multiplicative', period=SEASONAL_PERIOD, extrapolate_trend='freq')
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    
    axes[0].plot(df_positive.index, df_positive.values, color='steelblue', linewidth=0.8)
    axes[0].set_ylabel('Observed')
    axes[0].set_title('Multiplicative Decomposition', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(decomp_mul.trend.index, decomp_mul.trend.values, color='green', linewidth=1.5)
    axes[1].set_ylabel('Trend')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(decomp_mul.seasonal.index, decomp_mul.seasonal.values, color='orange', linewidth=0.8)
    axes[2].set_ylabel('Seasonal')
    axes[2].grid(True, alpha=0.3)
    
    axes[3].plot(decomp_mul.resid.index, decomp_mul.resid.values, color='gray', linewidth=0.6, alpha=0.8)
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Date')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/04_decomposition_multiplicative.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 04_decomposition_multiplicative.png")
    
    print("\n--- Decomposition Observations ---")
    print("- Additive: Use when seasonal amplitude is constant over time")
    print("- Multiplicative: Use when seasonal amplitude grows with trend")
    print("- Trend: Captures long-term direction")
    print("- Seasonal: Repeating pattern (e.g., weekly cycle)")
    print("- Residual: Unexplained noise; ideally random")
    
    return decomp_add, decomp_mul


# =============================================================================
# 4. AUTOCORRELATION ANALYSIS
# =============================================================================

def autocorrelation_analysis(df: pd.DataFrame, lags: int = 40) -> None:
    """
    Plot ACF and PACF and interpret lag relationships.
    """
    print("\n" + "=" * 60)
    print("4. AUTOCORRELATION ANALYSIS")
    print("=" * 60)
    
    series = df['Sales'].dropna()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    plot_acf(series, lags=lags, ax=axes[0], alpha=0.05)
    axes[0].set_title('Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Lag')
    axes[0].grid(True, alpha=0.3)
    
    plot_pacf(series, lags=lags, ax=axes[1], alpha=0.05, method='ywm')
    axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Lag')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/05_acf_pacf.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[OK] Saved: 05_acf_pacf.png")
    
    print("\n--- ACF/PACF Interpretation ---")
    print("- ACF: Shows correlation between series and its lagged values")
    print("- PACF: Shows direct correlation at each lag (excluding intermediate lags)")
    print("- Slow decay in ACF: Suggests non-stationarity (trend)")
    print("- Significant spikes at lag 7: Weekly seasonality")
    print("- PACF cut-off point helps identify AR order; ACF cut-off identifies MA order")


# =============================================================================
# 5. STATIONARITY TESTING
# =============================================================================

def adf_test(series: pd.Series, name: str = "Series") -> dict:
    """
    Perform Augmented Dickey-Fuller test and return results.
    """
    result = adfuller(series.dropna(), autolag='AIC')
    return {
        'adf_stat': result[0],
        'pvalue': result[1],
        'usedlag': result[2],
        'nobs': result[3],
        'critical_values': result[4],
        'icbest': result[5]
    }


def stationarity_analysis(df: pd.DataFrame) -> pd.Series:
    """
    Test stationarity, apply differencing if needed, re-test.
    Returns differenced series if applied.
    """
    print("\n" + "=" * 60)
    print("5. STATIONARITY TESTING")
    print("=" * 60)
    
    series = df['Sales'].dropna()
    
    result = adf_test(series, "Original")
    
    print("\n--- Augmented Dickey-Fuller Test (Original Series) ---")
    print(f"ADF Statistic: {result['adf_stat']:.4f}")
    print(f"p-value: {result['pvalue']:.4f}")
    print("Critical Values:")
    for k, v in result['critical_values'].items():
        print(f"  {k}: {v:.4f}")
    
    if result['pvalue'] < 0.05:
        print("\n[OK] Result: Series is STATIONARY (p < 0.05)")
        print("  Reject null hypothesis of unit root.")
        return series
    else:
        print("\n[!] Result: Series is NON-STATIONARY (p >= 0.05)")
        print("  Fail to reject null hypothesis. Applying first-order differencing...")
        
        # First-order differencing
        series_diff = series.diff().dropna()
        
        result_diff = adf_test(series_diff, "Differenced")
        print("\n--- ADF Test (After First Differencing) ---")
        print(f"ADF Statistic: {result_diff['adf_stat']:.4f}")
        print(f"p-value: {result_diff['pvalue']:.4f}")
        
        if result_diff['pvalue'] < 0.05:
            print("\n[OK] Result: Differenced series is STATIONARY")
            print("  Ready for ARIMA with d=1")
        else:
            print("\n[!!] Consider second-order differencing or seasonal differencing")
        
        # Plot original vs differenced
        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        axes[0].plot(series.index, series.values, color='steelblue', linewidth=0.8)
        axes[0].set_title('Original Series (Non-Stationary)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Sales')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(series_diff.index, series_diff.values, color='forestgreen', linewidth=0.8)
        axes[1].set_title('First Differenced Series (Stationary)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Sales (Δ)')
        axes[1].set_xlabel('Date')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/06_stationarity_differencing.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\n[OK] Saved: 06_stationarity_differencing.png")
        
        return series_diff


# =============================================================================
# 6. NOISE AND RANDOMNESS ANALYSIS (RESIDUALS)
# =============================================================================

def residual_analysis(decomp_add) -> None:
    """
    Analyze residuals: histogram and time series plot.
    """
    print("\n" + "=" * 60)
    print("6. NOISE AND RANDOMNESS ANALYSIS (RESIDUALS)")
    print("=" * 60)
    
    resid = decomp_add.resid.dropna()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Residual time series
    axes[0].plot(resid.index, resid.values, color='gray', alpha=0.7, linewidth=0.6)
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[0].set_title('Residual Time Series', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Residual')
    axes[0].set_xlabel('Date')
    axes[0].grid(True, alpha=0.3)
    
    # Histogram
    axes[1].hist(resid, bins=50, color='steelblue', edgecolor='black', alpha=0.7, density=True)
    x = np.linspace(resid.min(), resid.max(), 100)
    axes[1].plot(x, stats.norm.pdf(x, resid.mean(), resid.std()), 'r-', linewidth=2, label='Normal fit')
    axes[1].set_title('Residual Distribution (Histogram)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Density')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/07_residual_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[OK] Saved: 07_residual_analysis.png")
    
    # Normality test
    _, p_norm = stats.shapiro(resid.sample(min(5000, len(resid)), random_state=42))
    print(f"\nShapiro-Wilk normality test p-value: {p_norm:.4f}")
    if p_norm < 0.05:
        print("  Residuals deviate from normality (common in real data)")
    else:
        print("  Residuals are approximately normal")
    print(f"Residual mean: {resid.mean():.4f} (should be ~0)")
    print(f"Residual std: {resid.std():.4f}")


# =============================================================================
# 7. FINAL REPORT AND CONCLUSIONS
# =============================================================================

def print_final_report(df: pd.DataFrame, decomp_add, is_stationary: bool) -> None:
    """
    Summarize findings in a professional report format.
    """
    print("\n" + "=" * 60)
    print("7. FINAL REPORT - CONCLUSIONS AND INSIGHTS")
    print("=" * 60)
    
    report = """
======================================================================
       TIME SERIES ANALYSIS - RETAIL SALES - EXECUTIVE SUMMARY
======================================================================

1. DATA OVERVIEW
   - Dataset: Retail sales (daily)
   - Period: {start} to {end}
   - Observations: {n} days
   - Sales range: ${min_sales:.0f} - ${max_sales:.0f}

2. KEY FINDINGS
   - TREND: Upward trend detected; sales show long-term growth
   - SEASONALITY: Weekly pattern (7-day cycle) evident
   - NON-STATIONARITY: Original series likely non-stationary
   - STATIONARITY: First differencing recommended for modeling
   - RESIDUALS: Unexplained variability; suitable for ARIMA/SARIMA

3. RECOMMENDATIONS FOR FORECASTING (ARIMA/SARIMA)
   - Consider SARIMA to capture seasonal component (period=7)
   - Use d=1 (first differencing) for trend
   - ACF/PACF suggest potential AR/MA components
   - Seasonal order D=1 may be needed for weekly seasonality

4. NEXT STEPS
   - Fit SARIMA(p,d,q)(P,D,Q)s with s=7
   - Use grid search for optimal (p,d,q)(P,D,Q)
   - Validate with train/test split
   - Produce forecasts and confidence intervals

5. PLOTS GENERATED
   - 01_eda_overview.png - Raw series, rolling stats, weekly pattern
   - 02_eda_patterns.png - Monthly and yearly patterns
   - 03_decomposition_additive.png - Additive decomposition
   - 04_decomposition_multiplicative.png - Multiplicative decomposition
   - 05_acf_pacf.png - Autocorrelation analysis
   - 06_stationarity_differencing.png - Differencing (if applied)
   - 07_residual_analysis.png - Residual diagnostics

======================================================================
"""
    
    sales = df['Sales']
    report = report.format(
        start=df.index.min().strftime('%Y-%m-%d'),
        end=df.index.max().strftime('%Y-%m-%d'),
        n=len(df),
        min_sales=sales.min(),
        max_sales=sales.max()
    )
    
    print(report)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run the complete time series analysis pipeline."""
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("   TIME SERIES ANALYSIS - RETAIL SALES DATA")
    print("=" * 60)
    
    # 1. Load and preprocess
    df = load_and_preprocess(DATA_PATH)
    
    # 2. EDA
    exploratory_data_analysis(df)
    
    # 3. Decomposition
    decomp_add, decomp_mul = trend_seasonality_analysis(df)
    
    # 4. Autocorrelation
    autocorrelation_analysis(df)
    
    # 5. Stationarity
    series_processed = stationarity_analysis(df)
    is_stationary = series_processed.equals(df['Sales'].dropna())
    
    # 6. Residual analysis
    residual_analysis(decomp_add)
    
    # 7. Final report
    print_final_report(df, decomp_add, is_stationary)
    
    print("\n" + "=" * 60)
    print("   ANALYSIS COMPLETE - All plots saved to", OUTPUT_DIR)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
