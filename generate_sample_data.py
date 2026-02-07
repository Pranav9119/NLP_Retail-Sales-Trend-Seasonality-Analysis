"""
Generate sample retail sales data for Time Series Analysis.
Run this script once to create retail_sales.csv.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# Generate ~2 years of daily data
start_date = datetime(2022, 1, 1)
end_date = datetime(2023, 12, 31)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Base sales with trend (upward)
n_days = len(date_range)
trend = np.linspace(1000, 1500, n_days)

# Weekly seasonality (higher on weekends)
day_of_week = np.array([d.weekday() for d in date_range])
weekly_seasonality = np.where(day_of_week < 5, -100, 150)  # Weekends boost

# Monthly seasonality (year-end peaks)
month = np.array([d.month for d in date_range])
monthly_seasonality = 50 * np.sin(2 * np.pi * month / 12) + 30 * (month == 12)

# Random noise
noise = np.random.normal(0, 80, n_days)

# Combine components
sales = trend + weekly_seasonality + monthly_seasonality + noise
sales = np.maximum(sales, 100)  # Floor at 100

# Introduce few missing values (2%)
missing_mask = np.random.random(n_days) < 0.02
sales[missing_mask] = np.nan

# Introduce few outliers (1%)
outlier_mask = np.random.random(n_days) < 0.01
sales[outlier_mask] = sales[outlier_mask] * (2 + np.random.random(outlier_mask.sum()))

# Create DataFrame
df = pd.DataFrame({
    'Date': date_range,
    'Sales': sales
})

df.to_csv('retail_sales.csv', index=False)
print(f"Created retail_sales.csv with {len(df)} rows")
