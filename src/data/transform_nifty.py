"""
Transform NIFTY minute data to match synthetic dataset format.
"""
import pandas as pd
import numpy as np

# Load NIFTY data
nifty_df = pd.read_csv('/Users/nehapriya/Downloads/nifty_underlying_minute.csv')
nifty_df['timestamp'] = pd.to_datetime(nifty_df['timestamp'])

# Sort by timestamp
nifty_df = nifty_df.sort_values('timestamp').reset_index(drop=True)

# Extract date and time components
nifty_df['date'] = nifty_df['timestamp'].dt.date

# Create day index (0-indexed from first day)
unique_dates = nifty_df['date'].unique()
date_to_day = {date: idx for idx, date in enumerate(unique_dates)}
nifty_df['day'] = nifty_df['date'].map(date_to_day)

# Create intraday minute index (0-indexed within each day)
nifty_df['intraday'] = nifty_df.groupby('day').cumcount()

# Calculate volatility as absolute return: |price[t] - price[t-1]| / price[t-1]
nifty_df['price'] = nifty_df['close']
nifty_df['volatility'] = np.abs(nifty_df['price'].diff()) / nifty_df['price'].shift(1)
nifty_df['volatility'] = nifty_df['volatility'].fillna(0.0)

# Normalize price to start at 100 for each day (similar to synthetic data)
nifty_df['price_normalized'] = nifty_df.groupby('day')['price'].transform(
    lambda x: 100 * x / x.iloc[0]
)

# Select and order columns to match synthetic format
transformed_df = nifty_df[['day', 'intraday', 'price_normalized', 'volatility']].copy()
transformed_df.columns = ['day', 'intraday', 'price', 'volatility']

# Save transformed data
output_path = '/Users/nehapriya/desktop/research_project/data/raw/nifty_transformed.csv'
transformed_df.to_csv(output_path, index=False)

print(f"Transformed data saved to: {output_path}")
print(f"\nShape: {transformed_df.shape}")
print(f"Days: {transformed_df['day'].nunique()}")
print(f"Minutes per day range: {transformed_df.groupby('day')['intraday'].max().min()} - {transformed_df.groupby('day')['intraday'].max().max()}")
print(f"\nFirst 30 rows:")
print(transformed_df.head(30).to_string(index=False))
print(f"\nLast 10 rows:")
print(transformed_df.tail(10).to_string(index=False))
