"""
Transform NIFTY minute data and visualize outcomes.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load NIFTY data
nifty_df = pd.read_csv('/Users/nehapriya/Downloads/nifty_underlying_minute.csv')
nifty_df['timestamp'] = pd.to_datetime(nifty_df['timestamp'])
nifty_df = nifty_df.sort_values('timestamp').reset_index(drop=True)

# Extract date and create day index
nifty_df['date'] = nifty_df['timestamp'].dt.date
unique_dates = nifty_df['date'].unique()
date_to_day = {date: idx for idx, date in enumerate(unique_dates)}
nifty_df['day'] = nifty_df['date'].map(date_to_day)

# Create intraday minute index
nifty_df['intraday'] = nifty_df.groupby('day').cumcount()

# Calculate volatility
nifty_df['price'] = nifty_df['close']
nifty_df['volatility'] = np.abs(nifty_df['price'].diff()) / nifty_df['price'].shift(1)
nifty_df['volatility'] = nifty_df['volatility'].fillna(0.0)

# Create transformed dataframe
transformed_df = nifty_df[['day', 'intraday', 'price', 'volatility']].copy()

# Save transformed data
output_path = '/Users/nehapriya/desktop/research_project/data/raw/nifty_transformed.csv'
transformed_df.to_csv(output_path, index=False)
print(f"Transformed data saved to: {output_path}")
print(f"Shape: {transformed_df.shape}")
print(f"Days: {transformed_df['day'].nunique()}")

# Create output directory for figures
figures_dir = '/Users/nehapriya/desktop/research_project/data/processed'
os.makedirs(figures_dir, exist_ok=True)

# Figure 1: Price evolution over time (sample days)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: First 5 days price evolution
ax1 = axes[0, 0]
for day in range(min(5, transformed_df['day'].nunique())):
    day_data = transformed_df[transformed_df['day'] == day]
    ax1.plot(day_data['intraday'], day_data['price'], label=f'Day {day}', alpha=0.8)
ax1.set_xlabel('Intraday Minute')
ax1.set_ylabel('Price')
ax1.set_title('Price Evolution - First 5 Days')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Volatility distribution
ax2 = axes[0, 1]
volatility_data = transformed_df['volatility'].values
ax2.hist(volatility_data, bins=100, edgecolor='black', alpha=0.7, color='steelblue')
ax2.set_xlabel('Volatility')
ax2.set_ylabel('Frequency')
ax2.set_title('Volatility Distribution')
ax2.axvline(np.mean(volatility_data), color='red', linestyle='--', label=f'Mean: {np.mean(volatility_data):.6f}')
ax2.axvline(np.median(volatility_data), color='orange', linestyle='--', label=f'Median: {np.median(volatility_data):.6f}')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Average intraday volatility pattern
ax3 = axes[1, 0]
avg_volatility_by_minute = transformed_df.groupby('intraday')['volatility'].mean()
ax3.plot(avg_volatility_by_minute.index, avg_volatility_by_minute.values, color='darkgreen', linewidth=1.5)
ax3.set_xlabel('Intraday Minute')
ax3.set_ylabel('Average Volatility')
ax3.set_title('Average Intraday Volatility Pattern')
ax3.grid(True, alpha=0.3)

# Plot 4: Daily volatility summary
ax4 = axes[1, 1]
daily_stats = transformed_df.groupby('day')['volatility'].agg(['mean', 'std', 'max'])
ax4.plot(daily_stats.index, daily_stats['mean'], label='Mean', color='blue', alpha=0.7)
ax4.fill_between(daily_stats.index, 
                  daily_stats['mean'] - daily_stats['std'],
                  daily_stats['mean'] + daily_stats['std'],
                  alpha=0.3, color='blue', label='±1 Std')
ax4.set_xlabel('Day')
ax4.set_ylabel('Volatility')
ax4.set_title('Daily Volatility Over Time')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'nifty_analysis_1.png'), dpi=150, bbox_inches='tight')
print(f"Figure 1 saved to: {os.path.join(figures_dir, 'nifty_analysis_1.png')}")

# Figure 2: Additional analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Price distribution
ax1 = axes[0, 0]
ax1.hist(transformed_df['price'], bins=50, edgecolor='black', alpha=0.7, color='coral')
ax1.set_xlabel('Price')
ax1.set_ylabel('Frequency')
ax1.set_title('Overall Price Distribution')
ax1.axvline(transformed_df['price'].mean(), color='red', linestyle='--', 
            label=f'Mean: {transformed_df["price"].mean():.2f}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Volatility heatmap by time of day
ax2 = axes[0, 1]
# Resample to 15-minute bins
transformed_df['time_bin'] = (transformed_df['intraday'] // 15) * 15
pivot_data = transformed_df.pivot_table(values='volatility', 
                                         index='day', 
                                         columns='time_bin', 
                                         aggfunc='mean')
# Show first 20 days for clarity
im = ax2.imshow(pivot_data.iloc[:20].values, aspect='auto', cmap='YlOrRd')
ax2.set_xlabel('Time Bin (15-min intervals)')
ax2.set_ylabel('Day')
ax2.set_title('Volatility Heatmap (First 20 Days)')
plt.colorbar(im, ax=ax2, label='Volatility')

# Plot 3: Cumulative returns
ax3 = axes[1, 0]
for day in range(min(5, transformed_df['day'].nunique())):
    day_data = transformed_df[transformed_df['day'] == day].copy()
    day_data['cum_return'] = (day_data['price'] / day_data['price'].iloc[0] - 1) * 100
    ax3.plot(day_data['intraday'], day_data['cum_return'], label=f'Day {day}', alpha=0.8)
ax3.set_xlabel('Intraday Minute')
ax3.set_ylabel('Cumulative Return (%)')
ax3.set_title('Cumulative Intraday Returns - First 5 Days')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)

# Plot 4: Volatility percentiles by minute
ax4 = axes[1, 1]
percentiles = [10, 25, 50, 75, 90]
colors = plt.cm.viridis(np.linspace(0, 1, len(percentiles)))
for p, c in zip(percentiles, colors):
    vol_p = transformed_df.groupby('intraday')['volatility'].quantile(p/100)
    ax4.plot(vol_p.index, vol_p.values, label=f'{p}th percentile', color=c, alpha=0.8)
ax4.set_xlabel('Intraday Minute')
ax4.set_ylabel('Volatility')
ax4.set_title('Volatility Percentiles by Minute')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'nifty_analysis_2.png'), dpi=150, bbox_inches='tight')
print(f"Figure 2 saved to: {os.path.join(figures_dir, 'nifty_analysis_2.png')}")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"\nPrice Statistics:")
print(f"  Min:    {transformed_df['price'].min():.2f}")
print(f"  Max:    {transformed_df['price'].max():.2f}")
print(f"  Mean:   {transformed_df['price'].mean():.2f}")
print(f"  Std:    {transformed_df['price'].std():.2f}")

print(f"\nVolatility Statistics:")
print(f"  Min:    {transformed_df['volatility'].min():.8f}")
print(f"  Max:    {transformed_df['volatility'].max():.8f}")
print(f"  Mean:   {transformed_df['volatility'].mean():.8f}")
print(f"  Median: {transformed_df['volatility'].median():.8f}")
print(f"  Std:    {transformed_df['volatility'].std():.8f}")

print(f"\nData Summary:")
print(f"  Total observations: {len(transformed_df)}")
print(f"  Number of days:     {transformed_df['day'].nunique()}")
print(f"  Minutes per day:    {transformed_df.groupby('day')['intraday'].max().min()} - {transformed_df.groupby('day')['intraday'].max().max()}")

print("\nFirst 15 rows of transformed data:")
print(transformed_df.head(15).to_string(index=False))
