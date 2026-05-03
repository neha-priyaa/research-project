"""
Complete Data Generation and Analysis Script

Based on: Matrix-based Prediction Approach for Intraday Instantaneous Volatility Vector
Phase 1: Generate synthetic intraday data using Geometric Brownian Motion
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.data import DataSimulator, process_volatility_matrix, save_matrix

def main():
    print("=" * 80)
    print("📊 PHASE 1: SYNTHETIC DATA GENERATION & ANALYSIS")
    print("=" * 80)
    print()

    # Initialize simulator with paper parameters
    print("🎯 STEP 1: Initialize Data Simulator")
    print("-" * 80)
    print("Paper-based Parameters:")
    print("  • Target: S&P 500 Intraday Volatility")
    print("  • Method: Geometric Brownian Motion")
    print("  • Initial Price: $100.00 (S&P 500 approximation)")
    print("  • Annual Volatility: 20% (σ = 0.20)")
    print("  • Annual Drift: 5% (μ = 0.05)")
    print()

    simulator = DataSimulator(
        initial_price=100.0,
        volatility=0.20,
        drift=0.05
    )

    # Generate full dataset
    print("🎯 STEP 2: Generate Full Synthetic Dataset")
    print("-" * 80)
    print("Data Specifications:")
    print("  • Trading Days: 252 (1 year)")
    print("  • Intraday Points: 390 (1-minute intervals)")
    print("  • Total Data Points: 98,280 (252 × 390)")
    print("  • Random Seed: 42 (reproducibility)")
    print()

    print("Generating data... This may take a moment...")
    data = simulator.generate_dataset(days=252, intraday_points=390)
    print(f"✅ Dataset generated: {len(data):,} data points")
    print()

    # Statistical analysis
    print("🎯 STEP 3: Statistical Analysis of Generated Data")
    print("-" * 80)

    print("Price Statistics:")
    print(f"  • Min Price: ${data['price'].min():.2f}")
    print(f"  • Max Price: ${data['price'].max():.2f}")
    print(f"  • Mean Price: ${data['price'].mean():.2f}")
    print(f"  • Std Deviation: ${data['price'].std():.2f}")
    print(f"  • Price Range: ${data['price'].max() - data['price'].min():.2f}")
    print()

    print("Volatility Statistics:")
    print(f"  • Min Volatility: {data['volatility'].min():.6f}")
    print(f"  • Max Volatility: {data['volatility'].max():.6f}")
    print(f"  • Mean Volatility: {data['volatility'].mean():.6f}")
    print(f"  • Std Volatility: {data['volatility'].std():.6f}")
    print(f"  • Volatility Range: {data['volatility'].max() - data['volatility'].min():.6f}")
    print()

    # Process volatility matrix
    print("🎯 STEP 4: Construct Volatility Matrix")
    print("-" * 80)
    print("Matrix Construction:")
    print("  • Dimensions: (252 days × 390 intraday points)")
    print("  • Normalization: Min-max scaling [0,1]")
    print()

    raw_matrix, normalized_matrix = process_volatility_matrix(data, normalize=True)
    print(f"✅ Volatility matrices constructed")
    print(f"  • Raw Matrix Shape: {raw_matrix.shape}")
    print(f"  • Normalized Matrix Shape: {normalized_matrix.shape}")
    print(f"  • Raw Range: {raw_matrix.min():.6f} to {raw_matrix.max():.6f}")
    print(f"  • Normalized Range: {normalized_matrix.min():.6f} to {normalized_matrix.max():.6f}")
    print()

    # Matrix analysis
    print("🎯 STEP 5: Volatility Matrix Analysis")
    print("-" * 80)

    print("Daily Volatility Patterns (First 5 days):")
    for i in range(5):
        daily_vol = raw_matrix[i]
        print(f"  • Day {i+1}:")
        print(f"    - Min: {daily_vol.min():.6f}")
        print(f"    - Max: {daily_vol.max():.6f}")
        print(f"    - Mean: {daily_vol.mean():.6f}")
        print(f"    - Std: {daily_vol.std():.6f}")
    print()

    print("Intraday Volatility Patterns (First 10 minutes):")
    for j in range(10):
        intraday_vol = normalized_matrix[:, j]
        print(f"  • Minute {j+1}:")
        print(f"    - Min: {intraday_vol.min():.6f}")
        print(f"    - Max: {intraday_vol.max():.6f}")
        print(f"    - Mean: {intraday_vol.mean():.6f}")
    print()

    # Save data and matrices
    print("🎯 STEP 6: Save Data and Matrices")
    print("-" * 80)

    # Save full dataset
    data.to_csv('data/raw/synthetic_dataset_full.csv', index=False)
    print(f"✅ Full dataset saved: data/raw/synthetic_dataset_full.csv")

    # Save matrices
    save_matrix(raw_matrix, 'data/processed/volatility_matrix_full_raw.csv')
    save_matrix(normalized_matrix, 'data/processed/volatility_matrix_full_normalized.csv')
    print(f"✅ Volatility matrices saved: data/processed/")
    print()

    # Generate visualizations
    print("🎯 STEP 7: Generate Visualizations")
    print("-" * 80)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Synthetic Intraday Volatility Analysis', fontsize=16, fontweight='bold')

    # 1. Price Path (First 5 days)
    ax1 = axes[0, 0]
    for day in range(5):
        day_data = data[data['day'] == day]
        ax1.plot(day_data['intraday'], day_data['price'], label=f'Day {day+1}', alpha=0.7)
    ax1.set_title('Price Paths (First 5 Days)', fontweight='bold')
    ax1.set_xlabel('Intraday Point (1-minute intervals)')
    ax1.set_ylabel('Price ($)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Volatility Patterns (First 5 days)
    ax2 = axes[0, 1]
    for day in range(5):
        day_data = data[data['day'] == day]
        ax2.plot(day_data['intraday'], day_data['volatility'], label=f'Day {day+1}', alpha=0.7)
    ax2.set_title('Volatility Patterns (First 5 Days)', fontweight='bold')
    ax2.set_xlabel('Intraday Point (1-minute intervals)')
    ax2.set_ylabel('Volatility')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Price Distribution
    ax3 = axes[0, 2]
    ax3.hist(data['price'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax3.set_title('Price Distribution', fontweight='bold')
    ax3.set_xlabel('Price ($)')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Volatility Distribution
    ax4 = axes[1, 0]
    ax4.hist(data['volatility'], bins=50, edgecolor='black', alpha=0.7, color='lightcoral')
    ax4.set_title('Volatility Distribution', fontweight='bold')
    ax4.set_xlabel('Volatility')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Volatility Matrix Heatmap (First 20 days, first 100 minutes)
    ax5 = axes[1, 1]
    heatmap_data = normalized_matrix[:20, :100]  # First 20 days, first 100 minutes
    im = ax5.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax5.set_title('Volatility Matrix Heatmap (First 20 Days)', fontweight='bold')
    ax5.set_xlabel('Intraday Point')
    ax5.set_ylabel('Day')
    plt.colorbar(im, ax=ax5, label='Normalized Volatility')

    # 6. Daily Volatility Summary
    ax6 = axes[1, 2]
    daily_mean_vol = np.mean(raw_matrix, axis=1)
    daily_std_vol = np.std(raw_matrix, axis=1)
    ax6.plot(range(len(daily_mean_vol)), daily_mean_vol, label='Mean Volatility', color='blue', linewidth=1.5)
    ax6.fill_between(range(len(daily_mean_vol)),
                    daily_mean_vol - daily_std_vol,
                    daily_mean_vol + daily_std_vol,
                    alpha=0.3, label='±1 Std Dev', color='blue')
    ax6.set_title('Daily Volatility Summary', fontweight='bold')
    ax6.set_xlabel('Trading Day')
    ax6.set_ylabel('Volatility')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save visualization
    viz_path = 'data/processed/volatility_analysis.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved: {viz_path}")
    plt.close()
    print()

    # Final summary
    print("=" * 80)
    print("🎉 DATA GENERATION & ANALYSIS COMPLETE!")
    print("=" * 80)
    print()
    print("📊 Summary:")
    print(f"  • Total Data Points Generated: {len(data):,}")
    print(f"  • Volatility Matrix Dimensions: {normalized_matrix.shape}")
    print(f"  • Files Created: 4 (dataset, matrices, visualization)")
    print(f"  • Processing Time: Complete")
    print()
    print("✅ Phase 1 - Data Generation: SUCCESSFUL")
    print("✅ Ready for Phase 2 - TIP-PCA Implementation")
    print()
    print("📁 Generated Files:")
    print("  • data/raw/synthetic_dataset_full.csv")
    print("  • data/processed/volatility_matrix_full_raw.csv")
    print("  • data/processed/volatility_matrix_full_normalized.csv")
    print("  • data/processed/volatility_analysis.png")
    print()

if __name__ == "__main__":
    main()
